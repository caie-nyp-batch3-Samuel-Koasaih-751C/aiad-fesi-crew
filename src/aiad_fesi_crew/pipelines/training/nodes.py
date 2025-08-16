"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.19.14
"""
from __future__ import annotations
import os
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report
import os
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=0")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
tf.config.optimizer.set_jit(False)

# Float32 everywhere
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy("float32")

# Gentle memory behavior
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass



from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D, Conv2D
from tensorflow.keras.applications import DenseNet121, ResNet50, EfficientNetB0
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import pickle
# Strategy: only mirror if >1 GPU
gpus = tf.config.list_physical_devices("GPU")
strategy = tf.distribute.MirroredStrategy() if len(gpus) > 1 else tf.distribute.get_strategy()

def _build_backbone(name: str, img_size: int, channels: int):
    """Return a Keras backbone (no top) + input layer."""
    input_ = Input(shape=(img_size, img_size, channels))
    # Ensure 3-channel input for imagenet backbones
    x = Conv2D(3, (3, 3), padding="same")(input_) if channels != 3 else input_

    name = name.lower()
    if name in ["densenet", "densenet121"]:
        base = DenseNet121(weights="imagenet", include_top=False)
    elif name in ["resnet50", "resnet"]:
        base = ResNet50(weights="imagenet", include_top=False)
    elif name in ["efficientnetb0", "efficientnet"]:
        base = EfficientNetB0(weights="imagenet", include_top=False)
    else:
        raise ValueError(f"Unknown backbone '{name}'")

    x = base(x)
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    x = Dense(256, activation="relu")(x)
    x = BatchNormalization()(x)
    x = Dropout(0.5)(x)
    return input_, x


def _plot_history(history: dict, out_png: Path):
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    plt.plot(history["accuracy"], label="train_acc")
    plt.plot(history["val_accuracy"], label="val_acc")
    plt.xlabel("epoch"); plt.ylabel("accuracy"); plt.legend(); plt.title("Accuracy")
    plt.subplot(1,2,2)
    plt.plot(history["loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss")
    plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.title("Loss")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, bbox_inches="tight"); plt.close()


def _plot_confusion(cm: np.ndarray, class_names: list[str], out_png: Path):
    fig, ax = plt.subplots(figsize=(10,10))
    im = ax.imshow(cm, cmap="Blues")
    fig.colorbar(im, ax=ax)
    ax.set_xticks(range(len(class_names))); ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual"); ax.set_title("Confusion Matrix")
    # annotate
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, int(cm[i, j]), ha="center", va="center")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout(); plt.savefig(out_png, bbox_inches="tight"); plt.close()


def train_validate_test_one_node(
    data_root: str,
    image_size: int,
    batch_size: int,
    epochs: int,
    backbone: str,
    learning_rate: float,
    augment: Dict[str, float],
    out_model_path: str,
    out_history_path: str,
    out_history_plot: str,
    out_cm_plot: str,
    out_report_path: str,
    seed: int = 42,
) -> Dict[str, float]:
    """
    All-in-one: build dataloaders from data_root (train/val/test), train, evaluate on test,
    save model (.h5), history (.pkl), plots (.png) and classification report (.txt).
    Returns a small metrics dict.
    """
    np.random.seed(seed)
    tf.random.set_seed(seed)

    data_root = Path(data_root)
    train_dir = data_root / "train"
    val_dir   = data_root / "val"
    test_dir  = data_root / "test"

    # ---- Data generators
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=augment.get("rotation_range", 0),
        width_shift_range=augment.get("width_shift_range", 0.0),
        height_shift_range=augment.get("height_shift_range", 0.0),
        zoom_range=augment.get("zoom_range", 0.0),
        horizontal_flip=augment.get("horizontal_flip", False),
        vertical_flip=augment.get("vertical_flip", False),
        shear_range=augment.get("shear_range", 0.0),
        brightness_range=augment.get("brightness_range", None),
    )
    val_datagen = ImageDataGenerator(rescale=1./255)
    test_datagen = ImageDataGenerator(rescale=1./255)

    train_gen = train_datagen.flow_from_directory(
        str(train_dir), target_size=(image_size, image_size),
        batch_size=batch_size, class_mode="categorical", shuffle=True, seed=seed
    )
    val_gen = val_datagen.flow_from_directory(
        str(val_dir), target_size=(image_size, image_size),
        batch_size=batch_size, class_mode="categorical", shuffle=False
    )
    test_gen = test_datagen.flow_from_directory(
        str(test_dir), target_size=(image_size, image_size),
        batch_size=batch_size, class_mode="categorical", shuffle=False
    )

    n_classes = len(train_gen.class_indices)
    channels = 3  # using RGB

    # ---- Strategy (single node uses MirroredStrategy if available)
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
    else:
        strategy = tf.distribute.get_strategy()

    with strategy.scope():
        inp, feat = _build_backbone(backbone, image_size, channels=3)
        out = Dense(n_classes, activation="softmax", name="root")(feat)
        model = Model(inp, out)
        model.compile(
            loss="categorical_crossentropy",
            optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
            metrics=["accuracy"],
        )


    # ---- Callbacks
    model_out = Path(out_model_path)
    model_out.parent.mkdir(parents=True, exist_ok=True)
    callbacks = [
        ModelCheckpoint(str(model_out), monitor="val_accuracy", save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor="val_accuracy", factor=0.5, patience=5, min_lr=1e-5, verbose=1),
        EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1),
    ]

    # ---- Train
    hist = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        steps_per_epoch=len(train_gen),
        callbacks=callbacks,
        verbose=1,
    )

    # ---- Save history (pickle) & plot
    history_out = Path(out_history_path); history_out.parent.mkdir(parents=True, exist_ok=True)
    with open(history_out, "wb") as f:
        pickle.dump(hist.history, f)
    _plot_history(hist.history, Path(out_history_plot))

    # ---- Test evaluation
    test_loss, test_acc = model.evaluate(test_gen, verbose=0)

    # Confusion matrix + report
    y_prob = model.predict(test_gen, verbose=0)
    y_pred = np.argmax(y_prob, axis=1)
    y_true = test_gen.classes
    class_names = list(test_gen.class_indices.keys())
    cm = confusion_matrix(y_true, y_pred)
    _plot_confusion(cm, class_names, Path(out_cm_plot))

    report = classification_report(y_true, y_pred, target_names=class_names)
    report_out = Path(out_report_path); report_out.parent.mkdir(parents=True, exist_ok=True)
    with open(report_out, "w") as f:
        f.write(report)

    # ---- Return short metrics for Kedro logs
    return {
        "val_best_acc": float(max(hist.history.get("val_accuracy", [0.0]))),
        "test_acc": float(test_acc),
        "test_loss": float(test_loss),
        "n_classes": float(n_classes),
        "model_path": str(model_out),
    }
