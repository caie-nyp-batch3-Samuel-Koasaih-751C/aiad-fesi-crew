from __future__ import annotations
from pathlib import Path
import os
from flask import Flask
from .routes import bp as ui_bp

def create_app() -> Flask:
    here = Path(__file__).parent

    app = Flask(
        __name__,
        static_folder=str(here / "static"),
        template_folder=str(here / "templates"),
    )

    # upload/body size limit (10MB)
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

    # Where to load the Keras model from (mounted in docker-compose)
    app.config["MODEL_PATH"] = os.environ.get(
        "MODEL_PATH", "/project/data/06_models/model.h5"
    )

    # lazy model loader (Flask 3 safe – no before_first_request)
    _model = {"obj": None}

    def ensure_model():
        if _model["obj"] is None:
            # delay heavy imports until needed
            from tensorflow.keras.models import load_model
            mp = app.config["MODEL_PATH"]
            _model["obj"] = load_model(mp)
        return _model["obj"]

    app.ensure_model = ensure_model  # type: ignore[attr-defined]

    # register blueprint (all routes are under this)
    app.register_blueprint(ui_bp)

    return app
