"""
This is a boilerplate pipeline 'training'
generated using Kedro 0.19.14
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import train_validate_test_one_node

def create_pipeline(**_):
    return pipeline([
        node(
            func=train_validate_test_one_node,
            inputs=[
                "params:train.data_root",
                "params:train.image_size",
                "params:train.batch_size",
                "params:train.epochs",
                "params:train.backbone",
                "params:train.learning_rate",
                "params:train.augment",
                "params:train.out_model_path",
                "params:train.out_history_path",
                "params:train.out_history_plot",
                "params:train.out_cm_plot",
                "params:train.out_report_path",
                "params:train.seed",
            ],
            outputs="train_metrics",
            name="train_validate_test",
        ),
    ])
