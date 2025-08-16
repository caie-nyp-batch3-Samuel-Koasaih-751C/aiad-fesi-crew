"""
This is a boilerplate pipeline 'data_split'
generated using Kedro 0.19.14
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import stratified_split_folders

def create_pipeline(**_):
    return pipeline([
        node(
            func=stratified_split_folders,
            inputs=[
                "params:data_split.src_root",
                "params:data_split.dst_root",
                "params:data_split.train_ratio",
                "params:data_split.val_ratio",
                "params:data_split.seed",
                "params:data_split.limit_per_class",
            ],
            outputs="data_split_summary",
            name="split_processed_dataset",
        )
    ])
