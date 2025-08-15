"""
This is a boilerplate pipeline 'mask_merge'
generated using Kedro 0.19.14
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import apply_folder_masks

def create_pipeline(**_):
    return pipeline([
        node(
            func=apply_folder_masks,
            inputs=[
                "params:mask_merge.input_folder",
                "params:mask_merge.mask_folder",
                "params:mask_merge.output_folder",
                "params:mask_merge.valid_exts",
            ],
            outputs="mask_merge_summary",
            name="apply_masks_to_dataset",
        ),
    ])