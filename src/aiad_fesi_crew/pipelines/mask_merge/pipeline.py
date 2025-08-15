"""
This is a boilerplate pipeline 'mask_merge'
generated using Kedro 0.19.14
"""

from kedro.pipeline import Pipeline, node, pipeline
from .nodes import combine_images_and_masks

def create_pipeline(**_):
    return pipeline([
        node(
            combine_images_and_masks,
            inputs=dict(
                raw_images="raw_images",
                leaf_masks="leaf_masks",
                mode="params:mask_merge.mode",
                background_color="params:mask_merge.background_color",
                bbox_margin="params:mask_merge.bbox_margin",
            ),
            outputs=["combined_images", "bbox_index"],
            name="combine_masks_with_images",
        ),
    ])
