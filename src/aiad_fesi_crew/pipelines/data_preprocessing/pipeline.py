from kedro.pipeline import Pipeline, node, pipeline
from .nodes import preprocess_images

def create_pipeline(**_):
    return pipeline([
        node(
            func=preprocess_images,
            inputs=["params:raw_images_path", "params:intermediate_masks_path"],
            outputs="intermediate_masks_path",
            name="preprocess_leaf_images"
        ),
    ])
