"""
This is a boilerplate pipeline 'data_ingestion'
generated using Kedro 0.19.14
"""
from kedro.pipeline import Pipeline, node, pipeline
from .nodes import download_from_gdrive

def create_pipeline(**_):
    return pipeline([
        node(
            func=download_from_gdrive,
            inputs=["params:gdrive_file_id", "params:raw_data_dir"],
            outputs="raw_data_dir",
            name="download_dataset_gdrive"
        ),
    ])
