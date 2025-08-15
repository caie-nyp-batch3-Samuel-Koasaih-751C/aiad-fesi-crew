"""Project pipelines."""

from kedro.pipeline import Pipeline
from aiad_fesi_crew.pipelines import data_ingestion, data_preprocessing

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "__default__": data_ingestion.create_pipeline() + data_preprocessing.create_pipeline(),
        "data_ingestion": data_ingestion.create_pipeline(),
        "data_preprocessing": data_preprocessing.create_pipeline(),
    }
