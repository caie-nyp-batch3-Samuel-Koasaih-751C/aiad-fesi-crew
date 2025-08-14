"""Project pipelines."""

from kedro.pipeline import Pipeline
from aiad_fesi_crew.pipelines import data_ingestion


def register_pipelines() -> dict[str, Pipeline]:
    """
    Register the project's pipelines.
    Returns:
        A mapping from pipeline names to ``Pipeline`` objects.
    """
    return {
        "__default__": data_ingestion.create_pipeline(),  # run ingestion by default (optional)
        "data_ingestion": data_ingestion.create_pipeline(),
    }
