"""Project pipelines."""

from kedro.pipeline import Pipeline
import aiad_fesi_crew.pipelines.data_ingestion as data_ingestion
import aiad_fesi_crew.pipelines.data_preprocessing as data_preprocessing  # your masking pipeline
import aiad_fesi_crew.pipelines.mask_merge as mask_merge                  # <-- new

def register_pipelines() -> dict[str, Pipeline]:
    return {
        "__default__": data_ingestion.create_pipeline()
                       + data_preprocessing.create_pipeline()
                       + mask_merge.create_pipeline(),
        "data_ingestion": data_ingestion.create_pipeline(),
        "data_preprocessing": data_preprocessing.create_pipeline(),
        "mask_merge": mask_merge.create_pipeline(),
    }
