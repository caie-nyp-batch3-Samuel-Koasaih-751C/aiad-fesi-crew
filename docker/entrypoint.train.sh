#!/usr/bin/env bash
set -e

# Optional: log versions for reproducibility
python --version
pip freeze | grep -E 'kedro|tensorflow|torch|opencv|pandas|numpy' || true

# If you want to pass a pipeline, leave a default
PIPELINE="${PIPELINE:-train_validate_test}"

# Use Kedro to run the training pipeline
exec kedro run --pipeline "$PIPELINE"
