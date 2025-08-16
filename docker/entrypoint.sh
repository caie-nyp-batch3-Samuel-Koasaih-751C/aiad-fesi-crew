#!/usr/bin/env bash
set -e

# Which Kedro pipeline to run (for batch), or special "ui" to run Flask
PIPELINE="${PIPELINE:-__default__}"
ENV_NAME="${KEDRO_ENV:-base}"

if [ "$PIPELINE" = "ui" ]; then
  echo ">>> Starting Flask UI via Gunicorn..."
  exec gunicorn "aiad_fesi_crew.ui:create_app()" -c docker/gunicorn.conf.py
else
  echo ">>> Running Kedro pipeline: $PIPELINE (env: $ENV_NAME)"
  exec kedro run --pipeline "$PIPELINE" -e "$ENV_NAME"
fi
