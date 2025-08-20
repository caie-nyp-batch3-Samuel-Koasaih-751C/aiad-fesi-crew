#!/bin/sh
set -eu
export PYTHONPATH="${PYTHONPATH:-}:/app/src"
PIPELINE="${PIPELINE:-ui}"

case "$PIPELINE" in
  ui|UI)
    # Explicitly tell Gunicorn where the Flask app factory is
    exec gunicorn -c /app/gunicorn.conf.py "aiad_fesi_crew.ui:create_app()"
    ;;
  *)
    echo "Running Kedro pipeline: $PIPELINE"
    exec kedro run --pipeline "$PIPELINE"
    ;;
esac
