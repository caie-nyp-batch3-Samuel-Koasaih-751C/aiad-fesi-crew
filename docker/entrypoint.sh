#!/bin/sh
set -eu
export PYTHONPATH="${PYTHONPATH:-}:/app/src"
PIPELINE="${PIPELINE:-ui}"

case "$PIPELINE" in
  ui|UI)
    exec gunicorn -c /app/gunicorn.conf.py
    ;;
  *)
    echo "Running Kedro pipeline: $PIPELINE"
    exec kedro run --pipeline "$PIPELINE"
    ;;
esac
