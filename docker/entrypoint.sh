#!/usr/bin/env bash
set -euo pipefail

# Make sure src is importable inside the image
export PYTHONPATH="${PYTHONPATH:-}:/app/src"

# default to UI if PIPELINE is unset/empty
PIPELINE="${PIPELINE:-ui}"

case "${PIPELINE}" in
  ui|UI)
    # Factory lives in aiad_fesi_crew/ui/__init__.py
    exec gunicorn -c /app/gunicorn.conf.py 'aiad_fesi_crew.ui:create_app'
    ;;
  *)
    echo "Running Kedro pipeline: ${PIPELINE}"
    exec kedro run --pipeline "${PIPELINE}"
    ;;
esac
