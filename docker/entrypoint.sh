#!/bin/sh
set -e

# Ensure src layout is visible even if editable install didn’t pick up
export PYTHONPATH="${PYTHONPATH}:/project/src"

case "${PIPELINE}" in
  ui|UI)
    # If your factory lives in aiad_fesi_crew/ui/routes.py:
    exec gunicorn -c /docker/gunicorn.conf.py aiad_fesi_crew.ui.routes:create_app
    ;;
  "" )
    echo "PIPELINE env var is empty. Set PIPELINE=ui or a Kedro pipeline name." >&2
    exit 1
    ;;
  * )
    exec kedro run --pipeline "${PIPELINE}"
    ;;
esac
