# docker/gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 1
threads = 2
timeout = 120

# If your ui/__init__.py exposes create_app()
wsgi_app = "aiad_fesi_crew.ui:create_app()"

# If you instead expose a global `app` object:
# wsgi_app = "aiad_fesi_crew.ui:app"
