bind = "0.0.0.0:8000"
workers = 1
threads = 2
timeout = 120
wsgi_app = "aiad_fesi_crew.ui:create_app()"  # <- call the factory
