# docker/gunicorn.conf.py
bind = "0.0.0.0:8000"
workers = 1
threads = 2
timeout = 120
# security/observability
errorlog = "-"           # send to stderr
accesslog = "-"          # send to stdout
server_header = False    # drop "Server: gunicorn"
