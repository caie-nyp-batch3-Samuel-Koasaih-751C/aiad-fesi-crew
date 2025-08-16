from __future__ import annotations
from pathlib import Path
from flask import Flask
from .routes import bp as ui_bp

def create_app() -> Flask:
    here = Path(__file__).parent
    app = Flask(
        __name__,
        static_folder=str(here / "static"),
        template_folder=str(here / "templates"),
    )

    # upload/body size limit (10MB) – adjust if needed
    app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

    # register blueprint
    app.register_blueprint(ui_bp)

    return app
