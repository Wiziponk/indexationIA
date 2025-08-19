from __future__ import annotations

"""Flask application factory."""

from flask import Flask

from indexation.config import get_config
from indexation.views_common import common_bp
from indexation.views_generate import generate_bp
from indexation.views_cluster import cluster_bp


def create_app() -> Flask:
    app = Flask(__name__)
    config_class = get_config()
    app.config.from_object(config_class)

    app.register_blueprint(common_bp)
    app.register_blueprint(generate_bp)
    app.register_blueprint(cluster_bp)
    return app


if __name__ == "__main__":
    app = create_app()
    app.run(debug=True)
