from flask import Flask

def create_app(config_filename=None):
    app = Flask(__name__)

    # Load configuration from a file
    if config_filename:
        app.config.from_pyfile(config_filename)

    # Register your blueprints (API routes)
    from .routes import api as api_blueprint
    app.register_blueprint(api_blueprint, url_prefix='/api')  # Prefix all routes with /api

    return app

