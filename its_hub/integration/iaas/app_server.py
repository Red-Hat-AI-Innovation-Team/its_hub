"""HTTP server for the IaaS FastAPI application."""

import argparse
import logging
import sys
from importlib import resources

try:
    import uvicorn
except ImportError:
    uvicorn = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _configure_logging(level_name: str) -> None:
    numeric_level = getattr(logging, level_name.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level_name}")

    logging.getLogger().setLevel(numeric_level)
    for logger_name in (
        "its_hub.integration.iaas.app",
        "its_hub.integration.iaas.app_server",
        "its_hub.core.gateway",
        "its_hub.core.algorithms.self_consistency",
    ):
        logging.getLogger(logger_name).setLevel(numeric_level)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ITS Inference-as-a-Service")
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Host to bind the server to (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8109,
        help="Port to bind the server (default: 8109)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g., DEBUG, INFO, WARNING)",
    )
    parser.add_argument(
        "--dev",
        action="store_true",
        help="Run in development mode with auto-reload",
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print the sample Envoy config to stdout and exit",
    )
    return parser.parse_args()


def _print_config() -> None:
    """Print the bundled sample Envoy config to stdout."""
    config = resources.files("its_hub.integration.iaas").joinpath("envoy_config.yaml")
    sys.stdout.write(config.read_text())


def serve(host: str = "127.0.0.1", port: int = 8109, dev: bool = False):
    """Start the uvicorn server."""
    from its_hub.integration.iaas.app import app

    uvicorn_config = {
        "host": host,
        "port": port,
        "log_level": "info" if not dev else "debug",
    }

    if dev:
        logger.info("Running in development mode with auto-reload")
        uvicorn.run("its_hub.integration.iaas.app:app", reload=True, **uvicorn_config)
    else:
        uvicorn.run(app, **uvicorn_config)


def main():
    """Entry point for the IaaS HTTP service."""
    args = _parse_args()

    if args.print_config:
        _print_config()
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _configure_logging(args.log_level)

    if uvicorn is None:
        print(
            "Error: iaas dependencies not installed.\n"
            "Install with: pip install 'its_hub[iaas]'"
        )
        raise SystemExit(1)

    logger.info("Starting IaaS on %s:%d", args.host, args.port)
    serve(host=args.host, port=args.port, dev=args.dev)


if __name__ == "__main__":
    main()
