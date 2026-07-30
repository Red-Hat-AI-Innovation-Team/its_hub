"""gRPC server for the IaaS ext_proc."""

import argparse
import asyncio
import logging

try:
    import grpc
    from grpc_health.v1 import health, health_pb2_grpc
except ImportError:
    grpc = None  # type: ignore[assignment]
    health = None  # type: ignore[assignment]
    health_pb2_grpc = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)


def _configure_logging(level_name: str) -> None:
    numeric_level = getattr(logging, level_name.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level_name}")

    logging.getLogger().setLevel(numeric_level)
    for logger_name in (
        "its_hub.integration.iaas.ext_processor",
        "its_hub.integration.iaas.grpc_server",
    ):
        logging.getLogger(logger_name).setLevel(numeric_level)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ITS IaaS ext_proc")
    parser.add_argument(
        "--port",
        type=int,
        default=50051,
        help="Port to bind the gRPC server (default: 50051)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (e.g., DEBUG, INFO, WARNING)",
    )
    return parser.parse_args()


async def serve(port: int = 50051):
    """Start the gRPC ext_proc server."""
    from its_hub.integration.iaas.ext_processor import (
        ExternalProcessorService,
        ext_proc_grpc,
    )

    server = grpc.aio.server()
    ext_proc_grpc.add_ExternalProcessorServicer_to_server(
        ExternalProcessorService(), server
    )

    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    server.add_insecure_port(f"[::]:{port}")

    logger.info("Starting IaaS ext_proc on port %d", port)
    await server.start()
    logger.info("IaaS ext_proc ready — routing X-ITS-Budget requests to IaaS")

    try:
        await server.wait_for_termination()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("ext_proc shutting down")
        await server.stop(grace=5)


def main():
    """Entry point for the IaaS ext_proc gRPC service."""
    args = _parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    _configure_logging(args.log_level)

    if grpc is None:
        print(
            "Error: ext_proc dependencies not installed.\n"
            "Install with: pip install 'its_hub[ext_proc]'"
        )
        raise SystemExit(1)

    asyncio.run(serve(port=args.port))


if __name__ == "__main__":
    main()
