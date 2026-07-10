"""gRPC server for the Envoy External Processor."""

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
    """Adjust log levels for root and key ITS modules."""
    numeric_level = getattr(logging, level_name.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level_name}")

    logging.getLogger().setLevel(numeric_level)
    for logger_name in (
        "its_hub.integration.ext_proc.processor",
        "its_hub.integration.ext_proc.server",
        "its_hub.core.gateway",
        "its_hub.core.algorithms.self_consistency",
    ):
        logging.getLogger(logger_name).setLevel(numeric_level)


def _parse_args() -> argparse.Namespace:
    """Parse CLI arguments for the ext_proc server."""
    parser = argparse.ArgumentParser(description="ITS Envoy External Processor")
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
    """Start the gRPC server."""
    from its_hub.integration.ext_proc.processor import (
        ExternalProcessorService,
        ext_proc_grpc,
    )

    server = grpc.aio.server()

    processor = ExternalProcessorService()

    ext_proc_grpc.add_ExternalProcessorServicer_to_server(processor, server)

    health_servicer = health.aio.HealthServicer()
    health_pb2_grpc.add_HealthServicer_to_server(health_servicer, server)

    server.add_insecure_port(f"[::]:{port}")

    logger.info("Starting External Processor on port %d", port)
    await server.start()

    logger.info("External Processor is ready to receive requests from Envoy")

    try:
        await server.wait_for_termination()
    except (KeyboardInterrupt, asyncio.CancelledError):
        logger.info("Received shutdown signal")
        await processor.shutdown()
        await server.stop(grace=5)


def main():
    """Entry point for the external processor service."""
    if grpc is None:
        print(
            "Error: ext_proc dependencies not installed.\n"
            "Install with: pip install 'its_hub[ext_proc]'"
        )
        raise SystemExit(1)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    args = _parse_args()
    _configure_logging(args.log_level)
    try:
        asyncio.run(serve(port=args.port))
    except KeyboardInterrupt:
        logger.info("Service stopped")


if __name__ == "__main__":
    main()
