"""Lightweight ext_proc for IaaS.

A minimal gRPC External Processor that acts purely as a routing decision maker.
When X-ITS-Budget is present, it sets an X-ITS-Route header and calls
clear_route_cache so Envoy re-routes the request to the IaaS service.

No body buffering, no algorithm execution — completes in microseconds.
"""

# ruff: noqa: I001
import logging
import uuid

import grpc
import its_hub.integration.proto  # noqa: F401
from envoy.config.core.v3 import base_pb2
from envoy.service.ext_proc.v3 import external_processor_pb2 as ext_proc_pb2
from envoy.service.ext_proc.v3 import external_processor_pb2_grpc as ext_proc_grpc

_ITS_ROUTE_HEADER = "X-ITS-Route"
_ITS_ROUTE_VALUE = "its-service"

logger = logging.getLogger(__name__)


class ExternalProcessorService(ext_proc_grpc.ExternalProcessorServicer):
    """Lightweight ext_proc that routes ITS requests to the IaaS service.

    Checks for X-ITS-Budget in request headers. If present, injects an
    X-ITS-Route header and sets clear_route_cache so Envoy re-evaluates
    the route table and forwards the request to the IaaS upstream cluster.

    ITS headers are preserved on the IaaS path so the service can read
    per-request config. Stray ITS headers on the pass-through path are
    stripped so they never leak to the upstream LLM.
    """

    async def Process(  # noqa: N802 — gRPC servicer interface requires this name
        self,
        request_iterator,
        context: grpc.ServicerContext,
    ):
        peer = context.peer()
        request_id = f"{peer}-{uuid.uuid4().hex[:8]}"

        try:
            async for request in request_iterator:
                if request.HasField("request_headers"):
                    has_budget = False
                    its_keys = []

                    for header in request.request_headers.headers.headers:
                        key = header.key.lower()
                        if key.startswith("x-its-"):
                            its_keys.append(key)
                            if key == "x-its-budget":
                                has_budget = True

                    if has_budget:
                        logger.info(
                            "[%s] ITS budget detected, routing to IaaS",
                            request_id,
                        )
                        yield _ROUTE_TO_IAAS
                    else:
                        yield self._pass_through(its_keys)

                elif request.HasField("response_headers"):
                    yield ext_proc_pb2.ProcessingResponse(
                        response_headers=ext_proc_pb2.HeadersResponse(
                            response=ext_proc_pb2.CommonResponse(
                                status=ext_proc_pb2.CommonResponse.CONTINUE
                            )
                        )
                    )

                elif request.HasField("response_body"):
                    yield ext_proc_pb2.ProcessingResponse(
                        response_body=ext_proc_pb2.BodyResponse(
                            response=ext_proc_pb2.CommonResponse(
                                status=ext_proc_pb2.CommonResponse.CONTINUE
                            )
                        )
                    )

        except Exception as e:
            logger.error("[%s] Stream error: %s", request_id, e, exc_info=True)
            await context.abort(grpc.StatusCode.INTERNAL, str(e))

    @staticmethod
    def _pass_through(its_keys: list[str]) -> ext_proc_pb2.ProcessingResponse:
        """Pass through without routing. Strips any stray X-ITS-* headers."""
        if not its_keys:
            return _PASS_THROUGH
        return ext_proc_pb2.ProcessingResponse(
            request_headers=ext_proc_pb2.HeadersResponse(
                response=ext_proc_pb2.CommonResponse(
                    status=ext_proc_pb2.CommonResponse.CONTINUE,
                    header_mutation=ext_proc_pb2.HeaderMutation(
                        remove_headers=its_keys,
                    ),
                )
            )
        )


_ROUTE_TO_IAAS = ext_proc_pb2.ProcessingResponse(
    request_headers=ext_proc_pb2.HeadersResponse(
        response=ext_proc_pb2.CommonResponse(
            status=ext_proc_pb2.CommonResponse.CONTINUE,
            header_mutation=ext_proc_pb2.HeaderMutation(
                set_headers=[
                    base_pb2.HeaderValueOption(
                        header=base_pb2.HeaderValue(
                            key=_ITS_ROUTE_HEADER,
                            raw_value=_ITS_ROUTE_VALUE.encode(),
                        )
                    ),
                ],
            ),
            clear_route_cache=True,
        )
    )
)

_PASS_THROUGH = ext_proc_pb2.ProcessingResponse(
    request_headers=ext_proc_pb2.HeadersResponse(
        response=ext_proc_pb2.CommonResponse(
            status=ext_proc_pb2.CommonResponse.CONTINUE
        )
    )
)
