"""
Generated protobuf code for Envoy External Processor.

This package contains auto-generated Python code from Envoy proto files.
The proto subdirectory is added to sys.path to allow imports like:
    from envoy.service.ext_proc.v3 import external_processor_pb2
"""

import sys
from pathlib import Path

# Add proto directory to sys.path so that generated imports work
_proto_dir = Path(__file__).parent
if str(_proto_dir) not in sys.path:
    sys.path.insert(0, str(_proto_dir))
