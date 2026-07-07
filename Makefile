# ITS Hub Makefile
# Handles proto compilation for Envoy external processor

.PHONY: help setup setup-envoy upgrade-protos proto-compile proto-clean submodule-init \
        iaas-start iaas-health envoy-stack envoy-stack-stop envoy-start envoy-grpc envoy-test envoy-health test

# Default target
help:
	@echo "ITS Hub Build Targets:"
	@echo ""
	@echo "Setup Commands:"
	@echo "  make setup          - General development setup (Python deps + requirements.txt)"
	@echo "  make setup-envoy    - Envoy gateway setup (submodules + proto compilation)"
	@echo ""
	@echo "Proto Commands:"
	@echo "  make submodule-init - Initialize git submodules for proto definitions"
	@echo "  make proto-compile  - Compile Envoy proto files to Python"
	@echo "  make proto-clean    - Remove generated proto files"
	@echo ""
	@echo "Service Commands:"
	@echo "  make iaas-start       - Start IaaS service on localhost:8108"
	@echo "  make iaas-health      - Check IaaS service health"
	@echo "  make envoy-stack      - Start Envoy proxy + gRPC service together"
	@echo "  make envoy-stack-stop - Stop Envoy stack"
	@echo "  make envoy-start      - Start Envoy proxy with ext_proc configuration"
	@echo "  make envoy-grpc       - Start Envoy external processor gRPC service"
	@echo "  make envoy-test       - Test Envoy external processor with sample requests"
	@echo "  make envoy-health     - Check Envoy cluster health and statistics"
	@echo ""
	@echo "Testing Commands:"
	@echo "  make test           - Run all pytest tests"
	@echo ""
	@echo "Maintenance Commands:"
	@echo "  make upgrade-protos - Restore proto submodules to pinned commits from .gitmodules"

# Directories
PROTO_OUT_DIR := its_hub/integration/ext_proc/proto
THIRD_PARTY := third_party

# Proto source directories
ENVOY_API := $(THIRD_PARTY)/envoy-data-plane-api
XDS_API := $(THIRD_PARTY)/xds
VALIDATE := $(THIRD_PARTY)/protoc-gen-validate

# Proto source files
PROTO_SOURCES := $(wildcard $(ENVOY_API)/envoy/config/core/v3/*.proto) \
                 $(wildcard $(ENVOY_API)/envoy/type/v3/*.proto) \
                 $(wildcard $(ENVOY_API)/envoy/type/matcher/v3/*.proto) \
                 $(wildcard $(ENVOY_API)/envoy/annotations/*.proto) \
                 $(wildcard $(ENVOY_API)/envoy/extensions/filters/http/ext_proc/v3/*.proto) \
                 $(wildcard $(ENVOY_API)/envoy/service/ext_proc/v3/*.proto) \
                 $(wildcard $(XDS_API)/xds/annotations/v3/*.proto) \
                 $(wildcard $(XDS_API)/xds/core/v3/*.proto) \
                 $(wildcard $(XDS_API)/udpa/annotations/*.proto) \
                 $(VALIDATE)/validate/validate.proto

# Marker files to track state
SUBMODULE_MARKER := $(THIRD_PARTY)/.submodules-initialized
PROTO_MARKER := $(PROTO_OUT_DIR)/.proto-compiled

# Initialize git submodules
$(SUBMODULE_MARKER):
	@echo "Initializing git submodules for proto definitions..."
	git submodule update --init --recursive \
		$(ENVOY_API) \
		$(XDS_API) \
		$(VALIDATE)
	@touch $(SUBMODULE_MARKER)
	@echo "✓ Submodules initialized"

# PHONY target for submodule initialization
submodule-init: $(SUBMODULE_MARKER)

# Compile Envoy proto files to Python (with dependency tracking)
$(PROTO_MARKER): $(SUBMODULE_MARKER) $(PROTO_SOURCES)
	@echo "Compiling Envoy proto files..."
	uv run python -m grpc_tools.protoc \
		--proto_path=third_party/envoy-data-plane-api \
		--proto_path=third_party/xds \
		--proto_path=third_party/protoc-gen-validate \
		--python_out=$(PROTO_OUT_DIR) \
		--grpc_python_out=$(PROTO_OUT_DIR) \
		third_party/envoy-data-plane-api/envoy/config/core/v3/*.proto \
		third_party/envoy-data-plane-api/envoy/type/v3/*.proto \
		third_party/envoy-data-plane-api/envoy/type/matcher/v3/*.proto \
		third_party/envoy-data-plane-api/envoy/annotations/*.proto \
		third_party/envoy-data-plane-api/envoy/extensions/filters/http/ext_proc/v3/*.proto \
		third_party/envoy-data-plane-api/envoy/service/ext_proc/v3/*.proto \
		third_party/xds/xds/annotations/v3/*.proto \
		third_party/xds/xds/core/v3/*.proto \
		third_party/xds/udpa/annotations/*.proto \
		third_party/protoc-gen-validate/validate/validate.proto
	find $(PROTO_OUT_DIR) -type d -exec touch {}/__init__.py \;
	@touch $(PROTO_MARKER)
	@echo "✓ Proto files compiled successfully!"

# PHONY target for proto compilation
proto-compile: $(PROTO_MARKER)

# Remove generated proto files
proto-clean:
	@echo "Removing generated proto files..."
	rm -rf $(PROTO_OUT_DIR)/envoy $(PROTO_OUT_DIR)/xds $(PROTO_OUT_DIR)/udpa $(PROTO_OUT_DIR)/validate
	rm -f $(PROTO_MARKER)
	@echo "✓ Proto files removed"

# General development setup
setup:
	@echo "Setting up its_hub development environment..."
	@echo "Installing Python dependencies..."
	uv sync --extra dev
	@echo "Generating requirements.txt for other build tools..."
	uv export --no-hashes > requirements.txt
	@echo ""
	@echo "✓ General setup complete!"
	@echo ""
	@echo "For Envoy gateway development, also run: make setup-envoy"

# Envoy gateway development setup
setup-envoy: submodule-init proto-compile
	@echo ""
	@echo "✓ Envoy gateway setup complete!"
	@echo "You can now run: make envoy-stack"

# Restore proto submodules to pinned commits from .gitmodules
upgrade-protos:
	@echo "Restoring proto submodules to pinned commits from .gitmodules..."
	@echo ""
	@echo "Checking out pinned commits:"
	@ENVOY_COMMIT=$$(grep -A 2 'envoy-data-plane-api' .gitmodules | grep 'pinned-commit' | cut -d'=' -f2 | tr -d ' '); \
	cd $(ENVOY_API) && git fetch && git checkout $$ENVOY_COMMIT && \
	echo "  ✓ envoy-data-plane-api: $$ENVOY_COMMIT"
	@XDS_COMMIT=$$(grep -A 2 'third_party/xds' .gitmodules | grep 'pinned-commit' | cut -d'=' -f2 | tr -d ' '); \
	cd $(XDS_API) && git fetch && git checkout $$XDS_COMMIT && \
	echo "  ✓ xds: $$XDS_COMMIT"
	@VALIDATE_COMMIT=$$(grep -A 3 'protoc-gen-validate' .gitmodules | grep 'pinned-commit' | cut -d'=' -f2 | tr -d ' '); \
	cd $(VALIDATE) && git fetch && git checkout $$VALIDATE_COMMIT && \
	echo "  ✓ protoc-gen-validate: $$VALIDATE_COMMIT"
	@echo ""
	@echo "✓ All submodules restored to pinned commits"
	@echo ""
	@echo "To update the pinned commits, edit .gitmodules and update the pinned-commit values"

# =============================================================================
# Service Management
# =============================================================================

# Start both Envoy proxy and ext_proc gRPC service in parallel
envoy-stack:
	@echo "Starting Envoy stack (proxy + gRPC service)..."
	@echo "Logs will be written to:"
	@echo "  - envoy.log (Envoy proxy)"
	@echo "  - envoy-grpc.log (gRPC service)"
	@echo ""
	@echo "Press Ctrl+C to stop both services"
	@trap 'kill 0' INT; \
	(uv run envoy-grpc 2>&1 | tee envoy-grpc.log) & \
	(envoy -c config/envoy/ext_proc.yaml 2>&1 | tee envoy.log) & \
	wait

# Stop Envoy stack
envoy-stack-stop:
	@echo "Stopping Envoy stack..."
	@pkill -f "envoy -c config/envoy/ext_proc.yaml" || echo "Envoy proxy not running"
	@pkill -f "envoy-grpc" || echo "gRPC service not running"
	@echo "✓ Envoy stack stopped"

# Start IaaS service on localhost:8108
iaas-start:
	uv run its-iaas --host 0.0.0.0 --port 8108

# Check IaaS service health
iaas-health:
	curl -v -s http://localhost:8108/v1/models | jq .

# Start Envoy proxy with ext_proc configuration
envoy-start:
	envoy -c config/envoy/ext_proc.yaml

# Start Envoy External Processor gRPC service
envoy-grpc:
	uv run envoy-grpc

# Test Envoy External Processor with sample requests
envoy-test:
	uv run python scripts/test_envoy_grpc.py

# Check Envoy cluster health and statistics
envoy-health:
	@echo "=== Cluster Status ==="
	@curl -s http://localhost:9901/clusters | grep -A 10 "ext_proc_cluster" || echo "Envoy not running or admin port not accessible"
	@echo ""
	@echo "=== ext_proc Statistics ==="
	@curl -s http://localhost:9901/stats | grep ext_proc || echo "No ext_proc stats found"

# =============================================================================
# Testing
# =============================================================================

# Run all tests
test:
	uv run pytest tests/
