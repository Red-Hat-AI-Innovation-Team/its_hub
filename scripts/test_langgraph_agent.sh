#!/bin/bash

# Test LangGraph Agent - Standard vs IaaS endpoints
# Shows drop-in replacement capability with tool calls

set -e

echo "🤖 Testing LangGraph Math Agent"
echo "==============================="

# Check if IaaS server is running
if ! curl -s http://localhost:8108/v1/models >/dev/null 2>&1; then
    echo "❌ IaaS server not running. Start it first:"
    echo "   ./scripts/start_iaas_server.sh"
    exit 1
fi

# Check for .env file
if [ ! -f ".env" ]; then
    echo "❌ .env file not found!"
    exit 1
fi

# Check if minimal LangGraph agent exists
if [ ! -f "scripts/minimal_langgraph_agent.py" ]; then
    echo "❌ Minimal LangGraph agent script not found!"
    exit 1
fi

MATH_PROBLEM="Calculate (45 * 67) + (89 - 23) and put the answer in boxed format"

echo ""
echo "📝 Problem: $MATH_PROBLEM"
echo ""

# Test 1: LangGraph agent with standard OpenAI endpoint
echo "🔄 Testing LangGraph Agent with Standard OpenAI..."
echo "================================================="

export OPENAI_API_KEY=$(grep OPENAI_API_KEY .env | cut -d'=' -f2)

uv run python scripts/minimal_langgraph_agent.py \
    --endpoint https://api.openai.com/v1 \
    --problem "$MATH_PROBLEM"

echo ""
echo ""

# Test 2: Same LangGraph agent with IaaS endpoint (inference-time scaling)
echo "⚡ Testing LangGraph Agent with IaaS Scaling..."
echo "=============================================="

uv run python scripts/minimal_langgraph_agent.py \
    --endpoint http://localhost:8108/v1 \
    --problem "$MATH_PROBLEM" \
    --budget 3

echo ""
echo ""
echo "✅ LangGraph Agent Comparison Complete!"
echo ""
echo "🎯 Key Insights:"
echo "  • Same agent code, different endpoints"
echo "  • Tool calls work identically on both endpoints"  
echo "  • IaaS provides inference-time scaling transparently"
echo "  • Expected: calculator(\"45 * 67\"), calculator(\"89 - 23\"), final \\boxed{3081}"