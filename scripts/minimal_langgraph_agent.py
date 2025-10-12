#!/usr/bin/env python3
"""
Minimal LangGraph agent with calculator tool.
Based on solid React agent pattern from volcengine/verl.
"""

import argparse
import os
from typing import TypedDict, List
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import BaseMessage, HumanMessage, SystemMessage
from typing_extensions import Annotated


@tool
def calculator(expression: str) -> str:
    """Evaluate a mathematical expression safely."""
    try:
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return f"Error: Invalid characters"
        result = eval(expression)
        return str(result)
    except Exception as e:
        return f"Error: {str(e)}"


class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]


def call_model(state: AgentState):
    """Call the LLM with tools."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    
    # Show tool calls if present
    if hasattr(response, 'tool_calls') and response.tool_calls:
        for tc in response.tool_calls:
            print(f"Tool call: {tc}")
    
    return {"messages": [response]}


def should_continue(state: AgentState):
    """Decide whether to continue with tools or end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
        return "tools"
    return "end"


def create_agent(endpoint: str, budget=None):
    """Create the minimal LangGraph agent."""
    global llm_with_tools
    
    # Setup LLM
    api_key = "not-needed" if "localhost" in endpoint else os.getenv("OPENAI_API_KEY")
    extra_body = {"budget": budget} if budget else None
    
    llm = ChatOpenAI(
        api_key=api_key,
        base_url=endpoint,
        model="gpt-4.1-mini",
        extra_body=extra_body
    )
    
    llm_with_tools = llm.bind_tools([calculator])
    
    # Create tools node
    tools = [calculator]
    tool_node = ToolNode(tools)
    
    # Create graph like volcengine/verl pattern
    workflow = StateGraph(AgentState)
    workflow.add_node("agent", call_model)
    workflow.add_node("tools", tool_node)
    workflow.set_entry_point("agent")
    workflow.add_conditional_edges("agent", should_continue, {"tools": "tools", "end": END})
    workflow.add_edge("tools", "agent")
    
    return workflow.compile()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", required=True)
    parser.add_argument("--problem", required=True)
    parser.add_argument("--budget", type=int)
    args = parser.parse_args()
    
    # Create agent
    agent = create_agent(args.endpoint, args.budget)
    
    # Run agent
    result = agent.invoke({
        "messages": [
            SystemMessage(content="Use calculator tool for math. Put final answer in \\boxed{}."),
            HumanMessage(content=args.problem)
        ]
    })
    
    # Print final response
    if result["messages"]:
        final_msg = result["messages"][-1]
        if hasattr(final_msg, 'content') and final_msg.content:
            print(final_msg.content)


if __name__ == "__main__":
    main()