"""An agent graph with a post-response vibe checker check loop.

After the agent responds, a secondary node evaluates vibe.
If vibe is acceptable, end; otherwise, continue the loop or terminate after a safe limit.
"""
from __future__ import annotations
from pydantic import BaseModel, Field
from typing import TypedDict, Annotated, NotRequired

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages

from app.models import get_chat_model
from app.tools import get_tool_belt

MAX_VIBE_ATTEMPTS = 3

class VibeCheckerResult(BaseModel):
    vibe_acceptable: bool = Field(description="Whether the response vibe is acceptable.")
    vibe_style: str = Field(description="Short tone label, e.g. warm, cold, formal.")
    vibe_feedback: str = Field(default="", description="Feedback message to the assistant if the vibe is not acceptable.")
    
class VibeState(TypedDict):
    messages: Annotated[list, add_messages]
    vibe_attempts: NotRequired[int]
    vibe_passed: NotRequired[bool]

def _build_model_with_tools():
    """Return a chat model instance bound to the current tool belt."""
    model = get_chat_model()
    return model.bind_tools(get_tool_belt())


def call_model(state: VibeState) -> dict:
    """Invoke the model with the accumulated messages and append its response."""
    model = _build_model_with_tools()
    response = model.invoke(state["messages"])
    return {"messages": [response]}


def route_to_action_or_vibe_checker(state: VibeState):
    """Decide whether to execute tools or run the vibe checker evaluator."""
    last_message = state["messages"][-1]
    if getattr(last_message, "tool_calls", None):
        return "action"
    return "vibechecker"


_vibechecker_prompt = ChatPromptTemplate.from_template(
    "You are a response quality evaluator.\n\n"
    "Given the initial user query and the assistant's final response, decide if the vibe is acceptable.\n"
    "Criteria: respectful tone, clarity, and helpfulness.\n\n"
    "Initial Query:\n{initial_query}\n\n"
    "Final Response:\n{final_response}\n\n"
    "If the vibe is not acceptable, provide one short actionable rewrite instruction as feedback.\n\n"
    "Also return a short tone label (1-2 words).\n\n"
    )


def vibechecker_node(state: VibeState) -> dict:
    """Evaluate vibe of the latest response relative to the initial query."""
    attempts = int(state.get("vibe_attempts", 0)) + 1

    if attempts > MAX_VIBE_ATTEMPTS:
        return {"vibe_attempts": attempts, "vibe_passed": False}

    initial_query = next((m for m in state["messages"] if isinstance(m, HumanMessage)), None)
    last_ai = next((m for m in reversed(state["messages"]) if isinstance(m, AIMessage)),None,)
    if last_ai is None:
        return {
            "vibe_attempts": attempts, 
            "vibe_passed": False, 
            "messages": [SystemMessage(content="No assistant response found. Please provide an answer.")]}
    final_response = last_ai

    structured_model = get_chat_model(model_name="gpt-4.1-mini").with_structured_output(VibeCheckerResult)
    result = (_vibechecker_prompt | structured_model).invoke(
        {
            "initial_query": getattr(initial_query, "content", ""),
            "final_response": getattr(final_response, "content", ""),
        }
    )

    if result.vibe_acceptable:
        return {"vibe_attempts": attempts, "vibe_passed": True}

    return {
        "vibe_attempts": attempts,
        "vibe_passed": False,
        "messages": [
            SystemMessage(
                content=(
                    "Please rewrite your previous answer.\n"
                    f"Detected tone: {result.vibe_style}.\n"
                    f"Rewrite guidance: {result.vibe_feedback}\n"
                    "Keep it concise, helpful, and natural."
                )
            )
        ],
    }

def vibechecker_decision(state: VibeState):
    """End when vibe passes or when attempts exceed the max limit; otherwise continue."""
    if state.get("vibe_passed", False):
        return "end"
    if int(state.get("vibe_attempts", 0)) > MAX_VIBE_ATTEMPTS:
        return "end"   
    return "continue"

def build_graph():
    """Build an agent graph with an auxiliary vibechecker evaluation subgraph."""
    graph = StateGraph(VibeState)
    tool_node = ToolNode(get_tool_belt())
    graph.add_node("agent", call_model)
    graph.add_node("action", tool_node)
    graph.add_node("vibechecker", vibechecker_node)
    graph.add_edge(START, "agent")
    graph.add_conditional_edges(
        "agent",
        route_to_action_or_vibe_checker,
        {"action": "action", "vibechecker": "vibechecker"},
    )
    graph.add_conditional_edges(
        "vibechecker",
        vibechecker_decision,
        {"continue": "agent", "end": END},
    )
    graph.add_edge("action", "agent")
    return graph


graph = build_graph().compile()
