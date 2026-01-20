import json
from typing import Annotated, List
from pydantic import Field
from agentlite.commons.fastmcp import mcp

@mcp.tool(
    name="think",
    description="Designed to synthesize information gathered from preceding operations and to articulate the necessary subsequent actions. This tool must be invoked after every tool call to guarantee the logical continuity and coherence of the inferential chain.",
)
def think(
    response: Annotated[str, Field(description="The detailed content of the internal thought process and current state evaluation.")],
):
    """
    Simulates the reasoning step in the ReAct paradigm.
    Does not interact with external systems; it returns the thought content itself 
    to be added to the conversation history as an Observation.
    """
    return "Thinking Finish"

@mcp.tool(
    name="finish",
    description="The final step in the reasoning process. Used only when all necessary data has been retrieved and the clinical prediction is ready. The output MUST be a list of strings, where each string represents a plausible prediction or synthesized finding.",
)
def finish(
    response: Annotated[List[str], Field(description="A list of final, plausible clinical predictions or synthesized findings. Each element in the list must be a separate string.")],
):
    """
    Indicates the end of the reasoning process and provides the final prediction result.
    """
    return "Finish"

# @mcp.tool(
#     name="plan",
#     description="Used to outline the high-level strategic plan or sequence of actions needed to complete the complex clinical prediction task.",
# )
# def plan(
#     content: Annotated[str, Field(description="The detailed strategic plan, breaking down the task into sequential steps and necessary data sources.")],
# ):
#     """
#     Simulates the planning step in the ReAct paradigm.
#     Does not interact with external systems; it returns the plan content itself 
#     to be added to the conversation history as an Observation.
#     """
#     return {"status": "plan_recorded", "content": content}