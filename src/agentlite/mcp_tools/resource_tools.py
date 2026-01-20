import json
from typing import Annotated
from pydantic import Field
from fastmcp import Context

from agentlite.commons.fastmcp import mcp
from agentlite.mcp_tools.tool_utils import get_resource

@mcp.tool(
    name="read_resource_data",
    description="Read data from a resource URI in the cache system."
)
async def read_resource_data(
    ctx: Context,
    uri: Annotated[str, Field(description="The URI of the resource to read (e.g., 'cache://ehr/ehr_data/10000032/table_list.json')")]
) -> str:
    """
    Read resource and return as JSON string.
    
    Args:
        ctx: Context object
        uri: Resource URI to read
        
    Returns:
        str: JSON string of the resource data, or error message
    """
    try:
        data = await get_resource(ctx, uri)
        if data is None:
            return f"Error: Resource not found at URI: {uri}"
        return json.dumps(data)
    except Exception as e:
        return f"Error reading resource: {str(e)}"


