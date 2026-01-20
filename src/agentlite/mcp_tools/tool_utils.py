# utils.py
import json
import pandas as pd
from typing import Optional
from fastmcp import Context

def normalize_datetime(dt_str: str) -> Optional[pd.Timestamp]:
    """
    Converts a date/time string to a standardized pandas Timestamp object.
    
    This function handles various formats and attempts to fill in missing components
    to ensure the timestamp is accurate to the second.
    
    Args:
        dt_str (str): The date/time string to normalize.
        
    Returns:
        Optional[pd.Timestamp]: A normalized Timestamp object, or None if conversion fails.
    """
    try:
        # 使用 pd.to_datetime 来处理多种格式
        timestamp = pd.to_datetime(dt_str)
        # 如果格式不完整，补全到秒
        if not timestamp.minute:
            timestamp = timestamp.replace(minute=0)
        if not timestamp.second:
            timestamp = timestamp.replace(second=0)
        return timestamp
    except (ValueError, TypeError):
        return None

def find_timestamp_column(df: pd.DataFrame) -> Optional[str]:
    """
    Finds the first column in a DataFrame whose name contains 'time' or 'date'.
    Returns the column name if found, otherwise returns None.
    """
    for col in df.columns:
        if 'time' in col.lower() or 'date' in col.lower():
            return col
    return None


async def get_resource_df(ctx: Context, uri: str) -> pd.DataFrame:
    try:
        blocks = await ctx.read_resource(uri)
        if not blocks:
            return None

        blk = blocks[0]
        text = blk.content
        data = json.loads(text)
        return pd.DataFrame(data)
    except:
        return None

async def get_resource(ctx: Context, uri: str) -> pd.DataFrame:
    try:
        blocks = await ctx.read_resource(uri)
        if not blocks:
            return None

        blk = blocks[0]
        text = blk.content
        data = json.loads(text)
        return data
    except:
        return None
