"""Analytics query schemas."""

from __future__ import annotations

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    """SQL query request for DuckDB analytics."""

    sql: str = Field(..., description="Read-only SQL query", max_length=2000)


class QueryResponse(BaseModel):
    """Query result response."""

    columns: list[str]
    data: list[dict]
    row_count: int
