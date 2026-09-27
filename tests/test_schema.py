"""Unit tests for tool decorators, schema extraction, and execution registry."""

import pytest
from yantra.data.schema import Field, ToolDefinition, ToolRegistry, tool


def test_tool_decorator_and_schema():
    """Verify tool decorator schema extraction with type annotations."""
    @tool
    def search_flights(origin: str, destination: str, passengers: int = 1) -> str:
        """Search available flights between destinations."""
        return f"Flight from {origin} to {destination} for {passengers} pax"

    assert isinstance(search_flights, ToolDefinition)
    assert search_flights.name == "search_flights"
    assert search_flights.description == "Search available flights between destinations."

    params = search_flights.parameters
    assert params["type"] == "object"
    assert "origin" in params["properties"]
    assert "destination" in params["properties"]
    assert "passengers" in params["properties"]
    assert params["properties"]["origin"]["type"] == "string"
    assert params["properties"]["passengers"]["type"] == "integer"
    assert params["properties"]["passengers"]["default"] == 1
    assert "origin" in params["required"]
    assert "destination" in params["required"]
    assert "passengers" not in params["required"]

    # Test execution
    res = search_flights.validate_and_execute({"origin": "SFO", "destination": "JFK"})
    assert res == "Flight from SFO to JFK for 1 pax"


def test_tool_registry():
    """Verify tool registry management and wire format generation."""
    @tool
    def turn_on_light(room: str) -> bool:
        """Turn on light in a room."""
        return True

    @tool
    def lock_door(door: str, locked: bool = True) -> bool:
        """Lock or unlock door."""
        return locked

    registry = ToolRegistry([turn_on_light, lock_door])
    assert registry.list_names() == ["turn_on_light", "lock_door"]

    wire_str = registry.to_wire_format()
    assert wire_str.startswith("<tools>[")
    assert wire_str.endswith("]</tools>")
    assert "turn_on_light" in wire_str
    assert "lock_door" in wire_str

    # Test dispatch
    res = registry.execute_call("lock_door", {"door": "front"})
    assert res is True
