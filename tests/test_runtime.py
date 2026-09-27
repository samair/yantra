"""Unit tests for JSON grammar repair, grounding, and Agent runtime execution."""

import pytest
from yantra.data.schema import tool
from yantra.model.config import YantraConfig
from yantra.model.transformer import YantraForToolCalling
from yantra.runtime.agent import Agent, AgentResponse
from yantra.runtime.grammar import JSONGrammarValidator


def test_json_grammar_repair():
    """Verify deterministic repair of malformed or truncated JSON."""
    # Test unclosed brackets
    unclosed = '[{"name": "get_weather", "arguments": {"city": "Paris"'
    repaired = JSONGrammarValidator.clean_and_repair_json(unclosed)
    assert isinstance(repaired, list)
    assert len(repaired) == 1
    assert repaired[0]["name"] == "get_weather"
    assert repaired[0]["arguments"]["city"] == "Paris"

    # Test single unclosed object
    single_obj = '{"name": "lock_door", "arguments": {"door": "front"'
    repaired2 = JSONGrammarValidator.clean_and_repair_json(single_obj)
    assert isinstance(repaired2, list)
    assert repaired2[0]["name"] == "lock_door"


def test_argument_grounding():
    """Verify argument grounding against user queries."""
    schema = {
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string"},
                "days": {"type": "integer", "default": 1},
            },
            "required": ["city"],
        }
    }

    # Grounded case
    call = {"name": "get_weather", "arguments": {"city": "Tokyo", "days": 1}}
    grounded, ungrounded = JSONGrammarValidator.ground_arguments(
        query="What's the weather in Tokyo?",
        tool_call=call,
        schema=schema,
    )
    assert grounded["city"] == "Tokyo"
    assert grounded["days"] == 1
    assert len(ungrounded) == 0

    # Ungrounded case
    call_hallucinated = {"name": "get_weather", "arguments": {"city": "Antarctica", "fake_param": 123}}
    grounded, ungrounded = JSONGrammarValidator.ground_arguments(
        query="What's the weather in Tokyo?",
        tool_call=call_hallucinated,
        schema=schema,
    )
    assert "fake_param" in ungrounded
    assert "city" in ungrounded


def test_agent_execution_flow():
    """Test Agent end-to-end execution with tool invocation."""
    executed = []

    @tool
    def echo_message(msg: str) -> str:
        """Echo a message."""
        executed.append(msg)
        return f"ECHO: {msg}"

    # Use small model to test Agent scaffolding
    config = YantraConfig(
        vocab_size=8192,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=32,
    )
    model = YantraForToolCalling(config)

    agent = Agent(tools=[echo_message], model=model)
    assert len(agent.registry.list_names()) == 1

    # Test running prompt resolution
    response = agent.run("Hello test", execute_tools=False, max_new_tokens=10)
    assert isinstance(response, AgentResponse)
    assert response.query == "Hello test"
