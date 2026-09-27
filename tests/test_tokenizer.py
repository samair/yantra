"""Unit tests for YantraTokenizer wire protocol and special tokens."""

import pytest
from yantra.data.schema import tool
from yantra.tokenizer.tokenization import YantraTokenizer, SPECIAL_TOKENS


def test_tokenizer_vocab_and_specials():
    """Verify exact 8,192 vocabulary size and special tokens."""
    tokenizer = YantraTokenizer.load()
    assert tokenizer.vocab_size == 8192
    assert tokenizer.pad_token_id == 0
    assert tokenizer.eos_token_id == 1
    assert tokenizer.bos_token_id == 2


def test_tokenizer_format_and_parse():
    """Verify prompt formatting, target serialization, and response parsing."""
    tokenizer = YantraTokenizer.load()

    @tool
    def set_temp(val: int):
        """Set temperature."""
        pass

    prompt = tokenizer.format_prompt([set_temp], "Make it 70 degrees")
    assert "<tools>" in prompt
    assert "set_temp" in prompt
    assert "<|im_start|>user" in prompt
    assert "<|im_start|>assistant" in prompt

    target = tokenizer.format_target(
        thought="Set thermostat to 70",
        tool_calls=[{"name": "set_temp", "arguments": {"val": 70}}],
    )
    assert "<think>Set thermostat to 70</think>" in target
    assert "<tool_call>[{\"name\":\"set_temp\",\"arguments\":{\"val\":70}}]</tool_call>" in target
    assert target.endswith("<|im_end|></s>")

    thought, calls = tokenizer.parse_assistant_response(target)
    assert thought == "Set thermostat to 70"
    assert len(calls) == 1
    assert calls[0]["name"] == "set_temp"
    assert calls[0]["arguments"] == {"val": 70}


def test_encoding_decoding_roundtrip():
    """Verify text round-trip encoding and decoding."""
    tokenizer = YantraTokenizer.load()
    text = "Hello world! This is a tool call to set_thermostat with temp: 72."
    ids = tokenizer.encode(text)
    decoded = tokenizer.decode(ids, skip_special_tokens=True)
    assert "set_thermostat" in decoded
