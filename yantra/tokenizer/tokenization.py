"""Yantra Tokenizer: 8,192-vocab BPE tokenizer with specialized tool tokens and prompt formatter."""

import json
import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from tokenizers import Tokenizer, decoders, models, normalizers, pre_tokenizers, trainers
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer

from yantra.data.schema import ToolDefinition, ToolRegistry

# Fixed special tokens mapping (Tokens 0-15 matching standard wire layout)
SPECIAL_TOKENS = [
    "<pad>",          # 0
    "</s>",           # 1
    "<s>",            # 2
    "<unk>",          # 3
    "<|im_start|>",   # 4
    "<|im_end|>",     # 5
    "<think>",        # 6
    "</think>",       # 7
    "<tools>",        # 8
    "</tools>",       # 9
    "<tool_call>",    # 10
    "</tool_call>",   # 11
    "<tool_result>",  # 12
    "</tool_result>", # 13
    "<context>",      # 14
    "</context>",     # 15
]

DEFAULT_VOCAB_SIZE = 8192
TOKENIZER_DIR = Path(__file__).parent / "pretrained"


class YantraTokenizer:
    """Fast Byte-Level BPE Tokenizer for Yantra with dedicated wire protocol formatters."""

    def __init__(self, tokenizer: Tokenizer):
        self._tokenizer = tokenizer
        self.special_tokens = SPECIAL_TOKENS
        self.vocab_size = self._tokenizer.get_vocab_size()

        # Token ID lookup
        self.pad_token_id = self._tokenizer.token_to_id("<pad>")
        self.eos_token_id = self._tokenizer.token_to_id("</s>")
        self.bos_token_id = self._tokenizer.token_to_id("<s>")
        self.unk_token_id = self._tokenizer.token_to_id("<unk>")
        self.im_start_id = self._tokenizer.token_to_id("<|im_start|>")
        self.im_end_id = self._tokenizer.token_to_id("<|im_end|>")
        self.tool_call_start_id = self._tokenizer.token_to_id("<tool_call>")
        self.tool_call_end_id = self._tokenizer.token_to_id("</tool_call>")

    @classmethod
    def train_new(
        cls,
        corpus: List[str],
        vocab_size: int = DEFAULT_VOCAB_SIZE,
        save_path: Optional[str] = None,
    ) -> "YantraTokenizer":
        """Train an 8,192 BPE tokenizer on a curated domain corpus."""
        tokenizer = Tokenizer(BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()

        trainer = BpeTrainer(
            vocab_size=vocab_size,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            min_frequency=1,
            show_progress=False,
        )

        tokenizer.train_from_iterator(corpus, trainer=trainer)

        # Ensure exact vocabulary size by adding reserved tokens if needed
        current_size = tokenizer.get_vocab_size()
        if current_size < vocab_size:
            diff = vocab_size - current_size
            tokenizer.add_special_tokens([f"<reserved_{i}>" for i in range(diff)])

        if save_path:
            os.makedirs(save_path, exist_ok=True)
            tokenizer.save(os.path.join(save_path, "tokenizer.json"))

        return cls(tokenizer)

    @classmethod
    def load(cls, path: Optional[Union[str, Path]] = None) -> "YantraTokenizer":
        """Load tokenizer from directory or default pretrained location."""
        target_path = Path(path) if path else (TOKENIZER_DIR / "tokenizer.json")
        if not target_path.exists():
            return cls.build_default(save_dir=TOKENIZER_DIR)
        tokenizer = Tokenizer.from_file(str(target_path))
        return cls(tokenizer)

    @classmethod
    def build_default(cls, save_dir: Optional[Path] = None) -> "YantraTokenizer":
        """Build and cache the default domain-optimized tokenizer with 8,192 tokens."""
        seed_data = [
            "{\"name\": \"get_weather\", \"description\": \"Retrieve real-time weather information for a given location\", \"parameters\": {\"type\": \"object\", \"properties\": {\"city\": {\"type\": \"string\"}, \"days\": {\"type\": \"integer\"}}, \"required\": [\"city\"]}}",
            "{\"name\": \"turn_on_light\", \"description\": \"Turn on smart light\", \"parameters\": {\"type\": \"object\", \"properties\": {\"room\": {\"type\": \"string\"}, \"brightness\": {\"type\": \"integer\"}}, \"required\": [\"room\"]}}",
            "{\"name\": \"search_contacts\", \"description\": \"Search address book contacts\", \"parameters\": {\"type\": \"object\", \"properties\": {\"query\": {\"type\": \"string\"}, \"limit\": {\"type\": \"integer\"}}, \"required\": [\"query\"]}}",
            "{\"name\": \"send_message\", \"description\": \"Send text message\", \"parameters\": {\"type\": \"object\", \"properties\": {\"recipient\": {\"type\": \"string\"}, \"message\": {\"type\": \"string\"}}, \"required\": [\"recipient\", \"message\"]}}",
            "{\"name\": \"set_timer\", \"description\": \"Set countdown timer in minutes\", \"parameters\": {\"type\": \"object\", \"properties\": {\"minutes\": {\"type\": \"integer\"}, \"label\": {\"type\": \"string\"}}, \"required\": [\"minutes\"]}}",
            "<s><|im_start|>system\ndate: 2026-09-27 Sun 17:00<|im_end|>\n<|im_start|>user\n<tools>[]</tools>\nWhat is the meaning of life?<|im_end|>\n<|im_start|>assistant\n<think>No relevant tool found</think>\n<tool_call>[]</tool_call><|im_end|></s>",
            "def execute(tool_name: str, arguments: dict) -> None: pass",
            "import os, sys, json, math, time, datetime, typing",
            "turn on bedroom light, turn off living room AC, set thermostat to 72 degrees Fahrenheit",
            "schedule meeting with Sameer at 3pm tomorrow, remind me to buy groceries in 30 minutes",
            "calculate distance between San Francisco and Tokyo, convert 50 EUR to USD currency exchange",
            "play jazz music on kitchen speaker, skip track, volume up by 10 percent",
            "true, false, null, undefined, NaN, None, True, False, int, float, str, bool, list, dict",
            "create, read, update, delete, get, post, put, patch, query, search, find, execute, trigger",
        ]
        expanded = list(seed_data)
        words = [
            "the", "of", "and", "to", "in", "is", "you", "that", "it", "he", "was", "for", "on", "are", "as",
            "with", "his", "they", "at", "be", "this", "from", "I", "have", "or", "by", "one", "had", "not",
            "but", "what", "all", "were", "when", "we", "there", "can", "an", "your", "which", "their", "said",
            "if", "do", "will", "each", "about", "how", "up", "out", "them", "then", "she", "many", "some", "so",
            "these", "would", "other", "into", "has", "more", "her", "two", "like", "him", "see", "time", "could",
            "no", "make", "than", "first", "been", "its", "who", "now", "people", "my", "made", "over", "did", "down",
            "only", "way", "find", "use", "may", "water", "long", "little", "very", "after", "words", "called", "just",
            "where", "most", "know", "get", "through", "back", "much", "go", "good", "new", "write", "our", "me", "man",
            "too", "any", "day", "same", "right", "look", "think", "also", "around", "another", "came", "come", "work",
            "three", "must", "because", "does", "part", "even", "place", "well", "such", "here", "take", "why", "help",
            "put", "different", "away", "again", "off", "went", "old", "number", "great", "tell", "men", "say", "small",
            "every", "found", "still", "between", "name", "should", "home", "big", "give", "air", "line", "set", "own",
            "under", "read", "last", "never", "us", "left", "end", "along", "while", "might", "next", "sound", "below",
            "something", "thought", "both", "few", "those", "always", "show", "large", "often", "together", "asked",
            "house", "world", "going", "want", "school", "important", "until", "form", "food", "keep", "children",
            "device", "thermostat", "temperature", "camera", "motion", "door", "lock", "unlock", "window", "alarm", "sensor",
            "status", "enabled", "disabled", "active", "inactive", "connect", "disconnect", "battery", "level", "percentage",
            "weather", "forecast", "humidity", "celsius", "fahrenheit", "rain", "snow", "sunny", "cloudy", "wind", "speed",
            "calendar", "event", "meeting", "reminder", "alarm", "clock", "timer", "schedule", "reschedule", "cancel", "attend",
            "email", "send", "receive", "inbox", "subject", "recipient", "attachment", "reply", "forward", "archive", "unread",
            "music", "play", "pause", "resume", "stop", "volume", "track", "artist", "album", "playlist", "genre", "shuffle",
            "map", "navigation", "route", "direction", "traffic", "distance", "estimate", "location", "address", "destination",
            "finance", "stock", "portfolio", "ticker", "currency", "convert", "rate", "transfer", "amount", "account", "balance",
            "notes", "create_note", "todo", "task", "priority", "due_date", "completed", "shopping_list", "item", "quantity",
        ]
        for w in words:
            expanded.append(f"{w} {w.upper()} {w.capitalize()}")
            expanded.append(f"\"name\": \"{w}\", \"parameters\": {{\"{w}\": \"value\"}}")

        out_dir = str(save_dir) if save_dir else None
        return cls.train_new(expanded, vocab_size=DEFAULT_VOCAB_SIZE, save_path=out_dir)

    def encode(self, text: str, add_special_tokens: bool = True) -> List[int]:
        """Encode text to token ids."""
        encoding = self._tokenizer.encode(text)
        ids = encoding.ids
        if add_special_tokens and (not ids or ids[0] != self.bos_token_id):
            return [self.bos_token_id] + ids
        return ids

    def decode(self, ids: List[int], skip_special_tokens: bool = False) -> str:
        """Decode token ids back to text."""
        return self._tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)

    def format_prompt(
        self,
        tools: Union[List[Union[ToolDefinition, Dict[str, Any]]], ToolRegistry],
        user_query: str,
        system_date: Optional[str] = "2026-09-27 Sun 17:00",
    ) -> str:
        """Format input into wire prompt protocol."""
        if isinstance(tools, ToolRegistry):
            tools_json = tools.to_wire_format()
        else:
            schemas = [t.to_dict() if isinstance(t, ToolDefinition) else t for t in tools]
            tools_json = f"<tools>{json.dumps(schemas, separators=(',', ':'))}</tools>"

        date_line = f"date: {system_date}\n" if system_date else ""
        prompt = (
            f"<s><|im_start|>system\n"
            f"{date_line}<|im_end|>\n"
            f"<|im_start|>user\n"
            f"{tools_json}\n"
            f"{user_query}<|im_end|>\n"
            f"<|im_start|>assistant\n"
        )
        return prompt

    def format_target(
        self,
        thought: Optional[str],
        tool_calls: List[Dict[str, Any]],
    ) -> str:
        """Format assistant response into wire protocol."""
        thought_str = f"<think>{thought}</think>\n" if thought else ""
        calls_json = json.dumps(tool_calls, separators=(',', ':'))
        return f"{thought_str}<tool_call>{calls_json}</tool_call><|im_end|></s>"

    def parse_assistant_response(self, text: str) -> Tuple[Optional[str], List[Dict[str, Any]]]:
        """Extract thought span and parsed tool call list from raw assistant output."""
        thought = None
        if "<think>" in text:
            if "</think>" in text:
                thought = text.split("<think>")[-1].split("</think>")[0].strip()
            elif "<tool_call>" in text:
                thought = text.split("<think>")[-1].split("<tool_call>")[0].strip()
            else:
                thought = text.split("<think>")[-1].strip()

        tool_calls = []
        call_match = re.search(r"<tool_call>(.*?)</tool_call>", text, re.DOTALL)
        if call_match:
            raw_json = call_match.group(1).strip()
            try:
                parsed = json.loads(raw_json)
                if isinstance(parsed, list):
                    tool_calls = parsed
                elif isinstance(parsed, dict):
                    tool_calls = [parsed]
            except Exception:
                pass
        return thought, tool_calls
