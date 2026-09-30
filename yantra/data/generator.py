"""Synthetic Data Generation Pipeline for Yantra Tool Calling Specialist."""

import json
import random
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple


# Rich diverse catalog of seed tool schemas
SEED_TOOLS: List[Dict[str, Any]] = [
    {
        "name": "set_thermostat",
        "description": "Adjust HVAC thermostat temperature and mode",
        "parameters": {
            "type": "object",
            "properties": {
                "temperature": {"type": "integer", "description": "Target temperature in degrees"},
                "mode": {"type": "string", "enum": ["cool", "heat", "eco", "auto"], "default": "auto"},
                "unit": {"type": "string", "enum": ["fahrenheit", "celsius"], "default": "fahrenheit"},
            },
            "required": ["temperature"],
        },
        "templates": [
            ("set the thermostat to {temp} degrees", {"temperature": "{temp}"}),
            ("set thermostat to {temp}", {"temperature": "{temp}"}),
            ("make it {temp} degrees in here", {"temperature": "{temp}"}),
            ("make it {temp} degrees", {"temperature": "{temp}"}),
            ("turn the heat to {temp}", {"temperature": "{temp}", "mode": "heat"}),
            ("cool down the room to {temp} fahrenheit", {"temperature": "{temp}", "mode": "cool", "unit": "fahrenheit"}),
            ("set AC to {temp} celsius", {"temperature": "{temp}", "mode": "cool", "unit": "celsius"}),
            ("adjust temperature to {temp}", {"temperature": "{temp}"}),
        ],
        "args_pool": {
            "temp": [68, 70, 72, 74, 75, 76, 19, 21, 22, 24],
        },
    },
    {
        "name": "control_light",
        "description": "Turn on/off or dim lighting fixtures in a designated room",
        "parameters": {
            "type": "object",
            "properties": {
                "room": {"type": "string", "description": "Room name"},
                "state": {"type": "string", "enum": ["on", "off"], "description": "Power state"},
                "brightness": {"type": "integer", "description": "Brightness percentage 0-100"},
            },
            "required": ["room", "state"],
        },
        "templates": [
            ("turn on the {room} lights", {"room": "{room}", "state": "on"}),
            ("turn on {room} light", {"room": "{room}", "state": "on"}),
            ("switch on {room} lights", {"room": "{room}", "state": "on"}),
            ("switch off the lights in the {room}", {"room": "{room}", "state": "off"}),
            ("turn off {room} lights", {"room": "{room}", "state": "off"}),
            ("turn off {room}", {"room": "{room}", "state": "off"}),
            ("dim {room} light to {brightness} percent", {"room": "{room}", "state": "on", "brightness": "{brightness}"}),
            ("set the {room} brightness to {brightness}", {"room": "{room}", "state": "on", "brightness": "{brightness}"}),
        ],
        "args_pool": {
            "room": ["living room", "bedroom", "kitchen", "office", "garage", "hallway", "bathroom", "patio"],
            "brightness": [10, 25, 50, 75, 80, 100],
        },
    },
    {
        "name": "get_weather",
        "description": "Fetch current weather and atmospheric conditions for a city",
        "parameters": {
            "type": "object",
            "properties": {
                "city": {"type": "string", "description": "City name"},
                "forecast_days": {"type": "integer", "description": "Days of forecast (1-7)", "default": 1},
            },
            "required": ["city"],
        },
        "templates": [
            ("what's the weather in {city}?", {"city": "{city}"}),
            ("what is the weather in {city}", {"city": "{city}"}),
            ("what's the weather like in {city}", {"city": "{city}"}),
            ("how's the weather in {city}", {"city": "{city}"}),
            ("how is the weather in {city}", {"city": "{city}"}),
            ("weather in {city}", {"city": "{city}"}),
            ("weather of {city}", {"city": "{city}"}),
            ("fetch weather of {city}", {"city": "{city}"}),
            ("fetch the weather for {city}", {"city": "{city}"}),
            ("get weather for {city}", {"city": "{city}"}),
            ("check the weather in {city}", {"city": "{city}"}),
            ("tell me the weather in {city}", {"city": "{city}"}),
            ("is it raining in {city} today?", {"city": "{city}"}),
            ("give me the {days} day forecast for {city}", {"city": "{city}", "forecast_days": "{days}"}),
            ("current temperature in {city}", {"city": "{city}"}),
            ("how's the weather like in {city} right now?", {"city": "{city}"}),
        ],
        "args_pool": {
            "city": [
                "Seattle", "Tokyo", "London", "San Francisco", "New York", "Paris", "Berlin",
                "Bengaluru", "Sydney", "Toronto", "India", "Chicago", "Boston", "Mumbai",
                "Delhi", "California", "Texas", "Austin", "Miami", "Singapore"
            ],
            "days": [1, 3, 5, 7],
        },
    },
    {
        "name": "create_calendar_event",
        "description": "Create an event on the user's primary calendar",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Title of the meeting or event"},
                "date": {"type": "string", "description": "Event date (YYYY-MM-DD or relative like 'tomorrow')"},
                "time": {"type": "string", "description": "Start time (e.g. '15:00', '3pm')"},
                "duration_minutes": {"type": "integer", "description": "Duration in minutes", "default": 30},
            },
            "required": ["title", "time"],
        },
        "templates": [
            ("schedule {title} at {time}", {"title": "{title}", "time": "{time}"}),
            ("schedule a meeting called {title} at {time}", {"title": "{title}", "time": "{time}"}),
            ("add {title} to my calendar tomorrow at {time}", {"title": "{title}", "date": "tomorrow", "time": "{time}"}),
            ("book a {duration} minute meeting for {title} at {time}", {"title": "{title}", "time": "{time}", "duration_minutes": "{duration}"}),
            ("put {title} on my schedule at {time}", {"title": "{title}", "time": "{time}"}),
            ("set up {title} at {time}", {"title": "{title}", "time": "{time}"}),
        ],
        "args_pool": {
            "title": ["Team Sync", "Dentist Appointment", "1-on-1 with Alex", "Project Review", "Coffee with Sarah", "Gym Session"],
            "time": ["9am", "11:30am", "2pm", "3:30pm", "4pm", "5:00pm"],
            "duration": [15, 30, 45, 60],
        },
    },
    {
        "name": "set_timer",
        "description": "Set a countdown timer on the device",
        "parameters": {
            "type": "object",
            "properties": {
                "minutes": {"type": "integer", "description": "Timer duration in minutes"},
                "label": {"type": "string", "description": "Optional timer label"},
            },
            "required": ["minutes"],
        },
        "templates": [
            ("set a timer for {minutes} minutes", {"minutes": "{minutes}"}),
            ("start a {minutes} minute timer for {label}", {"minutes": "{minutes}", "label": "{label}"}),
            ("set timer for {minutes} minutes", {"minutes": "{minutes}"}),
            ("timer for {minutes} minutes", {"minutes": "{minutes}"}),
            ("time {minutes} minutes", {"minutes": "{minutes}"}),
            ("remind me in {minutes} minutes to {label}", {"minutes": "{minutes}", "label": "{label}"}),
        ],
        "args_pool": {
            "minutes": [5, 10, 15, 20, 25, 30, 45],
            "label": ["pasta", "tea", "laundry", "study break", "pizza"],
        },
    },
    {
        "name": "send_message",
        "description": "Send a text or chat message to a contact",
        "parameters": {
            "type": "object",
            "properties": {
                "recipient": {"type": "string", "description": "Name or phone number of contact"},
                "message": {"type": "string", "description": "Body text of the message"},
            },
            "required": ["recipient", "message"],
        },
        "templates": [
            ("send a message to {recipient} saying {msg}", {"recipient": "{recipient}", "message": "{msg}"}),
            ("text {recipient} that {msg}", {"recipient": "{recipient}", "message": "{msg}"}),
            ("tell {recipient} {msg}", {"recipient": "{recipient}", "message": "{msg}"}),
            ("send text to {recipient} saying {msg}", {"recipient": "{recipient}", "message": "{msg}"}),
            ("message {recipient}: {msg}", {"recipient": "{recipient}", "message": "{msg}"}),
        ],
        "args_pool": {
            "recipient": ["Alice", "Bob", "Mom", "Dad", "David", "Emma"],
            "msg": ["I'm running 5 minutes late", "Dinner is ready", "Can we reschedule?", "See you at the office!"],
        },
    },
    {
        "name": "lock_door",
        "description": "Lock or unlock smart security door lock",
        "parameters": {
            "type": "object",
            "properties": {
                "door": {"type": "string", "description": "Door identifier (front, back, garage)"},
                "locked": {"type": "boolean", "description": "True to lock, False to unlock"},
            },
            "required": ["door", "locked"],
        },
        "templates": [
            ("lock the {door} door", {"door": "{door}", "locked": True}),
            ("lock {door} door", {"door": "{door}", "locked": True}),
            ("unlock the {door} door", {"door": "{door}", "locked": False}),
            ("unlock {door} door", {"door": "{door}", "locked": False}),
            ("open the {door} door", {"door": "{door}", "locked": False}),
            ("open {door} door", {"door": "{door}", "locked": False}),
            ("close the {door} door", {"door": "{door}", "locked": True}),
            ("close {door} door", {"door": "{door}", "locked": True}),
            ("shut the {door} door", {"door": "{door}", "locked": True}),
            ("make sure the {door} door is locked", {"door": "{door}", "locked": True}),
        ],
        "args_pool": {
            "door": ["front", "back", "garage", "side"],
        },
    },
]

# Hard negative chit-chat / unsupported queries
UNSUPPORTED_QUERIES = [
    "What is the capital of France?",
    "Can you write a poem about the ocean?",
    "Explain quantum entanglement in simple terms.",
    "Who won the 1994 World Cup?",
    "How do I bake sourdough bread?",
    "Solve the quadratic equation x^2 + 5x + 6 = 0.",
    "Tell me a funny joke.",
    "What is your opinion on artificial intelligence?",
    "Recommend a good sci-fi movie from the 80s.",
    "Translate 'hello world' to Spanish.",
    "Summarize the plot of Hamlet.",
]


@dataclass
class SyntheticExample:
    """A generated training turn with tools, user query, thought span, and target tool call(s)."""
    tools: List[Dict[str, Any]]
    query: str
    thought: str
    tool_calls: List[Dict[str, Any]]
    category: str  # 'single', 'multi', 'refusal', 'distractor'


class SyntheticDataGenerator:
    """Generates balanced synthetic datasets for tool dispatching models."""

    def __init__(self, seed_tools: Optional[List[Dict[str, Any]]] = None, seed: int = 42):
        self.seed_tools = seed_tools or SEED_TOOLS
        self.rng = random.Random(seed)

    def _sample_single_positive(self) -> SyntheticExample:
        """Sample a single tool call request."""
        tool_spec = self.rng.choice(self.seed_tools)
        template, arg_map = self.rng.choice(tool_spec["templates"])

        # Fill args
        resolved_args = {}
        query_text = template
        for placeholder, value in arg_map.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                pool_key = value[1:-1]
                choice = self.rng.choice(tool_spec["args_pool"][pool_key])
                resolved_args[placeholder] = choice
                query_text = query_text.replace(value, str(choice))
            else:
                resolved_args[placeholder] = value

        # Tools list: either just this tool or with 1-2 random other tools
        other_tools = [t for t in self.seed_tools if t["name"] != tool_spec["name"]]
        num_others = self.rng.randint(0, 2)
        active_tools = [self._extract_schema(tool_spec)] + [
            self._extract_schema(t) for t in self.rng.sample(other_tools, num_others)
        ]
        self.rng.shuffle(active_tools)

        thought = f"Intent matches '{tool_spec['name']}' with args: {resolved_args}"
        tool_call = [{"name": tool_spec["name"], "arguments": resolved_args}]

        return SyntheticExample(
            tools=active_tools,
            query=query_text,
            thought=thought,
            tool_calls=tool_call,
            category="single",
        )

    def _sample_multi_call(self) -> SyntheticExample:
        """Sample a multi-tool parallel or compound request (e.g. light + thermostat)."""
        tools_subset = self.rng.sample(self.seed_tools, 2)
        queries = []
        calls = []

        for tool_spec in tools_subset:
            template, arg_map = self.rng.choice(tool_spec["templates"])
            resolved_args = {}
            query_text = template
            for placeholder, value in arg_map.items():
                if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                    pool_key = value[1:-1]
                    choice = self.rng.choice(tool_spec["args_pool"][pool_key])
                    resolved_args[placeholder] = choice
                    query_text = query_text.replace(value, str(choice))
                else:
                    resolved_args[placeholder] = value
            queries.append(query_text)
            calls.append({"name": tool_spec["name"], "arguments": resolved_args})

        connector = self.rng.choice([" and ", " and also ", ", then ", " and please "])
        compound_query = connector.join(queries)

        all_schemas = [self._extract_schema(t) for t in self.seed_tools]
        self.rng.shuffle(all_schemas)

        names = [c["name"] for c in calls]
        thought = f"Compound intent requires tools: {', '.join(names)}"

        return SyntheticExample(
            tools=all_schemas[:4],
            query=compound_query,
            thought=thought,
            tool_calls=calls,
            category="multi",
        )

    def _sample_hard_negative(self) -> SyntheticExample:
        """Sample a refusal / out-of-scope query where target output is []."""
        # 50% chit-chat / general knowledge refusal
        # 50% out-of-toolset refusal (user asks for a tool that is not provided in active tools)
        if self.rng.random() < 0.5:
            query = self.rng.choice(UNSUPPORTED_QUERIES)
            num_tools = self.rng.randint(1, 4)
            sampled = self.rng.sample(self.seed_tools, num_tools)
            schemas = [self._extract_schema(t) for t in sampled]
            thought = "Query is chit-chat or out-of-scope; no available tool matches."
        else:
            # Pick a target tool to generate query from
            target_tool = self.rng.choice(self.seed_tools)
            template, arg_map = self.rng.choice(target_tool["templates"])
            resolved_args = {}
            query = template
            for placeholder, value in arg_map.items():
                if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                    pool_key = value[1:-1]
                    choice = self.rng.choice(target_tool["args_pool"][pool_key])
                    resolved_args[placeholder] = choice
                    query = query.replace(value, str(choice))
                else:
                    resolved_args[placeholder] = value

            # Active tools specifically EXCLUDE target_tool!
            other_tools = [t for t in self.seed_tools if t["name"] != target_tool["name"]]
            num_others = self.rng.randint(1, min(3, len(other_tools)))
            sampled_others = self.rng.sample(other_tools, num_others)
            schemas = [self._extract_schema(t) for t in sampled_others]
            thought = f"Required tool '{target_tool['name']}' is not available in provided tools."

        return SyntheticExample(
            tools=schemas,
            query=query,
            thought=thought,
            tool_calls=[],
            category="refusal",
        )

    def _sample_distractor(self) -> SyntheticExample:
        """Sample single tool with 4-6 distractor tools."""
        tool_spec = self.rng.choice(self.seed_tools)
        template, arg_map = self.rng.choice(tool_spec["templates"])

        resolved_args = {}
        query_text = template
        for placeholder, value in arg_map.items():
            if isinstance(value, str) and value.startswith("{") and value.endswith("}"):
                pool_key = value[1:-1]
                choice = self.rng.choice(tool_spec["args_pool"][pool_key])
                resolved_args[placeholder] = choice
                query_text = query_text.replace(value, str(choice))
            else:
                resolved_args[placeholder] = value

        other_tools = [t for t in self.seed_tools if t["name"] != tool_spec["name"]]
        all_schemas = [self._extract_schema(tool_spec)] + [self._extract_schema(t) for t in other_tools]
        self.rng.shuffle(all_schemas)

        thought = f"Disregard distractor tools, match '{tool_spec['name']}'"
        tool_call = [{"name": tool_spec["name"], "arguments": resolved_args}]

        return SyntheticExample(
            tools=all_schemas,
            query=query_text,
            thought=thought,
            tool_calls=tool_call,
            category="distractor",
        )

    def _extract_schema(self, tool_spec: Dict[str, Any]) -> Dict[str, Any]:
        return {
            "name": tool_spec["name"],
            "description": tool_spec["description"],
            "parameters": tool_spec["parameters"],
        }

    def generate_dataset(
        self,
        num_samples: int = 1000,
        ratios: Optional[Dict[str, float]] = None,
    ) -> List[SyntheticExample]:
        """Generate balanced dataset across the specified distribution."""
        ratios = ratios or {
            "single": 0.50,
            "multi": 0.15,
            "refusal": 0.25,
            "distractor": 0.10,
        }

        examples = []
        counts = {cat: int(num_samples * weight) for cat, weight in ratios.items()}
        # Handle rounding remainder
        counts["single"] += num_samples - sum(counts.values())

        for _ in range(counts["single"]):
            examples.append(self._sample_single_positive())
        for _ in range(counts["multi"]):
            examples.append(self._sample_multi_call())
        for _ in range(counts["refusal"]):
            examples.append(self._sample_hard_negative())
        for _ in range(counts["distractor"]):
            examples.append(self._sample_distractor())

        self.rng.shuffle(examples)
        return examples

    def save_jsonl(self, examples: List[SyntheticExample], path: str):
        """Save examples to JSONL file."""
        with open(path, "w", encoding="utf-8") as f:
            for ex in examples:
                record = {
                    "tools": ex.tools,
                    "query": ex.query,
                    "thought": ex.thought,
                    "tool_calls": ex.tool_calls,
                    "category": ex.category,
                }
                f.write(json.dumps(record) + "\n")
