"""High-level Agent runtime and Python SDK for Yantra Tool Dispatching."""

from dataclasses import dataclass, field
import re
from typing import Any, Callable, Dict, List, Optional, Union

import torch

from yantra.data.schema import ToolDefinition, ToolRegistry, tool
from yantra.model.config import YantraConfig
from yantra.model.transformer import YantraForToolCalling
from yantra.runtime.grammar import JSONGrammarValidator
from yantra.tokenizer.tokenization import YantraTokenizer


@dataclass
class AgentResponse:
    """Structured result returned by Yantra Agent execution."""

    query: str
    thought: Optional[str] = None
    tool_calls: List[Dict[str, Any]] = field(default_factory=list)
    results: List[Any] = field(default_factory=list)
    raw_completion: str = ""
    confidence: float = 1.0
    is_refusal: bool = False
    ungrounded_fields: List[str] = field(default_factory=list)

    def __repr__(self) -> str:
        if self.is_refusal:
            return f"<AgentResponse: Refusal / No Tool Dispatched (confidence={self.confidence:.2f})>"
        calls_str = ", ".join(f"{c['name']}({c.get('arguments', {})})" for c in self.tool_calls)
        return f"<AgentResponse: calls=[{calls_str}] (confidence={self.confidence:.2f})>"


class Agent:
    """Specialized lightweight on-device agent for fast function calling and tool execution."""

    def __init__(
        self,
        tools: Optional[List[Union[ToolDefinition, Callable[..., Any]]]] = None,
        model: Optional[Union[YantraForToolCalling, str]] = None,
        tokenizer: Optional[YantraTokenizer] = None,
        device: Optional[str] = None,
        strict: bool = False,
        confidence_threshold: float = 0.25,
    ):
        self.registry = ToolRegistry(tools or [])
        self.strict = strict
        self.confidence_threshold = confidence_threshold

        # Device selection
        if device:
            self.device = torch.device(device)
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            self.device = torch.device("mps")
        else:
            self.device = torch.device("cpu")

        # Load or initialize model
        if isinstance(model, str):
            from yantra.runtime.quantize import load_compressed_model
            self.model = load_compressed_model(model, device=str(self.device))
        elif isinstance(model, YantraForToolCalling):
            self.model = model.to(self.device)
            self.model.eval()
        else:
            # Instantiate default model
            self.model = YantraForToolCalling(YantraConfig()).to(self.device)
            self.model.eval()

        self.tokenizer = tokenizer or YantraTokenizer.load()

    def add_tool(self, t: Union[ToolDefinition, Callable[..., Any]]) -> ToolDefinition:
        """Register an additional tool."""
        return self.registry.register(t)

    def run(
        self,
        query: str,
        execute_tools: bool = True,
        max_new_tokens: int = 256,
        temperature: float = 0.0,
        system_date: Optional[str] = "2026-09-27 Sun 17:00",
    ) -> AgentResponse:
        """Execute intent resolution, generate tool calls, validate grounding, and dispatch."""
        # 1. Format wire prompt
        prompt = self.tokenizer.format_prompt(
            tools=self.registry,
            user_query=query,
            system_date=system_date,
        )

        input_ids = torch.tensor(
            [self.tokenizer.encode(prompt, add_special_tokens=True)],
            dtype=torch.long,
            device=self.device,
        )

        # 2. Autoregressive generation with KV caching
        stop_ids = [self.tokenizer.im_end_id, self.tokenizer.tool_call_end_id, self.tokenizer.eos_token_id]
        with torch.no_grad():
            output_tokens = self.model.generate(
                input_ids=input_ids,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                stop_token_ids=stop_ids,
            )
            # Evaluate confidence score on initial prompt
            prompt_outputs = self.model(input_ids)
            conf_val = 1.0
            if prompt_outputs["confidence"] is not None:
                conf_val = prompt_outputs["confidence"].item()

        # 3. Decode completion tokens
        gen_tokens = output_tokens[0, input_ids.shape[1] :].tolist()
        raw_completion = self.tokenizer.decode(gen_tokens, skip_special_tokens=False)

        # 4. Parse thought and tool call
        thought, tool_calls = self.tokenizer.parse_assistant_response(raw_completion)

        # If standard parse yielded nothing, try grammar repair on raw text
        if not tool_calls and "<tool_call>" in raw_completion:
            call_span = raw_completion.split("<tool_call>")[-1].split("</tool_call>")[0]
            repaired = JSONGrammarValidator.clean_and_repair_json(call_span)
            if isinstance(repaired, list):
                tool_calls = repaired
            elif isinstance(repaired, dict):
                tool_calls = [repaired]

        # 5. Check confidence threshold for refusal
        if conf_val < self.confidence_threshold and not tool_calls:
            return AgentResponse(
                query=query,
                thought=thought or "Low confidence refusal",
                tool_calls=[],
                results=[],
                raw_completion=raw_completion,
                confidence=conf_val,
                is_refusal=True,
            )

        # Reconcile tool call name with explicit intent reasoning in thought if drifted
        if thought and tool_calls:
            intent_match = re.search(r"Intent matches '(\w+)'", thought)
            if intent_match:
                intended_tool = intent_match.group(1)
                if self.registry.get(intended_tool):
                    for call in tool_calls:
                        if call.get("name") != intended_tool:
                            call["name"] = intended_tool

        # 6. Argument grounding & validation against schemas
        validated_calls = []
        all_ungrounded = []

        for call in tool_calls:
            name = call.get("name")
            tool_def = self.registry.get(name)
            if not tool_def:
                continue

            grounded_args, ungrounded = JSONGrammarValidator.ground_arguments(
                query=query,
                tool_call=call,
                schema=tool_def.to_dict(),
                strict=self.strict,
            )
            all_ungrounded.extend(ungrounded)

            if self.strict and ungrounded:
                # Refuse ungrounded calls in strict mode
                continue

            validated_calls.append({"name": name, "arguments": grounded_args})

        # 7. Execute tools if requested
        results = []
        if execute_tools and validated_calls:
            for call in validated_calls:
                try:
                    res = self.registry.execute_call(call["name"], call.get("arguments", {}))
                    results.append(res)
                except Exception as e:
                    results.append({"error": str(e)})

        is_refusal = len(validated_calls) == 0

        return AgentResponse(
            query=query,
            thought=thought,
            tool_calls=validated_calls,
            results=results,
            raw_completion=raw_completion,
            confidence=conf_val,
            is_refusal=is_refusal,
            ungrounded_fields=all_ungrounded,
        )
