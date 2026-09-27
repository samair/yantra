"""Grammar constraints, deterministic repair, and argument grounding for Yantra."""

import json
import re
from typing import Any, Dict, List, Optional, Set, Tuple


class JSONGrammarValidator:
    """Validates and deterministically repairs generated JSON tool call strings."""

    @staticmethod
    def clean_and_repair_json(raw_text: str) -> Optional[Any]:
        """Attempt to parse or deterministically repair truncated/malformed JSON strings."""
        raw_text = raw_text.strip()
        if not raw_text:
            return []

        # 1. Try direct JSON parsing
        try:
            return json.loads(raw_text)
        except json.JSONDecodeError:
            pass

        # 2. Extract substring between '[' and ']' or '{' and '}'
        bracket_start = raw_text.find("[")
        brace_start = raw_text.find("{")

        if bracket_start != -1 and (brace_start == -1 or bracket_start < brace_start):
            candidate = raw_text[bracket_start:]
            # Count open brackets and balance them
            open_count = candidate.count("[") - candidate.count("]")
            open_brace = candidate.count("{") - candidate.count("}")
            candidate = candidate + ("}" * max(0, open_brace)) + ("]" * max(0, open_count))
            try:
                return json.loads(candidate)
            except json.JSONDecodeError:
                pass

        if brace_start != -1:
            candidate = raw_text[brace_start:]
            open_brace = candidate.count("{") - candidate.count("}")
            candidate = candidate + ("}" * max(0, open_brace))
            try:
                parsed = json.loads(candidate)
                return [parsed] if isinstance(parsed, dict) else parsed
            except json.JSONDecodeError:
                pass

        return None

    @staticmethod
    def ground_arguments(
        query: str,
        tool_call: Dict[str, Any],
        schema: Dict[str, Any],
        strict: bool = False,
    ) -> Tuple[Dict[str, Any], List[str]]:
        """Validate that tool call arguments are grounded in the query or schema defaults.
        
        Returns:
            Tuple of (grounded_arguments, list_of_ungrounded_fields)
        """
        raw_args = tool_call.get("arguments", {})
        properties = schema.get("parameters", {}).get("properties", {})
        required = set(schema.get("parameters", {}).get("required", []))

        grounded = {}
        ungrounded = []
        lower_query = query.lower()

        for param_name, val in raw_args.items():
            if param_name not in properties:
                ungrounded.append(param_name)
                continue

            param_meta = properties[param_name]
            str_val = str(val).lower()

            # Numbers, booleans, or direct string matches in query
            is_in_query = str_val in lower_query or (isinstance(val, (int, float)) and str_val in lower_query)
            has_default = "default" in param_meta

            if is_in_query:
                grounded[param_name] = val
            elif has_default and val == param_meta["default"]:
                grounded[param_name] = val
            elif isinstance(val, bool):
                # Check for polar polarity words
                if val is True and any(w in lower_query for w in ["on", "lock", "enable", "start", "yes", "true"]):
                    grounded[param_name] = True
                elif val is False and any(w in lower_query for w in ["off", "unlock", "disable", "stop", "no", "false"]):
                    grounded[param_name] = False
                else:
                    ungrounded.append(param_name)
            else:
                ungrounded.append(param_name)

        # Check required fields
        for req in required:
            if req not in grounded:
                if "default" in properties.get(req, {}):
                    grounded[req] = properties[req]["default"]

        return grounded, ungrounded
