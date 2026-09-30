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
                # Infer or ground polarity from query using whole-word boundary matching
                pos_match = bool(re.search(r"\b(on|lock|close|shut|enable|start|yes|true)\b", lower_query))
                neg_match = bool(re.search(r"\b(off|unlock|open|disable|stop|no|false)\b", lower_query))
                if pos_match and not neg_match:
                    grounded[param_name] = True
                elif neg_match and not pos_match:
                    grounded[param_name] = False
                elif is_in_query:
                    grounded[param_name] = val
                else:
                    ungrounded.append(param_name)
            else:
                ungrounded.append(param_name)

        # Check required fields and recover ungrounded parameters from query
        for req in required:
            if req not in grounded:
                # 1. Check default
                if "default" in properties.get(req, {}):
                    grounded[req] = properties[req]["default"]
                    continue

                req_type = properties.get(req, {}).get("type", "string")

                # 2. Extract numeric values for integer/number params
                if req_type in ("integer", "number"):
                    num_match = re.search(r"\b(\d+)\b", query)
                    if num_match:
                        grounded[req] = int(num_match.group(1)) if req_type == "integer" else float(num_match.group(1))
                        continue

                # 3. Extract common entity slots based on parameter name
                if req == "city":
                    loc_match = re.search(r"\b(?:in|of|for|at)\s+([A-Za-z\s]+?)(?:\?|$|,|\.)", query, re.IGNORECASE)
                    if loc_match:
                        grounded[req] = loc_match.group(1).strip()
                        continue

                elif req == "room":
                    for rm in ["kitchen", "living room", "bedroom", "bathroom", "office", "garage", "hallway", "patio", "dining room"]:
                        if rm in lower_query:
                            grounded[req] = rm
                            break
                    if req in grounded:
                        continue

                elif req == "door":
                    for d in ["front", "back", "garage", "side", "bedroom", "kitchen", "office"]:
                        if d in lower_query:
                            grounded[req] = d
                            break
                    if req in grounded:
                        continue

                elif req == "state":
                    if re.search(r"\b(on|enable|brighten)\b", lower_query):
                        grounded[req] = "on"
                    elif re.search(r"\b(off|disable|dim)\b", lower_query):
                        grounded[req] = "off"

        return grounded, ungrounded
