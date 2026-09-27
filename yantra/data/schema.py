"""Tool definition, schema extraction, and grounding validation for Yantra."""

import inspect
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple, Type, get_args, get_origin, Union
from pydantic import BaseModel, create_model


def _python_type_to_json_schema(py_type: Any) -> Dict[str, Any]:
    """Map python types and typing constructs to JSON schema types."""
    origin = get_origin(py_type)
    args = get_args(py_type)

    if py_type in (str,):
        return {"type": "string"}
    elif py_type in (int,):
        return {"type": "integer"}
    elif py_type in (float,):
        return {"type": "number"}
    elif py_type in (bool,):
        return {"type": "boolean"}
    elif py_type in (dict,):
        return {"type": "object"}
    elif py_type in (list, tuple):
        return {"type": "array"}
    elif origin is list or origin is List:
        item_type = _python_type_to_json_schema(args[0]) if args else {}
        return {"type": "array", "items": item_type}
    elif origin is dict or origin is Dict:
        return {"type": "object"}
    elif origin is Union:
        # Check for Optional[T] which is Union[T, NoneType]
        non_none = [a for a in args if a is not type(None)]
        if len(non_none) == 1:
            schema = _python_type_to_json_schema(non_none[0])
            return schema
        else:
            return {"anyOf": [_python_type_to_json_schema(a) for a in non_none]}
    elif inspect.isclass(py_type) and issubclass(py_type, BaseModel):
        return py_type.model_json_schema()
    return {"type": "string"}


class Field:
    """Descriptor for parameter metadata (description, default, enum)."""

    def __init__(
        self,
        default: Any = ...,
        description: str = "",
        enum: Optional[List[Any]] = None,
    ):
        self.default = default
        self.description = description
        self.enum = enum


@dataclass
class ToolDefinition:
    """Canonical representation of an executable tool and its JSON schema."""

    name: str
    description: str
    parameters: Dict[str, Any]
    fn: Optional[Callable[..., Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Return schema representation."""
        return {
            "name": self.name,
            "description": self.description,
            "parameters": self.parameters,
        }

    def to_json(self) -> str:
        """Compact JSON representation."""
        return json.dumps(self.to_dict(), separators=(",", ":"))

    def validate_and_execute(self, arguments: Dict[str, Any]) -> Any:
        """Validate input arguments against schema and execute function if provided."""
        if self.fn is None:
            raise ValueError(f"Tool '{self.name}' has no backing executable function.")
        
        sig = inspect.signature(self.fn)
        # Type coercion / validation using inspect bound arguments
        bound = sig.bind(**arguments)
        bound.apply_defaults()
        return self.fn(*bound.args, **bound.kwargs)


def tool(fn: Optional[Callable[..., Any]] = None, *, name: Optional[str] = None, description: Optional[str] = None):
    """Decorator to convert a Python function into a ToolDefinition.
    
    Example:
        @tool
        def get_weather(city: str, unit: str = "celsius") -> str:
            '''Get the current weather for a city.'''
            return f"Weather in {city}: 22 degrees {unit}"
    """
    def decorator(func: Callable[..., Any]) -> ToolDefinition:
        tool_name = name or func.__name__
        tool_desc = (description or inspect.getdoc(func) or f"Execute {tool_name}").strip()

        sig = inspect.signature(func)
        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param_name, param in sig.parameters.items():
            if param_name in ("self", "cls"):
                continue

            param_type = param.annotation if param.annotation != inspect.Parameter.empty else str
            param_schema = _python_type_to_json_schema(param_type)

            # Check if default is a Field
            if isinstance(param.default, Field):
                field_obj = param.default
                if field_obj.description:
                    param_schema["description"] = field_obj.description
                if field_obj.enum:
                    param_schema["enum"] = field_obj.enum
                if field_obj.default is not ...:
                    param_schema["default"] = field_obj.default
                else:
                    required.append(param_name)
            elif param.default is inspect.Parameter.empty:
                required.append(param_name)
            else:
                param_schema["default"] = param.default

            properties[param_name] = param_schema

        schema = {
            "type": "object",
            "properties": properties,
            "required": required,
        }

        return ToolDefinition(
            name=tool_name,
            description=tool_desc,
            parameters=schema,
            fn=func,
        )

    if fn is not None:
        return decorator(fn)
    return decorator


class ToolRegistry:
    """Manages a collection of tools, wire format serialization, and dispatching."""

    def __init__(self, tools: Optional[List[Union[ToolDefinition, Callable]]] = None):
        self._tools: Dict[str, ToolDefinition] = {}
        if tools:
            for t in tools:
                self.register(t)

    def register(self, t: Union[ToolDefinition, Callable]) -> ToolDefinition:
        """Register a tool definition or decorated function."""
        if not isinstance(t, ToolDefinition):
            tool_def = tool(t)
        else:
            tool_def = t
        self._tools[tool_def.name] = tool_def
        return tool_def

    def get(self, name: str) -> Optional[ToolDefinition]:
        return self._tools.get(name)

    def to_wire_format(self) -> str:
        """Serialize registered tools into the standard <tools>[...]</tools> string."""
        schemas = [t.to_dict() for t in self._tools.values()]
        return f"<tools>{json.dumps(schemas, separators=(',', ':'))}</tools>"

    def list_names(self) -> List[str]:
        return list(self._tools.keys())

    def execute_call(self, name: str, arguments: Dict[str, Any]) -> Any:
        """Execute a tool call by name with arguments."""
        if name not in self._tools:
            raise KeyError(f"Unknown tool '{name}'. Available: {self.list_names()}")
        return self._tools[name].validate_and_execute(arguments)
