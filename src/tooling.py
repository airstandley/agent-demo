from dataclasses import dataclass
from inspect import Signature, Parameter, signature
from collections import OrderedDict
from functools import wraps
import json

@dataclass
class FunctionMetadata:
    name: str
    description: str
    arguement_schema: dict

class Tool:
    def __init__(self, function: callable, name: str | None = None, description: str | None = None):
        self.metadata: FunctionMetadata = self._inspect_function(function)
        self.name = name if name is not None else self.metadata.name
        self.description = description if description is not None else self.metadata.description
        self.schema = self.metadata.arguement_schema
        self.function = function
    
    @classmethod
    def _inspect_function(cls, function: callable) -> FunctionMetadata:
        metadata = FunctionMetadata(
            name=function.__name__,
            description=function.__doc__,
            arguement_schema=cls._build_arguement_schema(signature(function))
        )
        return metadata
    
    @staticmethod
    def _build_arguement_schema(function_signature: Signature) -> dict:
        schema = OrderedDict()
        for name, parameter in function_signature.parameters.items():
            parameter_schema = {}
            if parameter.annotation != Parameter.empty:
                type_map = {
                    str: "string",
                    int: "integer",
                    float: "number",
                    bool: "boolean",
                    list: "array",
                    dict: "object"
                }
                parameter_schema["type"] = type_map[parameter.annotation] if parameter.annotation in type_map else parameter.annotation.__name__
            if not parameter.default:
                parameter_schema["required"] = "True"
            schema[name] = parameter_schema
        return schema

    def __str__(self):
        return f"""- Name: {self.name}
  Description: {self.description}
  Arguements: {json.dumps(self.schema)}"""

    def __call__(self, *args, **kwds):
        return self.function(*args, **kwds)

# Idea: Add basic RBAC so that tools can more easily be provisioned to agents based on their permission levels.
tool_registry = {}

def tool(function):
    """Decorator to register a tool"""
    tool = Tool(function)
    tool_registry[tool.name] = tool
    return function