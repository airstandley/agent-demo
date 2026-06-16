# agent.py
from enum import StrEnum, EnumMeta
import json
import logging
from typing import List, Dict, Callable
from dataclasses import dataclass

from ._types import Client, ModelOptions, Tool

logger = logging.getLogger(__name__)

class AgentStateMeta(EnumMeta):
    def __new__(cls, name, bases, dct):
        # Add values list to class
        klass = super().__new__(cls, name, bases, dct)
        klass.values = [str(value) for value in klass.__members__.values()]
        return klass

class AgentStates(StrEnum, metaclass=AgentStateMeta):
    RECIEVED_PROMPT = "recieved_prompt"
    TAKE_ACTION = "take_action"
    OBSERVE_ACTION_RESULT = "observe_action_result"
    GAVE_ANSWER = "gave_answer"


class Agent:

    def __init__(
            self, 
            client: Client, 
            system_prompt: str | None = None,
            options: ModelOptions | None = None,
            schema: Dict | None = None,
            tools: Dict[Tool] | None = None,
            stateful: bool = True,
        ):
        self.system_prompt = system_prompt
        self.model_options = options
        self.schema = schema
        self.retries = 3  # Number to times to attempt a valid generation per step
        self.client = client
        self.tools = tools
        if self.tools:
            self.system_prompt = "You are a tooling calling agent. Use the tools available to you. To call a tool respond using the 'tool_calls' property.\n" + self.system_prompt
            self.schema["tool_calls"] = {
                "type": "array", 
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "arguements": {"type": "array"}
                    }
                },
                "required": "False"
            }
        self.state = None
        if stateful:
            self.state = AgentStates.RECIEVED_PROMPT
            self.system_prompt = "/n".join([
                "You are an agent. You must decide on the next action.",
                "You must choose ONE of the following options:",
                *AgentStates.values,
                self.system_prompt
            ])
            self.schema["state"] = f"enum({"|".join(AgentStates.values)}"

    def _generate(self, message: str, options: ModelOptions | None = None) -> str:
        prompt = []
        # Add System Prompt for Agents
        if self.system_prompt:
            prompt += [
                "<SYSTEM>",
                self.system_prompt,
                "</SYSTEM>"
            ]
        # Add State Information if Stateful
        if self.state is not None:
            prompt += [
                "<STATE>",
                f"Current: {self.state}",
                f"Available States: {",".join(AgentStates.values)}",
                "</STATE>"
            ]
        # Add Tool Definitions if Defined
        if self.tools:
            prompt += [
                "<TOOLS>",
                "\n".join([str(tool) for tool in self.tools.values()]),
                "</TOOLS>"
            ]
        # Add Output Format Restrictions if Defined
        if self.schema:
            prompt += [
                "<FORMAT RULES>",
                "1. The response MUST be valid JSON",
                "2. Responses must follow the following schema:",
                f"{self.schema}",
                "3. No explanations, no markdown, no extra text before or after the JSON",
            ]
            if self.tools:
                prompt += ["4. Requests to call tools can be passed in this formatted response."]
            prompt += [
                "</FORMAT RULES>"
            ]
        # Add user message
        prompt += [
            "<USER>",
            message,
            "</USER>"
        ]
        # Generate the response
        prompt = "\n".join(prompt)
        logger.debug(f"Full Prompt:\n{prompt}")
        return self.client.generate(prompt, options=options if options is not None else self.model_options)

    def _parse_response_for_json(self, response: str) -> Dict:
        # Attempt to wrangle the garbage LLMs can generate into a usable JSON format.
        # Thankfully since these things are statistically predictors the kinds of mistakes they make are somewhat predictable.

        # 1. Extract only the text between the opening and closed brackets
        start = response.find('{')
        end = response.rfind('}')
        json_text = response[start:end+1]
        logger.debug(f"JSON: {json_text}")

        return json.loads(json_text)

    def generate(self, message: str, options: ModelOptions | None = None) -> Dict | str:
        if self.schema:
            for attempt in range(self.retries):
                response = self._generate(message)
                logger.debug(f"Raw Response Attempt {attempt+1}:\n{response}")
                try:
                    parsed = self._parse_response_for_json(response)
                except Exception as e:
                    logger.warning(f"LLM Output Parsing Failed! Reason: {e} Text:{response}")
                else:
                    return parsed
            raise RuntimeError("The model failed to generate formatted output")
        else:
            return self._generate(message, options=options)
    
    def excute_tool_call(self, name: str, arguements: list) -> str:
        logger.info(f"Tool Call <{name}> Arguements: {arguements}")
        try:
            tool = self.tools[name]
        except KeyError:
            return f"ValueError: Tool '{name}' does not exist."
        try:
            result = tool(*arguements)
        except Exception as e:
            logger.warning(f"Tool Call <{name}> Failed: {e}")
            return f"RuntimeError: Error calling tool '{name}': {e}"
        else:
            return result

    def process(self, message: str, max_steps: int = 10) -> Dict | str:
        # Stateless agents are simple call response
        if self.state is None:
            results = [self.generate(message)]
        else:        
            self.state = AgentStates.RECIEVED_PROMPT
            steps = 0
            results = []

            while self.state != AgentStates.GAVE_ANSWER and steps < max_steps:
                response = self.generate(message)
                try:
                    state = response["state"]
                except KeyError:
                    # Assume this was the final answer
                    state = AgentStates.GAVE_ANSWER
                
                match state:
                    case AgentStates.RECIEVED_PROMPT:
                        self.state = AgentStates.RECIEVED_PROMPT
                    case AgentStates.TAKE_ACTION:
                        if self.tools and "tool_calls" in response:
                            response["tool_results"] = {}
                            tool_calls = response["tool_calls"]
                            for call in tool_calls:
                                try:
                                    name = call["name"]
                                except KeyError:
                                    response["tool_calls"]["unknown"] = "KeyError: No name given. Please provide 'name' field to call a tool."
                                try:
                                    arguements = call["arguements"]
                                except KeyError:
                                    arguements = []
                                response["tool_results"][name] = self.excute_tool_call(name,  arguements)
                            self.state = AgentStates.OBSERVE_ACTION_RESULT
                    case AgentStates.GAVE_ANSWER:
                        self.state = AgentStates.GAVE_ANSWER
                results.append(response)
        return results

class DeciderAgent(Agent):

    def __init__(
            self, 
            client: Client, 
            system_prompt: str = "",
            choices = Dict[str,Callable],
            options: ModelOptions | None = None,
            tools: Dict[Tool] | None = None,
        ):
        self.choices = choices
        system_prompt += "\nYou must choose ONE of the following options:\n"
        system_prompt += "\n".join(choices.keys())
        schema = {
            "decision": "|".join(choices.keys())
        }
        super().__init__(client, system_prompt=system_prompt, schema=schema, options=options, tools=tools)

    def decide(self, question: str, options: ModelOptions | None = None) -> Callable:
        response = self.generate(question, options=options)
        try:
            action = response["decision"]
            function = self.choices[action]
        except KeyError:
            logger.warning(f"Model failed to return a valid response: '{response}'. Schema:{self.schema}")
            return None
        else:
            return function