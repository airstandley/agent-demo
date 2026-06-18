# agent.py
from enum import StrEnum, EnumMeta
import json
import logging
from typing import List, Dict, Callable
from dataclasses import dataclass, field

from ._types import Client, ModelOptions, Tool

logger = logging.getLogger(__name__)



class AgentActionMeta(EnumMeta):
    def __new__(cls, name, bases, dct):
        # Add values list to class
        klass = super().__new__(cls, name, bases, dct)
        klass.values = [str(value) for value in klass.__members__.values()]
        return klass

class AgentActions(StrEnum, metaclass=AgentActionMeta):
    REQUEST_PROMPT = "read_user_prompt"
    TAKE_ACTION = "take_action"
    GIVE_ANSWER = "give_answer"
    THROW_ERROR = "throw_error"

    @classmethod
    def valid_choices(cls) ->  list[AgentActions]:
        return [
            cls.TAKE_ACTION,
            cls.GIVE_ANSWER
        ]

@dataclass
class TaskStep:
    step: int = 0
    action: AgentActions = AgentActions.REQUEST_PROMPT
    result: str | None = None

    def __repr__(self):
        description = ""
        match self.action:
            case AgentActions.REQUEST_PROMPT:
                description = f"I recieved the user prompt and decided to {self.result["action"]} next."
                if self.result["action"] == AgentActions.REQUEST_PROMPT:
                    description += " I got confused. I've already read the user prompt. I should not do this again."
            case AgentActions.TAKE_ACTION:
                description = f"I used my available tools."
                if not self.result["tool_calls"]:
                    description += "But I forgot to specify a tool so got no results."
                else:
                    description += "I got the following results:"
                    for tool in self.result["tool_calls"]:
                        name, args = tool["name"], tool["arguements"]
                        result = self.result["tool_results"][name]
                        description += f"      - {name}({args}):{result}"
        return f"{self.step}. ({self.action}): {description}"

@dataclass
class AgentState:
    steps: list[TaskStep] = field(default_factory=list)
    step_count: int = 0
    current_action: AgentActions = AgentActions.REQUEST_PROMPT
    done: bool = False

    def reset(self):
        self.steps = list()
        self.step_count = 0
        self.current_action = AgentActions.REQUEST_PROMPT
        self.done = False


class Agent:
### Notes:
# We're starting to hit highly unreliable behaviour were multiple pointless steps are taken.
# We'll see if the later lessons help address this, but my current thoughts are that adding more structure could improve the
# probability of generating useful text outputs. There might also be some tricks to formatting the prompt that would preform better for a given model.
# It's pretty apparent that this black box apporach is 

    def __init__(
            self, 
            client: Client, 
            system_prompt: str | None = None,
            options: ModelOptions | None = None,
            schema: Dict | None = None,
            tools: Dict[Tool] | None = None,
            stateful: bool = True,
        ):
        self.system_prompt = system_prompt + "\n"
        self.model_options = options
        self.schema = schema or {}
        self.retries = 3  # Number to times to attempt a valid generation per step
        self.client = client
        self.tools = tools
        if self.tools:
            self.system_prompt += "As an agent, tools are available to you (below in <TOOLS>). You use tools when you need to access external resources like a file-system or the internet. To call a tool respond using the 'tool_calls' property.\n"
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
            self.state = AgentState()
            self.system_prompt = "\n".join([
                self.system_prompt,
                "As an agent you MUST chose an action. The following options are available:",
                "\n".join([f"- {value}" for value in AgentActions.values if value in AgentActions.valid_choices()])       
            ])
            self.schema["action"] = f"enum({"|".join(AgentActions.values)}"

    def _generate(self, message: str, options: ModelOptions | None = None) -> str:
        prompt = []
        # Add System Prompt for Agents
        if self.system_prompt:
            prompt += [
                "<SYSTEM>",
                self.system_prompt,
                "</SYSTEM>"
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
            "<USER PROMPT>",
            message,
            "</USER PROMPT>"
        ]
        # Add State Information if Stateful
        if self.state is not None:
            prompt += [
                "<STATE>",
                "\n".join([str(step) for step in self.state.steps]),
                "</STATE>"
            ]
        # Generate the response
        prompt = "\n".join(prompt)
        prompt += "\n<AGENT RESPONSE>"
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
            self.state.reset()
            results = []

            while not self.state.done and self.state.step_count < max_steps:
                self.state.step_count += 1
                step = TaskStep(step=self.state.step_count, action=self.state.current_action)
                response = self.generate(message)
                try:
                    action = response["action"]
                except KeyError:
                    # Assume this was the final answer
                    action = AgentActions.GIVE_ANSWER
                
                match action:
                    case AgentActions.REQUEST_PROMPT:
                        self.state.current_action = AgentActions.REQUEST_PROMPT
                    case AgentActions.TAKE_ACTION:
                        self.state.current_action = AgentActions.TAKE_ACTION
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
                    case AgentActions.GIVE_ANSWER:
                        self.state.current_action = AgentActions.GIVE_ANSWER
                        self.state.done = True
                    case AgentActions.THROW_ERROR:
                        logger.warning(f"Agent ran into an error with task: {message}")
                        self.state.done = True
                step.result = response
                self.state.steps.append(step)
                results.append(response)
        return results
