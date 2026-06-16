# agent.py
import json
import logging
from typing import List, Dict, Callable

from ._types import Client, ModelOptions, Tool

logger = logging.getLogger(__name__)

class Agent:

    def __init__(
            self, 
            client: Client, 
            system_prompt: str | None = None,
            options: ModelOptions | None = None,
            schema: Dict | None = None,
            tools: Dict[Tool] | None = None
        ):
        self.system_prompt = system_prompt
        self.model_options = options
        self.schema = schema
        self.retries = 3  # Number to times to attempt a valid generation per step
        self.client = client
        self.tools = tools
        if self.tools:
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
    
    def process(self, message: str) -> Dict | str:
        response = self.generate(message)
        if isinstance(response, str):
            return response
        if self.tools and "tool_calls" in response:
            response["tool_results"] = {}
            tool_calls = response["tool_calls"]
            logger.info(f"tool_calls: {tool_calls}")
            for call in tool_calls:
                try:
                    name = call["name"]
                    tool = self.tools[name]
                    result = tool(*call["arguements"])
                except Exception as e:
                    logger.warning(f"Tool Call Failed: {e}")
                else:
                    response["tool_results"][name] = result
        return response


class DeciderAgent(Agent):

    def __init__(
            self, 
            client: Client, 
            system_prompt: str = "",
            choices = Dict[str,Callable],
            options: ModelOptions | None = None,
        ):
        self.choices = choices
        system_prompt += "\nYou must choose ONE of the following options:\n"
        system_prompt += "\n".join(choices.keys())
        schema = {
            "decision": "|".join(choices.keys())
        }
        super().__init__(client, system_prompt=system_prompt, schema=schema, options=options)

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