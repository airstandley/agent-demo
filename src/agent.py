# agent.py
import json
import logging
from typing import List, Dict

from ._types import Client

logger = logging.getLogger(__name__)

class Agent:

    def __init__(self, client: Client, system_prompt: str | None = None, schema: Dict | None = None, tools: List[Dict] = None):
        self.system_prompt = system_prompt
        self.schema = schema
        self.retries = 3  # Number to times to attempt a valid generation per step
        self.client = client

    def _generate(self, message: str) -> str:
        prompt = []
        # Add System Prompt for Agents
        if self.system_prompt:
            prompt += [
                "<SYSTEM>",
                self.system_prompt,
                "</SYSTEM>"
            ]
        # Add Output Format Restrictions if Defined
        if self.schema:
            prompt += [
                "<FORMAT RULES>",
                "1. Responses must follow the following schema:",
                f"{self.schema}",
                "2. No explanations, no markdown, no extra text before or after the JSON",
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
        return self.client.generate(prompt)

    def _parse_response_for_json(self, response: str) -> Dict:
        # Attempt to wrangle the garbage LLMs can generate into a usable JSON format.
        # Thankfully since these things are statistically predictors the kinds of mistakes they make are somewhat predictable.

        # 1. Extract only the text between the opening and closed brackets
        start = response.find('{')
        end = response.rfind('}')
        json_text = response[start:end+1]
        logger.debug(f"JSON: {json_text}")

        return json.loads(json_text)

    def generate(self, message: str) -> Dict:
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
            raise RuntimeError("The AI can't generate formatted output worth shit")
        else:
            return self._generate(message)

    def agent_step(user_message: str) -> str:
        messages = [
            {"role": "system", "content": "You are a chatbot. Try to truthfully answer user questions. Use tools when needed."},
            {"role": "user", "content": user_message}
        ]

        for _ in range(self.max_step_atempts):
            response = self.client.generate(messages[1]["content"])

            #parsed = extract_json_from_text(response)
        
            # if parsed and "action" in parsed:
            #     if "reason" not in parsed:
            #         parsed["reason"] = f"Taking action: {parsed['action']}"
            #     self.state.increment_step()
            #     return parsed
            return response

    def run_loop(user_message: str, max_steps: int = 8) -> str:
        
        for _ in range(max_steps):
            # Step 1: Send message to LLM
            response = llm.generate(messages)  # Use Ollama via `OllamaLLM`
            message = response["choices"][0]["message"]
            # Append to conversation history
            messages.append(message)
            # Step 2: Check if LLM wants to call a tool
            if not message.tool_calls:
                return message.content  # Final answer, no more tools
            # Step 3: Execute each tool call
            for tool_call in message.tool_calls:
                name = tool_call.name
                args = json.loads(tool_call.function.arguments)
                # Execute the function (from tools.py)
                result = call_tool(name, args)
                # Append result to conversation
                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call.id,
                    "content": result
                })
        return "Step budget exceeded without a final answer."