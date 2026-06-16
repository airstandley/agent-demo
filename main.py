from src import Agent
from src import Client
from src.agent import DeciderAgent
from src.tooling import tool, tool_registry, Tool

from dataclasses import dataclass
import logging

@dataclass
class Options:
    seed: int | None = None
    temperature: float | None = None
    context_window: int | None= None
    max_tokens: int | None = None

@tool
def search_web(query: str):
    """Search the web for information"""
    return "The internet is not working right now. No search available."

@tool
def print_document(filename: str):
    """Print the given file on our connected HP printer"""
    return f"{filename} sucessfully printed"

@tool
def get_email(max: int):
    """Get a list of the latest emails (returns up to max emails requested)"""
    return """[
{"date_recieved": "2026-02-10 10:45", "subject": "Latest Product Spec Sheet", "attachments": ["product_spec.pdf"]}
]"""

def main() -> None:
    options = Options(
    )
    client = Client(options=options)
    agent = Agent(
        client = client, 
        system_prompt="""You are a tool calling assistant. Use the tools available to you. To call a tool respond using the 'tool_calls' property.
Give accurate answers in 1-2 sentences maximum. No elaboration unless asked.""",
        schema={
            "response": {"type": "string", "required": "True"},
            "certainty": {"type": "enum('low' | 'med' | 'high')", "required": "True"},
            "internal_thoughts": {"type": "string", "required": "False"}
            },
        tools=tool_registry  # Example of quickly creating an agent using all defined tools              
    )

    decider = DeciderAgent(
        client=client,
        system_prompt="You are a helpful office assistant.",
        choices={
            "summarize": lambda: print("summarizing"),
            "search_web": lambda: print("searching web"),
            "translate": lambda: print("translating"),
            "eat_cake": lambda: print("eating cake"),
            "dance": lambda: print("dancing!!"),
            "none_of_the_above": lambda: print("NONE")
        },
        tools={
            "search_web": Tool(search_web)  # Example of creating an agent with limited tool access.
        }
        options=Options(temperature=0.3)
    )

    # print(agent.process("Explain what an AI agent is?"))
    print(agent.process("What is the latest AI agent news?"))
    print(agent.process("Print README.md"))
    print(agent.process("What emails have I recieved today?"))
    # decider.decide("It's raining today.")()



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()