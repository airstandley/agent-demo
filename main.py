from src import Agent
from src import Client
from src.tooling import tool, tool_registry, Tool
from src.memory import SimpleMemoryStore

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
        temperature=0.5
    )
    client = Client(model="qwen2.5:3b", options=options)
    agent = Agent(
        client = client, 
        system_prompt="""You are an executive office assistant. Give accurate concise answers. No elaboration unless asked. Do not make up information.""",
        schema={
            "response": {"type": "string", "required": "True"},
            "certainty": {"type": "enum('low' | 'med' | 'high')", "required": "True"},
            "internal_thoughts": {"type": "string", "required": "False"}
            },
        tools=tool_registry,  # Example of quickly creating an agent using all defined tools           
        memory_store=SimpleMemoryStore()   
    )

    print("\n\n")
    print("Introduction...")
    print(agent.process("Hi there, my name is Alice. What should I call you?")[-1]["response"])
    print("\n\n")

    print("\n\n")
    print("Recall...")
    print(agent.process("What is my name?")[-1]["response"])
    print("\n\n")
  
    # print("\n\n")
    # print("Asking about agents...")
    # print(agent.process("Explain what an AI agent is?")[-1]["response"])
    # print("\n\n")
    
    # print("\n\n")
    # print("Asking for news...")
    # print(agent.process("What is the latest AI agent news?")[-1]["response"])
    # print("\n\n")
    
    # print("\n\n")
    # print("Asking to print a document...")
    # print(agent.process("Print README.md")[-1]["response"])
    # print("\n\n")

    # print("\n\n")
    # print("Asking to read emails...")
    # print(agent.process("What emails have I recieved today?")[-1]["response"])
    # print("\n\n")

    # print("\n\n")
    # print("Asking the complex task...")
    # print(agent.process("Print the latest product spec, it should have been emailed to me.")[-1]["response"])
    # print("\n\n")



if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()