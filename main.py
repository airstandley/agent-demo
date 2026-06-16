from src import Agent
from src import Client

from dataclasses import dataclass

@dataclass
class Options:
    seed: int | None = None
    temperature: float | None = None
    context_window: int | None= None
    max_tokens: int | None = None


def main() -> None:
    options = Options(
    )
    client = Client(options=options)
    agent = Agent(
        client = client, 
        system_prompt="You give accurate answers in 1-2 sentences maximum. No elaboration unless asked.",
        schema={
            "topic": "string",
            "difficulty": "enum('beginner' | 'intermediate' | 'advanced')"
            }              
    )

    print(agent.generate("Explain what an AI agent is?"))



if __name__ == "__main__":
    main()