from typing import Protocol


class Message(Protocol):
    role: str
    content: str | None
    tool: str | None

class ModelOptions(Protocol):
    seed: int | None = None
    temperature: float | None = None
    context_window: int | None= None
    max_tokens: int | None = None
    
class Client(Protocol):

    def generate(prompt: str, options: ModelOptions|None = None) -> str:
        ...
    
    def chat(message=list[Message]) -> Message:
        ...


