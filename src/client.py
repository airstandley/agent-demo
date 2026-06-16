import ollama
from ._types import Message, ModelOptions

class LlamaOptions(ollama.Options):

    @classmethod
    def create_from_options(cls, options: ModelOptions) -> LlamaOptions:
        return cls(
            seed = options.seed,
            temperature= options.temperature,
            num_ctx=options.context_window,
            num_predict= options.max_tokens
        )

    @property
    def context_window(self) -> int | None:
        return self.num_ctx

    @property
    def max_tokens(self) -> int | None:
        return self.num_predict


class Client:

    def __init__(self, host: str = "http://localhost:11434", model: str = "phi3:mini", options: ModelOptions | None = None):
        self.llm: ollama.Client = ollama.Client(host=host)
        self.model: str = model
        self.model_options: LlamaOptions | None = LlamaOptions.create_from_options(options) if options is not None else None
    
    def generate(self, message: str, options: ModelOptions|None = None) -> str:
        if options is not None:
            options = LlamaOptions.create_from_options(options)
        response: ollama.GenerateResponse = self.llm.generate(
            model=self.model,
            options=options if options is not None else self.model_options,
            prompt=message,
            stream=False,  # Set to True if you want streaming response
        )
        return response.response
    
    def chat(self, messages: list[Message]) -> Message:
        response: ollama.ChatResponse = self.llm.chat(
            model=self.model, 
            options=self.model_options,
            messages=messages,
            stream=False
        )
        return response.message