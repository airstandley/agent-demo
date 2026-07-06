from copy import deepcopy
import logging

logger = logging.getLogger(__name__)


class SimpleMemoryStore:

    def __init__(self):
        self.memory_bank = []
    
    def prompt_instructions(self) -> str:
        return "\n".join([
            "'context' to save a memory should be a simple string containing the information you wish to save for later.",
            "'context' to retrieve a memory should be blank, you get all your memories when requested."
        ])
    
    def save_memory(self, context: str):
        self.memory_bank.append(context)
    
    def retrieve_memories(self, context: str | None = None) -> list[str]:
        if context:
            logger.warning("SimpleMemoryStore does not support filtering memories by context")
        return deepcopy(self.memory_bank)