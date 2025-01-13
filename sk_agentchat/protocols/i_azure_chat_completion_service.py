from typing import Protocol

from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import (
    AzureChatCompletion,
)


class IAzureChatCompletionService(Protocol):
    def create_chat_completion(self, service_id: str) -> AzureChatCompletion: ...
