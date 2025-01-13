from dataclasses import dataclass

from azure.identity import DefaultAzureCredential
from lagom.environment import Env
from semantic_kernel.connectors.ai.open_ai.services.azure_chat_completion import (
    AzureChatCompletion,
)

from sk_agentchat.protocols.i_azure_chat_completion_service import (
    IAzureChatCompletionService,
)


class AzureChatCompletionServiceEnv(Env):
    azure_openai_endpoint: str
    azure_openai_api_key: str | None = None
    azure_openai_api_version: str
    azure_openai_deployed_model_name: str


@dataclass
class AzureChatCompletionService(IAzureChatCompletionService):
    env: AzureChatCompletionServiceEnv

    def create_chat_completion(self, service_id: str) -> AzureChatCompletion:
        if self.env.azure_openai_api_key:
            return AzureChatCompletion(
                service_id=service_id,
                api_key=self.env.azure_openai_api_key,
                deployment_name=self.env.azure_openai_deployed_model_name,
                endpoint=self.env.azure_openai_endpoint,
                api_version=self.env.azure_openai_api_version,
            )

        azure_ad_token = (
            DefaultAzureCredential()
            .get_token("https://cognitiveservices.azure.com/.default")
            .token
        )
        return AzureChatCompletion(
            service_id=service_id,
            deployment_name=self.env.azure_openai_deployed_model_name,
            ad_token=azure_ad_token,
            endpoint=self.env.azure_openai_endpoint,
            api_version=self.env.azure_openai_api_version,
        )
