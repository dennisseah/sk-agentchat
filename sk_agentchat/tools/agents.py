from semantic_kernel.agents import ChatCompletionAgent
from semantic_kernel.connectors.ai.function_choice_behavior import (
    FunctionChoiceBehavior,
)
from semantic_kernel.connectors.ai.open_ai.prompt_execution_settings.azure_chat_prompt_execution_settings import (  # noqa: E501
    AzureChatPromptExecutionSettings,
)
from semantic_kernel.kernel import Kernel

from sk_agentchat.hosting import container
from sk_agentchat.protocols.i_azure_chat_completion_service import (
    IAzureChatCompletionService,
)
from sk_agentchat.tools.tools import (
    AccountIdPlugin,
    AccountInvestmentPlugin,
    AccountSavingPlugin,
)

execution_settings = AzureChatPromptExecutionSettings()

execution_settings.function_choice_behavior = FunctionChoiceBehavior.Auto()
execution_settings.temperature = 0

AGENT_NAME_SAVING_ACCOUNT = "saving_account_agent"
AGENT_NAME_INVESTMENT_ACCOUNT = "investment_account_agent"
AGENT_NAME_CUSTOMER = "customer_agent"
azure_chat_completion_service = container[IAzureChatCompletionService]


def create_kernel_with_chat_completion(
    service_id: str,
) -> Kernel:
    kernel = Kernel()
    kernel.add_service(
        azure_chat_completion_service.create_chat_completion(service_id=service_id),
    )
    return kernel


def create_saving_account_agent() -> ChatCompletionAgent:
    kernel = create_kernel_with_chat_completion(AGENT_NAME_SAVING_ACCOUNT)
    kernel.add_plugin(plugin_name="AccountSavingPlugin", plugin=AccountSavingPlugin())

    agent_saving_account = ChatCompletionAgent(
        service_id=AGENT_NAME_SAVING_ACCOUNT,
        kernel=kernel,
        name=AGENT_NAME_SAVING_ACCOUNT,
        execution_settings=execution_settings,
        instructions="You are the bank saving account agent who can provide the saving "
        "account balance.",
    )
    return agent_saving_account


def create_investment_account_agent() -> ChatCompletionAgent:
    kernel = create_kernel_with_chat_completion(AGENT_NAME_INVESTMENT_ACCOUNT)
    kernel.add_plugin(
        plugin_name="AccountInvestmentPlugin", plugin=AccountInvestmentPlugin()
    )

    agent_investment_account = ChatCompletionAgent(
        service_id=AGENT_NAME_INVESTMENT_ACCOUNT,
        kernel=kernel,
        name=AGENT_NAME_INVESTMENT_ACCOUNT,
        execution_settings=execution_settings,
        instructions="You are the bank investment account agent who can provide the "
        "investment account balance.",
    )
    return agent_investment_account


def create_customer_agent() -> ChatCompletionAgent:
    kernel = create_kernel_with_chat_completion(AGENT_NAME_CUSTOMER)
    kernel.add_plugin(plugin_name="AccountIdPlugin", plugin=AccountIdPlugin())

    agent_customer = ChatCompletionAgent(
        service_id=AGENT_NAME_CUSTOMER,
        kernel=kernel,
        name=AGENT_NAME_CUSTOMER,
        execution_settings=execution_settings,
        instructions="You are the bank customer agent who can provide the account id.",
    )
    return agent_customer
