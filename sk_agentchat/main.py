import asyncio
from typing import Any

from semantic_kernel.agents import AgentGroupChat
from semantic_kernel.agents.strategies.selection.kernel_function_selection_strategy import (  # noqa: E501
    KernelFunctionSelectionStrategy,
)
from semantic_kernel.agents.strategies.termination.kernel_function_termination_strategy import (  # noqa: E501
    KernelFunctionTerminationStrategy,
)
from semantic_kernel.contents.chat_message_content import ChatMessageContent
from semantic_kernel.contents.utils.author_role import AuthorRole

from sk_agentchat.tools.agents import (
    AGENT_NAME_CUSTOMER,
    create_customer_agent,
    create_investment_account_agent,
    create_kernel_with_chat_completion,
    create_saving_account_agent,
)
from sk_agentchat.tools.functions import (
    TERMINATION_KEYWORD,
    selection_function,
    termination_function,
)


async def main():
    agent_customer = create_customer_agent()
    agent_saving_account = create_saving_account_agent()
    agent_investment_account = create_investment_account_agent()

    def fn(results: Any) -> bool:
        return TERMINATION_KEYWORD in str(results.value[0]).lower()

    chat = AgentGroupChat(
        agents=[agent_customer, agent_saving_account, agent_investment_account],
        selection_strategy=KernelFunctionSelectionStrategy(
            function=selection_function,
            kernel=create_kernel_with_chat_completion("selection"),
            result_parser=lambda result: (
                str(result.value[0])
                if result.value is not None
                else AGENT_NAME_CUSTOMER
            ),
            agent_variable_name="agents",
            history_variable_name="history",
        ),
        termination_strategy=KernelFunctionTerminationStrategy(
            agents=[agent_saving_account],
            function=termination_function,
            kernel=create_kernel_with_chat_completion("termination"),
            result_parser=fn,
            history_variable_name="history",
            maximum_iterations=10,
        ),
    )

    user_input = "Get the investment and saving account balance."

    # this can be a while loop to get user input
    await chat.add_chat_message(
        ChatMessageContent(role=AuthorRole.USER, content=user_input)
    )
    async for response in chat.invoke():
        print(f"# {response.role} - {response.name or '*'}: '{response.content}'")


if __name__ == "__main__":
    asyncio.run(main())
