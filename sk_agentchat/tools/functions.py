from semantic_kernel.functions.kernel_function_from_prompt import (
    KernelFunctionFromPrompt,
)

from sk_agentchat.tools.agents import (
    AGENT_NAME_CUSTOMER,
    AGENT_NAME_INVESTMENT_ACCOUNT,
    AGENT_NAME_SAVING_ACCOUNT,
)

TERMINATION_KEYWORD = "done"

selection_function = KernelFunctionFromPrompt(
    function_name="selection",
    prompt=f"""
    You are a helpful bank customer agent, collaborating with other product agents.
    Determine which agent takes the next turn in a conversation based on the the most recent conversation.
    State only the name of the participant to take the next turn.
    No participant should take more than one turn in a row.

    Choose only from these agents:
    - {AGENT_NAME_CUSTOMER}
    - {AGENT_NAME_INVESTMENT_ACCOUNT}
    - {AGENT_NAME_SAVING_ACCOUNT}

    Always follow these rules when selecting the next agent:
    - After user input, it is {AGENT_NAME_CUSTOMER}'s turn. 
    - After {AGENT_NAME_CUSTOMER} replies, {AGENT_NAME_INVESTMENT_ACCOUNT}'s turn.
    - After {AGENT_NAME_INVESTMENT_ACCOUNT} replies, {AGENT_NAME_SAVING_ACCOUNT}'s turn.

    History:
    {{{{$history}}}}
    """,  # noqa E501
)


termination_function = KernelFunctionFromPrompt(
    function_name="termination",
    prompt=f"""
        Examine the RESPONSE and determine whether the account balances are retunred in the conversation.
        If content is satisfactory, respond with a single word without explanation: {TERMINATION_KEYWORD}.
        If specific suggestions are being provided, it is not satisfactory.
        If no correction is suggested, it is satisfactory.

        RESPONSE:
        {{{{$history}}}}
        """,  # noqa E501
)
