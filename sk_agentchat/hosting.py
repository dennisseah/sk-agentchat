"""Defines our top level DI container.
Utilizes the Lagom library for dependency injection, see more at:

- https://lagom-di.readthedocs.io/en/latest/
- https://github.com/meadsteve/lagom
"""

import logging
import os

from dotenv import load_dotenv
from lagom import Container, dependency_definition

from sk_agentchat.protocols.i_azure_chat_completion_service import (
    IAzureChatCompletionService,
)

load_dotenv(dotenv_path=".env")


container = Container()
"""The top level DI container for our application."""


# Register our dependencies ------------------------------------------------------------


@dependency_definition(container, singleton=True)
def logger() -> logging.Logger:
    logging.basicConfig(level=os.getenv("LOG_LEVEL", "DEBUG"))
    logging.Formatter(fmt=" %(name)s :: %(levelname)-8s :: %(message)s")
    return logging.getLogger("sk_agentchat")


@dependency_definition(container, singleton=True)
def azure_openai_service() -> IAzureChatCompletionService:
    from sk_agentchat.services.azure_chat_completion_service import (
        AzureChatCompletionService,
    )

    return container[AzureChatCompletionService]
