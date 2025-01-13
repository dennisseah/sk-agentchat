from typing import Callable
from unittest import mock

import pytest
from pytest_mock import MockerFixture

from sk_agentchat.services.azure_chat_completion_service import (
    AzureChatCompletionService,
    AzureChatCompletionServiceEnv,
)


@pytest.fixture
def mock_env_key() -> Callable[[bool], AzureChatCompletionServiceEnv]:
    def wrapper(with_key) -> AzureChatCompletionServiceEnv:
        if with_key:
            return AzureChatCompletionServiceEnv(
                azure_openai_endpoint="https://test.openai.azure.com/",
                azure_openai_api_key="test_key",
                azure_openai_api_version="2023-05-15",
                azure_openai_deployed_model_name="gpt-35-turbo",
            )
        return AzureChatCompletionServiceEnv(
            azure_openai_endpoint="https://test.openai.azure.com/",
            azure_openai_api_version="2023-05-15",
            azure_openai_deployed_model_name="gpt-35-turbo",
        )

    return wrapper


def test_azure_chat_completion_service(
    mock_env_key: Callable[[bool], AzureChatCompletionServiceEnv],
):
    service = AzureChatCompletionService(env=mock_env_key(True))
    assert service is not None


def test_create_chat_completion(
    mock_env_key: Callable[[bool], AzureChatCompletionServiceEnv], mocker: MockerFixture
):
    patch_chat_completion = mocker.patch(
        "sk_agentchat.services.azure_chat_completion_service.AzureChatCompletion"
    )

    service = AzureChatCompletionService(env=mock_env_key(True))
    chat_completion = service.create_chat_completion("test")
    assert chat_completion is not None
    patch_chat_completion.assert_called_once()
    assert patch_chat_completion.call_args.kwargs == {
        "service_id": "test",
        "api_key": "test_key",
        "deployment_name": "gpt-35-turbo",
        "endpoint": "https://test.openai.azure.com/",
        "api_version": "2023-05-15",
    }


def test_create_chat_completion_with_api_key(
    mock_env_key: Callable[[bool], AzureChatCompletionServiceEnv], mocker: MockerFixture
):
    patch_chat_completion = mocker.patch(
        "sk_agentchat.services.azure_chat_completion_service.AzureChatCompletion"
    )
    patch_default_cred = mocker.patch(
        "azure.identity.DefaultAzureCredential.get_token",
        return_value=mock.Mock(token="test_token"),
    )

    service = AzureChatCompletionService(env=mock_env_key(False))
    chat_completion = service.create_chat_completion("test")
    assert chat_completion is not None
    patch_chat_completion.assert_called_once()
    assert patch_chat_completion.call_args.kwargs == {
        "service_id": "test",
        "ad_token": "test_token",
        "deployment_name": "gpt-35-turbo",
        "endpoint": "https://test.openai.azure.com/",
        "api_version": "2023-05-15",
    }
    assert patch_default_cred.call_count == 1
