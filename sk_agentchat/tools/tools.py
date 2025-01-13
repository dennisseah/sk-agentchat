import logging
import random

from semantic_kernel.functions import kernel_function

from sk_agentchat.hosting import container

logger = container[logging.Logger]


class AccountIdPlugin:
    @kernel_function(
        name="get_account_id", description="Get the account id for all accounts"
    )
    def get_account_id(self) -> str:
        """
        Get the account id.
        """
        id = random.choice(["123", "456", "789"])
        logger.info(f"[TOOL] Bank account ID: {id}")
        return id


class AccountSavingPlugin:
    @kernel_function(
        name="get_saving_balance", description="Get the balance of the saving account"
    )
    def get_saving_balance(self, account_id: str) -> float:
        """
        Get the balance of the saving account.
        """
        balance = {
            "123": 10000.10,
            "456": 20000.20,
            "789": 30000.30,
        }[account_id]
        logger.info(f"[TOOL] Saving balance: {balance}")
        return balance


class AccountInvestmentPlugin:
    @kernel_function(
        name="get_investment_balance",
        description="Get the balance of the investment account",
    )
    def get_investment_balance(self, account_id: str) -> float:
        """
        Get the balance of the investment account.
        """
        balance = {
            "123": 100000.10,
            "456": 200000.20,
            "789": 300000.30,
        }[account_id]
        logger.info(f"[TOOL] Investment balance: {balance}")
        return balance
