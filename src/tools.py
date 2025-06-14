from langchain_core.tools import tool

@tool
def process_billing_query(query: str) -> str:
    """Process a billing-related query from the customer.

    Args:
        query (str): The customer's billing-related question.

    Returns:
        str: Confirmation message.
    """
    return f"Billing query received. We'll review: '{query}'. Please allow 24–48 hours."

@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the current conversation to a human support agent.

    Args:
        reason (str): Why escalation is needed.

    Returns:
        str: Escalation confirmation.
    """
    return f"Your query has been escalated for human review. Reason: {reason}"

@tool
def handle_technical_issue(issue: str) -> str:
    """Handle technical support issues.

    Args:
        issue (str): Technical issue description.

    Returns:
        str: Logging confirmation.
    """
    return f"Technical issue logged: '{issue}'. Our support team will investigate and respond shortly."

tools = [process_billing_query, escalate_to_human, handle_technical_issue]
