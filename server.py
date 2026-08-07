import os
import sys
from typing import Any

from mcp.server import Server

mcp = Server("rag-server")


@mcp.tool()
def get_context(question: str) -> str:
    """Return a small amount of contextual information for a question."""
    question_lower = question.lower()

    if "leave" in question_lower or "vacation" in question_lower:
        return "Leave requests should be submitted at least 3 working days in advance. Supervisors must approve them before the leave starts."

    if "overtime" in question_lower or "hours" in question_lower:
        return "Overtime must be pre-approved by the team lead and is compensated according to company policy."

    if "support" in question_lower or "contact" in question_lower:
        return "For technical support, contact the IT help desk or your department manager."

    return "No special MCP context was found for this question."


if __name__ == "__main__":
    mcp.run()
