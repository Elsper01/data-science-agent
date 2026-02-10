from enum import StrEnum

class LLMModel(StrEnum):
    """Contains all supported and tested LLM models"""
    GPT_4o = "gpt-4o"
    GPT_5 = "gpt-5"
    GROK = "x-ai/grok-code-fast-1"
    CLAUDE_4 = "anthropic/claude-sonnet-4.5"
    MINIMAX = "minimax/minimax-m2.1"
    MISTRAL = "mistralai/mistral-large-2512"
    DEVSTRAL = "mistralai/devstral-2512:free"