from enum import StrEnum

class LLMModel(StrEnum):
    """Contains all supported and tested LLM models"""
    GPT_4o = "gpt-4o"
    GPT_5 = "gpt-5"
    GROK = "x-ai/grok-code-fast-1"