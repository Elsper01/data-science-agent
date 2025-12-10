import os

from dotenv import load_dotenv

from enums import Language


def __load_language() -> Language:
    """Loads the language setting from environment variables. Defaults to German if not set or unrecognized."""
    env = os.getenv("AGENT_LANGUAGE")
    if env:
        v = env.lower()
        if v.startswith("de"):
            return Language.DE
        elif v.startswith("en"):
            return Language.EN
    # Fallback: use german
    return Language.DE


load_dotenv()

MAX_REGENERATION_ATTEMPTS = int(os.getenv("MAX_REGENERATION_ATTEMPTS", "3"))
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
BASE_URL = "https://openrouter.ai/api/v1"
AGENT_LANGUAGE = __load_language()
