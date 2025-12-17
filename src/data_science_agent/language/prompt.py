from data_science_agent.utils.enums import Language


class Prompt:
    """A class representing a prompt for a data science agent."""

    def __init__(self, de: dict, en: dict):
        """Initialize the prompt with german and english versions."""
        self.de = de
        self.en = en

    def get_prompt(self, language: Language, key: str, **fmt):
        if language == Language.DE:
            prompt = self.de.get(key, "")
            if prompt != "":
                return prompt.format(**fmt)
        elif language == Language.EN:
            prompt = self.en.get(key, "")
            if prompt != "":
                return prompt.format(**fmt)
        else:
            raise ValueError(f"Unsupported language: {language}")
        raise ValueError(f"Prompt for key '{key}' not found in language '{language}'.")