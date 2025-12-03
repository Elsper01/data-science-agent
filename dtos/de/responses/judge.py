from pydantic import Field
from dtos.base.responses.judge_base import JudgeBase
from dtos.de.responses.judge_verdict import JudgeVerdict

class Judge(JudgeBase):
    """Urteil des Judge-LLMs bezüglich eines ganzen Skripts das alle Grafiken und Figuren erzeugt."""

    verdicts: list[JudgeVerdict] = Field(...)
    needs_regeneration: bool = Field(...)

    # allow population by field name or alias and validate on every assignment
    class Config:
        validate_by_name = True
        validate_assignment = True
