from pydantic import Field
from data_science_agent.dtos.base.responses.judge_base import JudgeBase
from data_science_agent.dtos.de.responses.judge_verdict import JudgeVerdict

class Judge(JudgeBase):
    """Verdict of the Judge LLM regarding an entire script that generates all graphics and figures."""

    verdicts: list[JudgeVerdict] = Field(..., description="List of verdicts from the Judge LLM regarding each individual figure and its corresponding code.")
    needs_regeneration: bool = Field(..., description="Indicates whether anything in the script should be modified.")