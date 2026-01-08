from pydantic import Field
from typing import Optional

from data_science_agent.dtos.base.responses.judge_verdict_base import JudgeVerdictBase


class JudgeVerdict(JudgeVerdictBase):
    """Represents the verdict of a Judge LLM regarding a single figure and its corresponding code that generates the figure."""

    file_name: str = Field(..., description="The name of the file under which the figure was saved.")
    figure_name: str = Field(..., description="The name of the figure to which this verdict refers.")
    critic_notes: str = Field(..., description="Critique points and notes from the Judge LLM regarding the figure and code.")
    suggestion_code: Optional[str] = Field(..., description="Suggested code from the Judge LLM to improve or correct the figure.")
    needs_regeneration: bool = Field(..., description="Indicates whether the figure should be regenerated based on the Judge LLM's verdict.")
    can_be_deleted: bool = Field(..., description="Indicates whether the figure and its corresponding code can be deleted as they do not meet requirements or provide no value.")