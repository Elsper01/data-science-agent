from pydantic import BaseModel, Field


class JudgeVerdictBase(BaseModel):
    """Base DTO for judge verdict responses (natural language neutral) regarding one figures and the appropriate code."""

    file_name: str = Field(...)
    figure_name: str = Field(...)
    critic_notes: str = Field(...)
    suggestion_code: str = Field(...)
    needs_regeneration: bool = Field(...)
    can_be_deleted: bool = Field(...)

    # allow population by field name or alias and validate on every assignment
    class Config:
        validate_by_name = True
        validate_assignment = True

