from pydantic import Field

from dtos.base.responses.regeneration_base import RegenerationBase


class Regeneration(RegenerationBase):
    """Indicates whether errors actually occurred during code execution and the code needs to be regenerated."""
    should_be_regenerated: bool = Field(..., description="Indicates whether the code must be regenerated.")
