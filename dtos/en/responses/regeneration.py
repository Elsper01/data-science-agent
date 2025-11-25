# python
from pydantic import BaseModel, Field

class Regeneration(BaseModel):
    """Indicates whether errors actually occurred during code execution and the code needs to be regenerated."""
    should_be_regenerated: bool = Field(..., description="Indicates whether the code must be regenerated.")
