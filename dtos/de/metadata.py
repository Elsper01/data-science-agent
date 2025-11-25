from pydantic import BaseModel, Field

class Metadata(BaseModel):
    """
    Repräsentiert ein Triple aus dem RDF Graphen.
    """

    subject: str = Field(
        ...,
        description="Subjekt des Tripels"
    )

    predicate: str = Field(
        ..., description="Prädikat des Tripels"
    )
    object: str = Field(
        ..., description="Objekt des Tripels"
    )