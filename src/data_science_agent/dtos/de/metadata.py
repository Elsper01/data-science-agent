from pydantic import Field
from dtos.base.metadata_base import MetadataBase

class Metadata(MetadataBase):
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