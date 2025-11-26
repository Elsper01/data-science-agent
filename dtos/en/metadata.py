from pydantic import Field
from dtos.base.metadata_base import MetadataBase

class Metadata(MetadataBase):
    """
    Represents a triple from the RDF graph.
    """

    subject: str = Field(
        ...,
        description="Subject of the triple"
    )

    predicate: str = Field(
        ..., description="Predicate of the triple"
    )
    object: str = Field(
        ..., description="Object of the triple"
    )
