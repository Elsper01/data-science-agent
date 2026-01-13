from pydantic import Field

from data_science_agent.dtos.base.responses.lida_evaluation_base import LidaEvaluationBase
from data_science_agent.dtos.en.responses.lida_evaluation_metrik import LidaEvaluationMetrik


class LidaEvaluation(LidaEvaluationBase):
    """DTO for the LIDA evaluation of a visualization."""

    bugs: LidaEvaluationMetrik = Field(..., description="Rating of how many bugs are present in the visualization code.")
    transformation: LidaEvaluationMetrik = Field(..., description="Rating of whether the data transformation was performed correctly and meaningfully.")
    compliance: LidaEvaluationMetrik = Field(..., description="Rating of whether the code fulfills the visualization goal.")
    type: LidaEvaluationMetrik = Field(..., description="Rating of whether the visualization type was chosen appropriately for the visualization goal, data, and intent.")
    encoding: LidaEvaluationMetrik = Field(..., description="Rating of whether the data is encoded correctly.")
    aesthetics: LidaEvaluationMetrik = Field(..., description="Rating of whether the visualization is aesthetically designed.")