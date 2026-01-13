from pydantic import Field

from data_science_agent.dtos.base.responses.lida_evaluation_base import LidaEvaluationBase

from data_science_agent.dtos.de.responses.lida_evaluation_metrik import LidaEvaluationMetrik

class LidaEvaluation(LidaEvaluationBase):
    """DTO für die LIDA Bewertung einer Visualisierung."""

    bugs: LidaEvaluationMetrik = Field(..., description="Bewertung wie viele Bugs im Code der Visualisierung vorhanden sind.")
    transformation: LidaEvaluationMetrik = Field(..., description="Bewertung ob die Daten-Transformation korrekt und sinnvoll durchgeführt wurde.")
    compliance: LidaEvaluationMetrik = Field(..., description="Bewertung ob der Code das Visualisierungsziel erfüllt.")
    type: LidaEvaluationMetrik = Field(..., description="Bewertung ob der Visualisierungstyp passend für das Visualisierungsziel, die Daten und die Intention gewählt wurde.")
    encoding: LidaEvaluationMetrik = Field(..., description="Bewertung ob die Daten korrekt kodiert sind.")
    aesthetics: LidaEvaluationMetrik = Field(..., description="Bewertung ob die Visualisierung ästhetisch ansprechend gestaltet ist.")