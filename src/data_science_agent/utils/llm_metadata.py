from dataclasses import dataclass
from typing import Optional

from langchain_core.messages import AIMessage
from pydantic import BaseModel, Field

from data_science_agent.utils import print_color
from data_science_agent.utils.enums import Color


class TokenUsage(BaseModel):
    """Token usage details returned by the LLM API."""
    completion_tokens: Optional[int] = Field(default=None)
    prompt_tokens: Optional[int] = Field(default=None)
    total_tokens: Optional[int] = Field(default=None)


class CostDetails(BaseModel):
    """Cost breakdown of the LLM request."""
    total_cost: Optional[float] = Field(default=None)
    cost_upstream_inference: Optional[float] = Field(default=None)
    cost_upstream_inference_prompt: Optional[float] = Field(default=None)
    cost_upstream_inference_completions: Optional[float] = Field(default=None)


@dataclass
class LLMMetadata:
    """Class to hold metadata about LLM communication."""
    method_name: str
    message: AIMessage
    token_usage: TokenUsage
    cost_details: CostDetails
    model_name: str

    @classmethod
    def from_ai_message(cls, ai_message: AIMessage, calling_method_name) -> "LLMMetadata":
        """Create an LLMMetadata instance from an AIMessage containing response metadata."""
        print(ai_message)
        metadata = getattr(ai_message, "response_metadata", {}) or {}
        token_usage_data = metadata.get("token_usage", {}) or {}
        cost_details_data = token_usage_data.get("cost_details", {}) or {}
        model_name = metadata.get("model_name", "Unknown")

        token_usage = TokenUsage(
            completion_tokens=token_usage_data.get("completion_tokens"),
            prompt_tokens=token_usage_data.get("prompt_tokens"),
            total_tokens=token_usage_data.get("total_tokens")
        )

        cost_details = CostDetails(
            total_cost=token_usage_data.get("cost"),
            cost_upstream_inference=cost_details_data.get("upstream_inference_cost"),
            cost_upstream_inference_prompt=cost_details_data.get("upstream_inference_prompt_cost"),
            cost_upstream_inference_completions=cost_details_data.get("upstream_inference_completion_cost")
        )

        return cls(message=ai_message, token_usage=token_usage, cost_details=cost_details,
                   method_name=calling_method_name, model_name=model_name)

    def print_costs(self):
        """Print the token and cost summary."""
        print_color("Token & Cost Summary", Color.HEADER)
        print_color(f"  - Input Tokens: {self.token_usage.completion_tokens}", Color.OK_BLUE)
        print_color(f"  - Output Tokens: {self.token_usage.prompt_tokens}", Color.OK_BLUE)
        print_color(f"  - Total Tokens: {self.token_usage.total_tokens}", Color.OK_BLUE)
        print_color(f"  - Total Cost: ${self.cost_details.total_cost:.6f}", Color.OK_BLUE)
        print_color(f"    - Upstream Inference Cost: ${self.cost_details.cost_upstream_inference}", Color.OK_BLUE)
        print_color(f"      - Upstream Inference Prompt Cost: ${self.cost_details.cost_upstream_inference_prompt}",
                    Color.OK_BLUE)
        print_color(
            f"      - Upstream Inference Completion Cost: ${self.cost_details.cost_upstream_inference_completions}",
            Color.OK_BLUE)
