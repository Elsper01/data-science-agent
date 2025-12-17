from dataclasses import dataclass


@dataclass
class DurationMetadata:
    """Keeps track of the timing and duration of a node in the workflow."""
    method_name: str
    start: int
    stop: int

    def get_total_duration(self) -> int:
        """Returns the total duration in seconds."""
        return self.stop - self.start
