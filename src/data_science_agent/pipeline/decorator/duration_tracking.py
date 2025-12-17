import time
from functools import wraps

from data_science_agent.graph import AgentState
from data_science_agent.utils import DurationMetadata


def track_duration(func):
    """Decorator to measure execution time of graph node functions and store in AgentState."""
    @wraps(func)
    def wrapper(state: AgentState, *args, **kwargs):
        start_time = time.time()

        # Execute node
        result_state = func(state, *args, **kwargs)

        stop_time = time.time()
        duration = DurationMetadata(
            method_name=func.__name__,
            start=start_time,
            stop=stop_time,
        )

        result_state["durations"].append(duration)
        return result_state
    return wrapper