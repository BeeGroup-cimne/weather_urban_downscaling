"""
Legacy utilities kept for backward compatibility.
All functions are re-exported from src.utils.training.
"""

from src.utils.training import (
    notify_completion,
    plot_comparative_history,
    run_experiment,
    visualize_results,
)

__all__ = [
    "notify_completion",
    "plot_comparative_history",
    "run_experiment",
    "visualize_results",
]
