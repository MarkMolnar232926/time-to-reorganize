"""reorg: time-to-reorganise metrics for defensive transitions from SkillCorner tracking data.

Public API (see ``reorg.api`` for an end-to-end example)::

    find_losses, build_reference, time_to_reorganise, reliability_report,
    plot_reorganisation, run_league, team_summary, load_config
"""

from reorg.api import load_config, plot_reorganisation, run_league
from reorg.reference import build_reference
from reorg.reliability import reliability_report
from reorg.reorganisation import team_summary, time_to_reorganise
from reorg.turnovers import find_losses

__version__ = "0.1.0"

__all__ = [
    "build_reference",
    "find_losses",
    "load_config",
    "plot_reorganisation",
    "reliability_report",
    "run_league",
    "team_summary",
    "time_to_reorganise",
]
