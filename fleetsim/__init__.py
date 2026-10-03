"""fleetsim: a discrete-event robotaxi fleet simulator with pluggable dispatch policies."""

from .config import SimConfig
from .simulator import Simulation
from .policies import get_policy, POLICIES
from .metrics import summarize

__all__ = ["SimConfig", "Simulation", "get_policy", "POLICIES", "summarize"]
__version__ = "0.1.0"
