"""Policy-preserving RF adapters used by the UAV MCP service.

These functions make Sionna observable to an agent without exposing scene
files, arbitrary frequencies, shell execution, or any flight-control field.
"""

from __future__ import annotations

from typing import Any

from ..skills.uav_spectrum_sim.runtime import get_runtime_status
from ..skills.uav_spectrum_sim.sionna_measurement import (
    collect_current_sionna_observation,
    get_sionna_spectrum_situation,
)


def get_spectrum_situation() -> dict[str, Any]:
    """Read the latest validated Sionna RT observation without recomputation."""
    return get_sionna_spectrum_situation(runtime_status=get_runtime_status())


def measure_spectrum_at_current_pose() -> dict[str, Any]:
    """Measure only the current simulator pose; never emit a flight command."""
    observation = collect_current_sionna_observation(runtime_status=get_runtime_status())
    return {
        "available": True,
        "current": True,
        "profile_id": observation["profile_id"],
        "captured_at": observation["timestamp"],
        "age_s": 0.0,
        "current_position_m": observation["position_m"],
        "pose_offset_m": 0.0,
        "observation": observation,
    }
