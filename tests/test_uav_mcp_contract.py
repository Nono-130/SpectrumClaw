"""MCP-facing UAV contract tests that do not require a live PX4 process."""

from __future__ import annotations


def test_mcp_contract_exposes_only_plan_level_flight_and_fixed_profile_rf_tools():
    from backend.mcp.uav_contract import MCP_UAV_TOOL_NAMES, capability_document

    assert MCP_UAV_TOOL_NAMES == (
        "uav_get_state",
        "uav_list_templates",
        "uav_validate_plan",
        "uav_execute_plan",
        "uav_cancel_mission",
        "uav_get_run_events",
        "uav_get_spectrum_situation",
        "uav_measure_spectrum_at_current_pose",
    )
    document = capability_document()
    assert "raw_mavlink" in document["not_exposed"]
    assert "arbitrary_velocity" in document["not_exposed"]
    assert "arbitrary_frequency" in document["not_exposed"]
    assert "arbitrary_sionna_scene" in document["not_exposed"]
    assert document["scope"] == "px4_gazebo_simulation_only"


def test_mcp_spectrum_tools_expose_only_current_state_and_current_pose_measurement(monkeypatch):
    from backend.mcp import uav_spectrum_tools

    status = {"runtime": {"state": "running", "camera": {"vehicle": {"position_m": [1.0, 2.0, 20.0]}}}}
    situation = {"available": True, "current": True, "observation": {"source": "sionna_rt"}}
    observed = {
        "kind": "spectrum_observation", "source": "sionna_rt", "profile_id": "sionna_urban_2_4ghz",
        "timestamp": 100.0, "position_m": [1.0, 2.0, 20.0],
    }
    monkeypatch.setattr(uav_spectrum_tools, "get_runtime_status", lambda: status)
    monkeypatch.setattr(uav_spectrum_tools, "get_sionna_spectrum_situation", lambda **kwargs: situation)
    monkeypatch.setattr(uav_spectrum_tools, "collect_current_sionna_observation", lambda **kwargs: observed)

    assert uav_spectrum_tools.get_spectrum_situation() == situation
    result = uav_spectrum_tools.measure_spectrum_at_current_pose()
    assert result["observation"] == observed
    assert result["current"] is True


def test_native_registry_exposes_the_same_declarative_plan_schema():
    from backend.tools.registry import get_schema, register_all

    register_all()
    schema = get_schema("execute_uav_mission_plan")

    assert schema is not None
    props = schema["function"]["parameters"]["properties"]
    assert props["template"]["enum"] == [
        "takeoff_and_hover", "hover", "land", "return_to_launch",
        "inspect_safe_perimeter", "collect_camera_evidence", "search_safe_route",
    ]
    assert "target_enu_m" not in props
    assert "velocity_mps" not in props
