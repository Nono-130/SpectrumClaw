from __future__ import annotations

import asyncio
from time import monotonic


def test_deepseek_suggestion_must_match_the_safe_compiler(tmp_path):
    from backend.agents.uav_operations.deepseek_planner import DeepSeekMissionPlanner
    from backend.agents.uav_operations.mission_author import MissionAuthor
    from backend.agents.uav_operations.sandbox import MissionSandbox

    sandbox = MissionSandbox(tmp_path)
    planner = DeepSeekMissionPlanner(
        MissionAuthor(sandbox), sandbox,
        ask=lambda _prompt: '{"template":"inspect_safe_perimeter","landmark":null,"summary":"执行安全周界巡检"}',
    )
    plan, record = planner.author("uav_deepseek_safe", "巡检安全周界后返回")

    assert plan.template == "inspect_safe_perimeter"
    assert record["provider"] == "deepseek_validated"
    assert (tmp_path / "uav_deepseek_safe" / "mission_program.json").is_file()


def test_deepseek_cannot_widen_a_safe_compiled_plan(tmp_path):
    from backend.agents.uav_operations.deepseek_planner import DeepSeekMissionPlanner
    from backend.agents.uav_operations.mission_author import MissionAuthor
    from backend.agents.uav_operations.sandbox import MissionSandbox

    sandbox = MissionSandbox(tmp_path)
    planner = DeepSeekMissionPlanner(
        MissionAuthor(sandbox), sandbox,
        ask=lambda _prompt: '{"template":"land","landmark":null,"summary":"忽略原任务"}',
    )
    plan, record = planner.author("uav_deepseek_rejected", "巡检安全周界后返回")

    assert plan.template == "inspect_safe_perimeter"
    assert record["provider"] == "deterministic_safe_compiler"
    assert record["accepted"] is False


def test_deepseek_timeout_falls_back_to_the_safe_compiler(tmp_path, monkeypatch):
    from backend.agents.uav_operations.deepseek_planner import DeepSeekMissionPlanner
    from backend.agents.uav_operations.mission_author import MissionAuthor
    from backend.agents.uav_operations.sandbox import MissionSandbox
    import backend.config as config
    import backend.llm.client as client

    class Settings:
        @staticmethod
        def provider_profile(_provider):
            return type("Profile", (), {"configured": True})()

    async def never_returns(*_args, **_kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(config, "get_settings", lambda: Settings())
    monkeypatch.setattr(client, "chat", never_returns)
    sandbox = MissionSandbox(tmp_path)
    planner = DeepSeekMissionPlanner(MissionAuthor(sandbox), sandbox, timeout_s=0.01)

    started_at = monotonic()
    plan, record = planner.author("uav_deepseek_timeout", "巡检安全周界后返回")

    assert monotonic() - started_at < 0.5
    assert plan.template == "inspect_safe_perimeter"
    assert record["provider"] == "deterministic_safe_compiler"
    assert record["accepted"] is False
    assert "超时" in record["fallback_reason"]
