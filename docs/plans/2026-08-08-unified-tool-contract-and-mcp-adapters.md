# Unified Tool Contract and MCP Adapters Implementation Plan

> **For Claude:** Use `${SUPERPOWERS_SKILLS_ROOT}/skills/collaboration/executing-plans/SKILL.md` to implement this plan task-by-task.

**Goal:** 将 SpectrumClaw 的工具名称、Schema、Handler、执行策略和暴露策略收敛到单一 `ToolSpec`，并从同一 Registry 派生 Native、LLM、LangChain 与 MCP 适配器。

**Architecture:** `backend.tools.registry` 保存唯一的 `ToolSpec`；`ToolRuntime` 负责调用、超时和结构化错误；各调用表面只负责协议转换。进程内 Agent 默认走 Native Runtime，外部 Agent 通过 MCP 发现和调用同一批明确允许暴露的工具，旧注册函数保留为兼容门面。

**Tech Stack:** Python 3.13、dataclasses、asyncio、FastMCP、LangChain、pytest。

---

### Task 1: 建立统一工具契约和执行运行时

**Files:**
- Create: `backend/tools/contracts.py`
- Create: `backend/tools/runtime.py`
- Modify: `backend/tools/registry.py`
- Modify: `backend/tools/executors.py`
- Test: `tests/test_tool_runtime.py`

**Step 1: Write the failing tests**

测试 `ToolSpec` 能生成 OpenAI Schema、Registry 能登记动态工具、`ToolRuntime` 能执行同步和异步 Handler，并对未知工具、超时和 Handler 异常返回统一 JSON 错误。

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_tool_runtime.py`

Expected: FAIL，因为 `backend.tools.contracts` 和 `backend.tools.runtime` 尚不存在。

**Step 3: Write minimal implementation**

新增不可变 `ToolSpec` 与 `ToolExposure`；Registry 保存 `ToolSpec`，同时保留 `register/get_handler/get_schema` 兼容 API；执行器统一委托 `ToolRuntime`。

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_tool_runtime.py tests/test_uav_agent_tools.py`

Expected: PASS。

### Task 2: 将 LLM 与 LangChain 适配器迁到单一 Registry

**Files:**
- Modify: `backend/llm/client.py`
- Modify: `backend/llm/tools.py`
- Modify: `backend/tools/langchain_tools.py`
- Modify: `backend/app.py`
- Test: `tests/test_tool_adapters.py`
- Test: `tests/test_chat_api.py`

**Step 1: Write the failing tests**

测试通过旧 `backend.llm.client.register_tool` 登记的工具实际进入统一 Registry，LLM Schema 与 LangChain Tool 都读取同一个 `ToolSpec`，且内置工具启动注册不再维护第二份运行时 Registry。

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_tool_adapters.py`

Expected: FAIL，因为 LLM Client 仍使用私有 `TOOL_REGISTRY`。

**Step 3: Write minimal implementation**

让 LLM 注册、Schema 发现和执行委托统一 Registry/Runtime；将 `backend.llm.tools.register_default_tools` 收敛为兼容门面；LangChain 从 `ToolSpec` 构建适配器。

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_tool_adapters.py tests/test_chat_api.py`

Expected: PASS。

### Task 3: 从统一契约自动生成 MCP 工具表面

**Files:**
- Create: `backend/tools/mcp_adapter.py`
- Create: `backend/mcp/tool_server.py`
- Modify: `backend/tools/registry.py`
- Test: `tests/test_tool_mcp_adapter.py`

**Step 1: Write the failing tests**

使用不依赖 MCP SDK 的 Fake Server 验证只有 `mcp_enabled=True` 的工具被注册，名称、描述和 Handler 均来自同一个 `ToolSpec`；Native-only 工具不得暴露。

**Step 2: Run tests to verify they fail**

Run: `pytest -q tests/test_tool_mcp_adapter.py`

Expected: FAIL，因为 MCP 自动注册适配器尚不存在。

**Step 3: Write minimal implementation**

新增 `register_mcp_tools(server, registry)` 薄适配器；新增通用 MCP Server 入口，默认 stdio，并只暴露明确标记的知识、Web 和频率规划能力。UAV Server 继续保留独立安全域和现有策略边界。

**Step 4: Run tests to verify they pass**

Run: `pytest -q tests/test_tool_mcp_adapter.py tests/test_uav_mcp_contract.py tests/test_uav_mcp_client.py`

Expected: PASS。

### Task 4: 回归验证和文档同步

**Files:**
- Modify: `docs/TECHNICAL_OVERVIEW.md`
- Modify: `docs/uav-simulation/MCP_FLIGHT_ADAPTER.md`

**Step 1: Run focused regression tests**

Run: `pytest -q tests/test_tool_runtime.py tests/test_tool_adapters.py tests/test_tool_mcp_adapter.py tests/test_chat_api.py tests/test_uav_agent_tools.py tests/test_uav_mcp_contract.py tests/test_uav_mcp_client.py`

Expected: PASS。

**Step 2: Run broader backend regression tests**

Run: `pytest -q tests/test_agent_runtime.py tests/test_agent_graph.py tests/test_uav_agent_e2e.py`

Expected: PASS；若仓库不存在某个测试文件，则改为运行现有对应 Agent 测试集合并记录。

**Step 3: Verify import and diff integrity**

Run: `python -m compileall -q backend/tools backend/llm backend/mcp`

Run: `git diff --check`

Expected: 两条命令均退出 0。

**Step 4: Document the extension rule**

明确后续新增工具只创建一个 `ToolSpec`；Native 是进程内默认调用面，MCP 是可选的外部标准调用面，低层控制能力保持私有且不得因统一注册而自动公开。
