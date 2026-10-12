## A. 总体结论

| 项目 | 结论 |
|---|---|
| 当前整体新颖性 | **5.8 / 10** |
| 决策 | **PROCEED WITH CAUTION** |
| 收敛后的潜力 | 若只保留 C2 为主算法，并把 C1/C3 变成支撑机制，可提升到约 **7.5 / 10** |
| 最大风险 | 当前像“工具型无线智能体 + 形式化验证 + 主动测量 + conformal + memory + benchmark”的模块拼装，缺少一个不可替代的核心算法 |

这个方向可以继续，但不能把五项都作为并列“首次贡献”。直接无线工作已经覆盖工具编排、EM 仿真、radio map、无线工作流搜索和无线 benchmark；跨领域顶会又分别覆盖形式化规划、决策感知实验设计、共形风险控制、工作流记忆和动态物理 benchmark。真正可能站住的主线，是：

> 面向动态频谱干扰诊断的、具有模拟器失配鲁棒性的异构信息源决策采集算法，而不是“构建一个频谱智能体”。

必须特别补入以下此前遗漏、但足以击穿宽泛主张的工作：

- [Amortized Bayesian Experimental Design for Decision-Making, NeurIPS 2024](https://proceedings.neurips.cc/paper_files/paper/2024/hash/c59f05d7ab3638b138cc61f32e1a7cd1-Abstract-Conference.html)：已经直接以 downstream decision utility 选择实验。
- [Conformal Constrained Policy Optimization, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/39739)：已经在可靠性约束下编排多个模型和工具、联合优化成本和风险门限。
- [Robots That Ask For Help, CoRL 2023](https://proceedings.mlr.press/v229/ren23a.html)：已用 conformal prediction 决定执行还是向人求助，并提供多步任务成功率保证。
- [Hive, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/b9a4215e2b23261056aeeba0f6f05371-Abstract-Conference.html) 与 [LLMFP, ICLR 2025](https://proceedings.iclr.cc/paper_files/paper/2025/hash/a1c8a68e52499c9396854e3f967e37c0-Abstract-Conference.html)：已覆盖 LLM 生成形式化表示、约束规划和可执行验证。
- [Active Fault Diagnosis under Bayesian Minimum Risk, IEEE TII 2023](https://doi.org/10.1109/TII.2023.3240736)：在 LLM 之前就已按误诊损失主动设计输入。因此，“以最终诊断风险驱动测量”本身也不是首次。
- [PhysGym, NeurIPS 2025](https://proceedings.neurips.cc/paper_files/paper/2025/hash/42abcf9dffa48ca44fea1c497539a914-Abstract-Datasets_and_Benchmarks_Track.html)：已有动态物理环境、主动探测、受约束顺序交互 benchmark。

无线领域的 WirelessAgent++、WirelessBench 和 AgentAbstain 当前是未审预印本；它们不能作为方法有效性的高质量证据，但依然会影响新颖性优先权，审稿人不会因为其未正式发表就忽略明显重叠。

## B. C1–C5 逐项判断

| 主张 | 评级 | 最接近工作与重叠 | 必须保留的 delta |
|---|---|---|---|
| C1 有类型的 evidence+physics+regulation 行动图 | **MEDIUM** | MAINTAINED、RadioSim Agent 已有无线工具/物理仿真编排；Hive、LLMFP 已有形式化行动与约束求解；C-RAG 已有检索风险认证 | 不能只是 LangGraph DAG。必须定义具有静态语义的 RF 中间表示：量纲、频段、坐标系、标准版本、工具前后置条件；每条结论产生可机器检查的 proof object，同时包含原文证据、计算依赖、单位推导、仿真残差和适用域 |
| C2 decision-aware active acquisition | **MEDIUM，五项中最强** | NeurIPS TNDP 已直接优化 downstream decision utility；IEEE TII 主动故障诊断已最小化误诊风险；TWC/TMC/TCCN 已有主动频谱测绘和 UAV 路径规划；AAAI CCPO 已按可靠性/成本选择模型与工具 | 必须研究“检索、询问、仿真、真实测量”四类异构且有偏的信息源；处理动态频谱、测量延迟、模拟器失配、跨动作相关性及法规约束。若只是给不同工具打分或使用普通熵下降，评级降为 LOW |
| C3 calibrated abstention/escalation | **LOW** | KnowNo、C-RAG、Conformal Factuality、CCPO、CoFineLLM，以及 AgentAbstain 已覆盖共形保证、求助、升级和拒绝执行 | 只有在“策略自适应取证导致分布漂移”的序列情形下，给出终端缓解决策风险保证，才可能成为配套技术点。固定阈值、模型置信度或普通 split conformal 没有新颖性 |
| C4 verified experience memory | **LOW** | Reflexion、AWM、A-Mem、MetaFlow、Agentic Plan Caching、APIGen-MT 已覆盖成功轨迹筛选、工作流提炼和复用 | “只存成功轨迹”接近工程常识。至少需加入适用域、环境版本、标准有效期、物理不变量以及失效传播；证明验证门控能降低 memory poisoning 或错误复用。更适合作为消融项，不应列为主贡献 |
| C5 SpectrumAgentBench | **MEDIUM** | WirelessBench 已有无线题、工具必要、3GPP ray tracing、数值容差和灾难错误；AgentAbstain 有成对 act/abstain 与不可逆动作；τ-bench 有动态状态；PhysGym 有交互物理发现 | 必须是可执行的动态 RF 环境，而非静态问答。需包含隐藏发射源/传播状态、动作导致的时间演化、标准版本、真实工具调用、成对行动/拒答任务、确定性终态判分、污染隔离和专家验证。否则会被视为 WirelessBench 的扩展 |

## C. 不能声称“首次”的内容

以下表述会被审稿人立即击穿：

- “首个无线/频谱 LLM 智能体。”
- “首次把 LLM 与无线规划工具、传播模型或 EM 仿真器结合。”
- “首次用智能体生成 radio map 或规划基站。”
- “首次使用主动测量/UAV 路径进行频谱感知。”
- “首次按最终决策效用设计实验或测量。”
- “首次用 conformal prediction 控制 LLM 可靠性、拒答或向人求助。”
- “首次从成功轨迹中抽取并复用工作流记忆。”
- “首个无线智能体 benchmark”或“首次工具必要、数值容差评测。”
- “首个形式化/神经符号 LLM 行动图。”
- “消除幻觉”“保证物理正确”或“保证法规合规”。

以下写法同样危险：

- 仅有文献引用就称“法规合规”；标准的版本、地区、业务类型和例外条款必须机器可判。
- 几个链路预算公式通过就称“物理正确”；传播模型失配和输入错误仍会产生确定但错误的答案。
- 把多个提示角色称为“多智能体创新”。
- 仅用仿真数据却称“真实频谱自治管理”。
- 在普通 split-conformal 假设不成立的自适应策略轨迹上宣称有限样本保证。
- 把模块数量、工具数量或任务数量当作算法贡献。

## D. 唯一应保留的主算法贡献

建议只保留 C2，定义为：

> **Robust Hybrid Decision-Aware Spectrum Acquisition，面向模拟器失配和动态频谱的鲁棒异构价值信息采集。**

### 最小形式化

隐状态：

\[
x_t=(h_t,\theta_t,z_t)
\]

其中 \(h_t\) 是干扰机制/来源假设，\(\theta_t\) 是传播和设备参数，\(z_t\) 是随时间变化的频谱环境；智能体维护历史条件信念 \(b_t=P(x_t\mid \mathcal H_t)\)。

动作集合：

\[
\mathcal A=\mathcal A_{\rm ret}\cup\mathcal A_{\rm ask}
\cup\mathcal A_{\rm sim}\cup\mathcal A_{\rm meas}
\cup\mathcal A_{\rm decide}\cup\{\text{escalate}\}.
\]

不同采集动作具有成本 \(c(a)\)、延迟 \(\Delta(a)\) 和不确定观测通道。传播仿真器不是 ground truth，其条件分布属于失配集合 \(\mathcal U_a\)。

终端决策 \(d\) 可以是干扰源判断、频率调整、功率调整或拒绝执行。目标是：

\[
\min_{\pi,\tau,d}\;
\sup_{P\in\mathcal U}
\mathbb E_P
\left[
L(d,x_\tau)+\sum_{t<\tau}c(a_t)
\right]
\]

并满足：

\[
\inf_{P\in\mathcal U}
P\!\left(g_j(d,x_\tau)\le 0,\ \forall j\right)
\ge 1-\delta .
\]

定义鲁棒终端 Bayes 风险：

\[
R_{\mathcal U}(b)=
\min_{d\in\mathcal D_{\rm valid}}
\sup_{Q\in\mathcal U(b)}
\mathbb E_Q[L(d,X)].
\]

动作评分为鲁棒期望样本价值：

\[
{\rm rEVSI}(a;b)=R_{\mathcal U}(b)-c(a)
-\sup_{P_a\in\mathcal U_a}
\mathbb E_{O\sim P_a}
\left[R_{\mathcal U}(T(b,a,O))\right].
\]

选择正 rEVSI 最大的动作；若所有采集动作收益非正，且终端风险上界低于阈值，则决策；否则升级人工。

### 至少一个必须实现的非显然技术点

仿真结果和真实测量不能被当作条件独立证据。多次仿真共享同一个未知传播偏差，若普通贝叶斯更新把它们当作独立样本，智能体会产生虚假的高置信度。论文应：

1. 显式引入共享模拟器失配变量；
2. 构造跨检索、仿真、测量的耦合 ambiguity set；
3. 推导可计算的 rEVSI 上下界或近似算法；
4. 在策略诱导的分布漂移下给出终端风险或安全停止保证。

理想理论结果是：当失配集合以至少 \(1-\beta\) 概率覆盖真实观测模型，且策略条件风险控制以至少 \(1-\alpha\) 成立时，终端违规概率有 \(\alpha+\beta\) 类型上界。这个结论必须严格证明，不能直接引用普通 conformal 结论。

LLM 在此算法中只生成候选动作和解释，不参与数值保证；否则审稿人会认为算法本质仍是不可控提示工程。

## E. 最强拒稿论证与反驳路径

### 最强拒稿论证

本文没有提出新的智能体或频谱算法，而是把现有模块组合到一个无线应用中。无线工具编排、EM 仿真和规划已由 MAINTAINED、RadioSim Agent 等覆盖；形式化行动规划已有 Hive 和 LLMFP；决策感知实验设计已有 TNDP，贝叶斯最小风险主动诊断甚至早于 LLM；共形拒答和求助已有 KnowNo、C-RAG 和 CCPO；成功轨迹记忆已有 AWM；WirelessBench 已提供工具必要和数值容差评测。论文所谓五项贡献均缺乏独立算法 delta，且若主要在作者自建仿真任务上评估，提升可能来自工具质量和提示工程，而非方法。没有真实 RF 测量、失配分析、理论保证和强基线，结论不足以支持“可信频谱智能体”。

### 能够反驳它的修改

- 删除五个并列主贡献，只保留鲁棒异构决策采集算法。
- 将 C1 作为确定性安全层，C3 作为停止规则，C4 仅作消融，C5 作为评测基础设施。
- 与 TNDP、传统主动故障诊断、频谱测绘、CCPO、ReAct、WirelessAgent++ 风格工作流进行同任务比较。
- 至少加入真实 SDR 数据或小规模外场采集，且训练/校准场景与测试场景分离。
- 系统改变地形、频段、发射功率、干扰类型、模型失配、标准版本和传感器噪声。
- 报告决策 regret、违规率、coverage-risk、测量成本、仿真调用成本和失配下的校准误差，而不只报告回答准确率。
- 对 benchmark 使用隐藏测试状态、程序化生成、专家双盲核验和独立 baseline 实现。

## F. 投稿方向

|  venue | 适配度 | 必须具备的证据 |
|---|---:|---|
| **IEEE TCCN** | **最高** | 认知频谱、动态信念、主动测量、风险控制；完整算法、无线仿真、SDR/测量验证 |
| **IEEE TWC** | **高** | 更严格的信道/干扰模型、理论推导、失配鲁棒性、跨场景无线性能；LLM 只能是外层接口 |
| IEEE TCOM | 中 | 需把问题提升为通信网络优化/管理机制，并给出收敛、复杂度或性能界 |
| IEEE JSAC | 中高但依赖专题 | 需要更完整的统一理论、系统和较大规模实验，单一 agent demo 不够 |
| MobiCom | 有条件 | 必须有真实 SDR、UAV 或多站点部署，展示端到端系统效果和实际故障案例 |
| SIGCOMM | 偏低 | 需要生产级网络管理、运营商数据或大规模部署；当前频谱诊断范围偏窄 |
| NeurIPS / ICML | 当前偏低 | 算法必须脱离频谱仍成立，具有新理论，并至少在频谱之外再验证一个科学实验或诊断领域 |

最终建议：以 **IEEE TCCN/TWC** 为第一目标。先完成“鲁棒异构决策采集”这一条算法线，再将 SpectrumClaw 做成其可复现实验平台。不要反过来先堆出一个功能繁多的智能体，再为已有模块寻找创新表述。
