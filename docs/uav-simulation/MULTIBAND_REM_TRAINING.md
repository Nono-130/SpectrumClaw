# 固定高度多频 REM 训练

当前训练契约为 `spectrumclaw.rem.hwf.v1`：

| 维度/字段 | 语义 |
|---|---|
| `H=128` | 北到南的 XY 空间行 |
| `W=128` | 西到东的 XY 空间列 |
| `F=5` | 五个中心频段，不是高度层 |
| `receiver_altitude_m=21` | 固定无人机测量平面，仅作元数据 |
| `source_xy_m=[1,2]` | 单个多频辐射源的 XY 监督标签 |

场景覆盖 120 m × 120 m，因此空间像素为 0.9375 m。训练数据与在线频谱地图现在共享同一套 128×128 ENU 空间网格；在线 Sionna RT 仍按 4 m 物理采样步长触发真实测量，再把测量投影并补全到细网格。这里的 4 m 仅是昂贵射线追踪的采样步长，不再是频谱地图分辨率，从而避免把一次航线扩大成 16,384 次 RT 调用。

## 数据字段

每个 NPZ 样本至少包含：

- `rss_dbm[128,128,5]`：五频段稠密真值；
- `sampled_rss_dbm[128,128,5]`：未测位置为 NaN；
- `observed_mask[128,128,5]`：逐频观测掩码；
- `frequencies_hz[5]`：频谱轴权威顺序；
- `building_mask/flyable_mask/valid_rx_mask[128,128]`；
- `source_xy_m[1,2]`、每频段发射功率和轨迹数据；
- `split_group_id` 与确定性 `split`，防止同一稠密场景跨数据划分泄漏；
- `schema_version=\"spectrumclaw.rem.hwf.v1\"`、`axis_order=\"HWF\"`。

契约拒绝旧的 `ZHW`、`height_map`、`source_z_m` 等字段。旧 Agent_UAV_REM 的五高度层结果继续由旧适配器读取，不能与本数据集混用。

## 生成 Pilot 数据

Sionna 环境中运行：

```bash
/root/miniconda3/envs/spectrumclaw-rt/bin/python \
  simulation/rf_engine/sionna_rem_dataset.py \
  --config config/uav_rem_dataset.yaml \
  --scenes 50 --masks-per-scene 8
```

生成器用 `RadioMapSolver` 对每个频段一次计算完整平面，而不是循环执行 81,920 次单点射线追踪。调试数据契约时可临时加 `--provider analytic`；该模式不能作为实验数据或论文结果。

正式扩展数据量之前，应在 3090 上完成：

1. 单场景五频段的运行时间和显存基准；
2. 用 `cell_centers` 核对 Sionna 南北方向，确保存盘 row 0 为北；
3. 抽取 16–32 个网格中心，对比 `RadioMapSolver` 与在线 `PathSolver` 的 dB 差；
4. 按 `split_group_id` 检查同一稠密地图没有跨 train/validation/test。

截至 2026-08-12，最终代码已完成单场景真实 Sionna 冒烟：五个频段生成并严格回读为 `[128,128,5]`，固定高度 21 m、486 个飞行轨迹点（470 个唯一五频同步观测点），仅包含 XY 定位标签；该次五频地图生成耗时约 2.53 s。另抽查 32 个 2.4 GHz 网格中心，其中 31 个有效点的 `RadioMapSolver`/在线 `PathSolver` 绝对差中位数为 0.10 dB、P90 为 0.73 dB、最大值为 3.31 dB。这里记录的是数据链路一致性，不代表补全或定位精度。

## 训练

训练依赖使用独立环境，避免 Torch 与 Sionna/PX4 主进程耦合：

```bash
python -m pip install -r config/dependencies/requirements-rem-training.txt
python scripts/training/train_multiband_rem.py \
  --data data/uav-rem-multiband \
  --output outputs/uav-rem-multiband \
  --epochs 50 --batch-size 4
```

模型是概率残差 U-Net：输入稀疏 RSS、逐频 mask、IDW 物理先验、建筑图、XY 编码和频率编码；输出五频 REM 均值、对数方差、共享辐射源连续 XY 和 2×2 协方差。实测点在输出端硬覆盖，保证不会被网络改写。

第一阶段先训练补全/定位模型并建立信息增益规划基线。强化学习仅在模型稳定后学习安全候选航点，不直接输出姿态或底层飞控指令。
