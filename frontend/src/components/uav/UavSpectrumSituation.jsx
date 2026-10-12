import { Activity, CircleDot, Clock3, Crosshair, Layers3, Map as MapIcon, Radio, RefreshCw, Route, Signal, Zap } from "lucide-react";
import { useMemo } from "react";
import SpectrumGridCanvas from "./SpectrumGridCanvas.jsx";
import { spectrumColorRange } from "../../lib/spectrumGrid.js";

const MAP_RANGE_M = 60;
const RAY_COLORS = ["#f7c948", "#48d8ff", "#b787ff", "#ff7a9f", "#64e7a3", "#ff9f43", "#83a7ff"];

function mapPoint([east = 0, north = 0]) {
  const x = ((Number(east) + MAP_RANGE_M) / (MAP_RANGE_M * 2)) * 100;
  const y = 100 - ((Number(north) + MAP_RANGE_M) / (MAP_RANGE_M * 2)) * 100;
  return [Math.max(2, Math.min(98, x)), Math.max(2, Math.min(98, y))];
}

function dbmLabel(value) {
  return Number.isFinite(Number(value)) ? `${Number(value).toFixed(1)} dBm` : "—";
}

function ageLabel(age) {
  if (!Number.isFinite(Number(age))) return "—";
  return age < 60 ? `${Math.round(age)} s 前` : `${Math.floor(age / 60)} min 前`;
}

function displayFrequency(hz) {
  return Number.isFinite(Number(hz)) ? `${(Number(hz) / 1e9).toFixed(2)} GHz` : "2.40 GHz";
}

function pointString(points) {
  return points.map((point) => mapPoint(point).join(",")).join(" ");
}

function pathColor(path) {
  const key = `${path.anchorId || ""}:${path.id || ""}`;
  const index = [...key].reduce((total, character) => total + character.charCodeAt(0), 0) % RAY_COLORS.length;
  return RAY_COLORS[index];
}

function samplePower(sample, selectedTx) {
  const anchors = Array.isArray(sample?.anchors) ? sample.anchors : [];
  const values = anchors
    .filter((anchor) => selectedTx === "all" || anchor.id === selectedTx)
    .map((anchor) => Number(anchor.received_power_dbm))
    .filter(Number.isFinite);
  return values.length ? Math.max(...values) : null;
}

// This is deliberately a compact orientation aid. The authoritative spatial
// rendering is the WebGL scene immediately beside it, where the same ENU
// coordinates carry the drone, transmitters and real Sionna ray vertices.
export default function UavSpectrumSituation({ scene, vehicle, situation, grid, loading, error, onRefresh, selectedTx, onSelectTx }) {
  const objects = Array.isArray(scene?.objects) ? scene.objects : [];
  const transmitters = Array.isArray(scene?.transmitters) ? scene.transmitters : [];
  const observation = situation?.observation;
  const measurements = Array.isArray(observation?.anchors) ? observation.anchors : [];
  const currentPosition = Array.isArray(vehicle?.position_m) ? vehicle.position_m : situation?.current_position_m;
  const rayPaths = useMemo(() => measurements.flatMap((anchor) => (Array.isArray(anchor.ray_paths) ? anchor.ray_paths.map((path) => ({ ...path, anchorId: anchor.id })) : [])), [measurements]);
  const samples = Array.isArray(situation?.history)
    ? situation.history.filter((sample) => Number(sample?.captured_at) >= (Date.now() / 1000) - 1800).slice(0, 8).reverse()
    : [];
  const visiblePaths = selectedTx === "all" ? rayPaths : rayPaths.filter((path) => path.anchorId === selectedTx);
  const pathTotal = rayPaths.length;
  const status = situation?.available ? situation.current ? "current" : "stale" : "empty";
  const statusLabel = situation?.computing ? "RT 计算中" : status === "current" ? "实时采集中" : status === "stale" ? "样本待更新" : "等待测量";
  const [vehicleX, vehicleY] = currentPosition ? mapPoint(currentPosition) : [50, 50];
  const [gridRows = 128, gridColumns = 128] = grid?.shape || [128, 128];
  const layerRange = grid?.height_range_m || [0, 5];

  return (
    <aside className="uav-spectrum-inspector uav-spectrum-inspector-card" aria-label="频谱链路检查器">
      <header className="uav-spectrum-inspector-head">
        <div>
          <span className="eyebrow">SIONNA RT · RF OVERLAY</span>
          <h2><Radio size={17} /> 电磁频谱态势</h2>
        </div>
        <span className="uav-spectrum-state" data-state={status}><i /> {statusLabel}</span>
      </header>

      <div className="uav-spectrum-kpis">
        <div><span>高度层</span><strong>{layerRange[0]}–{layerRange[1]} m</strong><small>{gridRows} × {gridColumns} grid</small></div>
        <div><span>实测网格</span><strong>{grid?.observed_cells ?? 0}</strong><small>{((grid?.coverage_ratio || 0) * 100).toFixed(2)}% coverage</small></div>
        <div><span>真实路径</span><strong>{pathTotal}</strong><small>{displayFrequency(observation?.frequency_hz)}</small></div>
      </div>

      <div className="uav-spectrum-inspector-title"><Route size={13} /> 发射源与链路</div>
      <div className="uav-spectrum-source-list" role="group" aria-label="选择查看发射源路径">
        <button type="button" className={selectedTx === "all" ? "active" : ""} onClick={() => onSelectTx("all")}><span><CircleDot size={12} /> 全部链路</span><strong>{rayPaths.length}</strong></button>
        {measurements.map((anchor) => <button type="button" key={anchor.id} className={selectedTx === anchor.id ? "active" : ""} onClick={() => onSelectTx(anchor.id)}><span><Signal size={12} /> {anchor.id}</span><strong>{dbmLabel(anchor.received_power_dbm)}</strong><small>{anchor.path_count ?? 0} 条路径</small></button>)}
      </div>

      <div className="uav-spectrum-mini-map" role="img" aria-label="无人机与发射源的二维频谱态势概览">
        <SpectrumGridCanvas grid={grid} colorRange={spectrumColorRange(grid)} className="is-mini" />
        <svg viewBox="0 0 100 100" preserveAspectRatio="xMidYMid meet">
          <defs><pattern id="rf-mini-grid" width="10" height="10" patternUnits="userSpaceOnUse"><path d="M10 0H0V10" fill="none" stroke="currentColor" strokeWidth=".35" /></pattern></defs>
          <rect className="uav-spectrum-ground" width="100" height="100" rx="3" />
          <rect className="uav-spectrum-grid" width="100" height="100" fill="url(#rf-mini-grid)" />
          <path className="uav-spectrum-road" d="M0 50 H100 M50 0 V100" />
          {objects.map((object) => { const [x, y] = mapPoint(object.position_m || []); const [width, height] = object.size_m || [8, 8]; return <rect key={object.id} className="uav-spectrum-building" x={x - Math.max(3, width / 3.3)} y={y - Math.max(3, height / 3.3)} width={Math.max(6, width / 1.65)} height={Math.max(6, height / 1.65)} rx="1.5" />; })}
          {visiblePaths.map((path) => <polyline className="uav-spectrum-mini-ray" key={`${path.anchorId}-${path.id}`} points={pointString(path.points_m || [])} style={{ stroke: pathColor(path) }} />)}
          {transmitters.map((transmitter) => { const [x, y] = mapPoint(transmitter.position_m || []); return <g key={transmitter.id} transform={`translate(${x} ${y})`} className="uav-spectrum-transmitter"><circle className="uav-spectrum-ring" r="6" /><circle className="uav-spectrum-tx" r="2.2" /></g>; })}
          {currentPosition && <g transform={`translate(${vehicleX} ${vehicleY})`} className="uav-spectrum-vehicle"><circle r="5" /><path d="M0 -4 L2 0 L4 2 L1 2 L0 4 L-1 2 L-4 2 L-2 0 Z" /></g>}
        </svg>
        <span><MapIcon size={11} /> ENU 概览</span><span><Layers3 size={11} /> {visiblePaths.length} 路径</span>
      </div>

      <div className="uav-spectrum-inspector-title"><Activity size={13} /> 实时采样序列</div>
      <div className="uav-spectrum-sample-strip" aria-label="Sionna RT 实时采样历史">
        {samples.length ? samples.map((sample, index) => {
          const power = samplePower(sample, selectedTx);
          const [east, north] = sample.position_m || [];
          return <div key={`${sample.run_id}-${sample.captured_at}`} className={index === samples.length - 1 ? "current" : ""}><span>S{index + 1}</span><strong>{dbmLabel(power)}</strong><small>{Number(east).toFixed(1)}, {Number(north).toFixed(1)} m</small></div>;
        }) : <span className="uav-spectrum-sample-empty">等待任务采样</span>}
      </div>

      <div className="uav-spectrum-inspector-title"><Activity size={13} /> 观测可信度</div>
      <dl className="uav-spectrum-facts">
        <div><dt><Clock3 size={12} /> 样本时间</dt><dd>{situation?.available ? ageLabel(situation.age_s) : "—"}</dd></div>
        <div><dt><Crosshair size={12} /> 位姿偏差</dt><dd>{situation?.pose_offset_m == null ? "待确认" : `${Number(situation.pose_offset_m).toFixed(2)} m`}</dd></div>
        <div><dt><Zap size={12} /> 计算来源</dt><dd>Sionna RT</dd></div>
      </dl>

      {!measurements.length && <p className="uav-spectrum-empty">{error || situation?.reason || "暂无频谱观测。更新测量会在当前无人机位姿执行一次受限 Sionna RT 计算，不会发送飞控命令。"}</p>}
      {measurements.length > 0 && rayPaths.length === 0 && <p className="uav-spectrum-empty">当前样本仅含链路功率统计；更新测量可生成含路径顶点的最新样本。</p>}
      <button className="uav-spectrum-refresh" type="button" onClick={onRefresh} disabled={loading}><RefreshCw size={13} className={loading ? "uav-spin" : ""} /> {loading ? "射线计算中…" : "更新测量"}</button>
      <footer className="uav-spectrum-footer"><span>三维画面仅叠加真实路径，不展示未经扫描验证的热力图。</span></footer>
    </aside>
  );
}
