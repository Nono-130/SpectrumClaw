import { useEffect, useMemo, useRef } from "react";

import { spectrumGridRaster } from "../../lib/spectrumGrid.js";


export default function SpectrumGridCanvas({ grid, colorRange, className = "" }) {
  const canvasRef = useRef(null);
  const raster = useMemo(() => spectrumGridRaster(grid, colorRange), [colorRange, grid]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas || raster.rows < 1 || raster.columns < 1) return;
    canvas.width = raster.columns;
    canvas.height = raster.rows;
    const context = canvas.getContext("2d", { alpha: true });
    if (!context) return;
    const image = context.createImageData(raster.columns, raster.rows);
    image.data.set(raster.pixels);
    context.putImageData(image, 0, 0);
  }, [raster]);

  return (
    <canvas
      ref={canvasRef}
      className={`uav-rem-raster${className ? ` ${className}` : ""}`}
      role="img"
      aria-label={`${raster.rows} × ${raster.columns} 在线频谱空间网格`}
    />
  );
}
