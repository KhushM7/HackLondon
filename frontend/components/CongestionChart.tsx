import { CongestionBand } from '../lib/api';

export function CongestionChart({ bands }: { bands: CongestionBand[] }) {
  const maxCount = Math.max(...bands.map((b) => b.object_count), 1);

  return (
    <div className="panel">
      <h3>Altitude Congestion Index</h3>
      <div className="disclaimer">10 km bins. Density and event rate from current catalog and recent screenings.</div>
      <svg width="100%" viewBox="0 0 680 260" preserveAspectRatio="none" style={{ marginTop: 12 }}>
        {bands.map((band, i) => {
          const barWidth = 560 / Math.max(bands.length, 1);
          const x = 40 + i * barWidth;
          const h = (band.object_count / maxCount) * 180;
          return (
            <g key={band.altitude_band_km}>
              <rect x={x} y={220 - h} width={Math.max(barWidth - 6, 6)} height={h} fill="#6de5c8" opacity="0.75" />
              <text x={x + 2} y={236} fill="#9bb1c8" fontSize="8">{band.altitude_band_km}</text>
            </g>
          );
        })}
      </svg>
    </div>
  );
}
