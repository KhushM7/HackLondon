import { CongestionBand } from '../lib/api';

// Layout constants
const ML = 48;  // margin left  (Y-axis labels)
const MR = 12;  // margin right
const MT = 18;  // margin top
const MB = 58;  // margin bottom (rotated X-axis labels)
const CW = 580; // chart area width
const CH = 180; // chart area height
const W = ML + CW + MR;   // total viewBox width  = 640
const H = MT + CH + MB;   // total viewBox height = 256

function barColor(ratio: number): string {
  if (ratio > 0.66) return '#ff8059';
  if (ratio > 0.33) return '#ffd56b';
  return '#6de5c8';
}

export function CongestionChart({ bands }: { bands: CongestionBand[] }) {
  if (bands.length === 0) {
    return <p className="disclaimer">No data yet — run a TLE refresh to populate.</p>;
  }

  const maxCount = Math.max(...bands.map((b) => b.object_count), 1);
  const barW = CW / bands.length;

  // Show at most ~18 X-axis labels so they never overlap
  const labelStep = Math.max(1, Math.ceil(bands.length / 18));

  // Y-axis ticks at 0 %, 25 %, 50 %, 75 %, 100 %
  const yTicks: { frac: number; label: string }[] = [
    { frac: 0,    label: '0' },
    { frac: 0.25, label: String(Math.round(maxCount * 0.25)) },
    { frac: 0.5,  label: String(Math.round(maxCount * 0.5)) },
    { frac: 0.75, label: String(Math.round(maxCount * 0.75)) },
    { frac: 1,    label: String(maxCount) },
  ];

  return (
    <>
      <div className="disclaimer">
        10 km bins · object count per altitude band · hover a bar for details
      </div>

      <svg
        width="100%"
        viewBox={`0 0 ${W} ${H}`}
        preserveAspectRatio="xMidYMid meet"
        style={{ display: 'block' }}
      >
        {/* Horizontal gridlines + Y-axis tick labels */}
        {yTicks.map(({ frac, label }) => {
          const y = MT + CH - frac * CH;
          return (
            <g key={frac}>
              <line
                x1={ML} y1={y} x2={ML + CW} y2={y}
                stroke="rgba(255,255,255,0.07)" strokeWidth="1"
              />
              <text x={ML - 6} y={y + 4} fill="#9bb1c8" fontSize="9" textAnchor="end">
                {label}
              </text>
            </g>
          );
        })}

        {/* Y-axis spine */}
        <line
          x1={ML} y1={MT} x2={ML} y2={MT + CH}
          stroke="rgba(255,255,255,0.18)" strokeWidth="1"
        />
        {/* X-axis baseline */}
        <line
          x1={ML} y1={MT + CH} x2={ML + CW} y2={MT + CH}
          stroke="rgba(255,255,255,0.18)" strokeWidth="1"
        />

        {/* Bars */}
        {bands.map((band, i) => {
          const ratio = band.object_count / maxCount;
          const x = ML + i * barW;
          const h = Math.max(ratio * CH, 1);
          const y = MT + CH - h;
          const tooltip =
            `${band.altitude_band_km} km\n` +
            `${band.object_count} objects\n` +
            `Conj. rate: ${band.conjunction_rate.toFixed(3)}`;
          return (
            <rect
              key={band.altitude_band_km}
              x={x}
              y={y}
              width={Math.max(barW - 0.5, 0.5)}
              height={h}
              fill={barColor(ratio)}
              opacity="0.82"
            >
              <title>{tooltip}</title>
            </rect>
          );
        })}

        {/* X-axis labels — every labelStep bars, rotated −45° */}
        {bands.map((band, i) => {
          if (i % labelStep !== 0) return null;
          const cx = ML + i * barW + barW / 2;
          const cy = MT + CH + 6;
          // Use only the lower bound of the range ("200-210" → "200")
          const label = band.altitude_band_km.split('-')[0];
          return (
            <text
              key={band.altitude_band_km}
              x={cx}
              y={cy}
              fill="#9bb1c8"
              fontSize="9"
              textAnchor="end"
              transform={`rotate(-45 ${cx} ${cy})`}
            >
              {label}
            </text>
          );
        })}

        {/* Axis titles */}
        <text
          x={ML + CW / 2}
          y={H - 3}
          fill="#9bb1c8"
          fontSize="9"
          textAnchor="middle"
        >
          Altitude (km)
        </text>
        <text
          x={10}
          y={MT + CH / 2}
          fill="#9bb1c8"
          fontSize="9"
          textAnchor="middle"
          transform={`rotate(-90 10 ${MT + CH / 2})`}
        >
          Objects
        </text>

        {/* Colour legend */}
        {(
          [
            { color: '#6de5c8', label: 'Low' },
            { color: '#ffd56b', label: 'Medium' },
            { color: '#ff8059', label: 'High' },
          ] as const
        ).map(({ color, label }, i) => (
          <g key={label} transform={`translate(${ML + CW - 140 + i * 48}, ${MT - 2})`}>
            <rect width="10" height="10" fill={color} opacity="0.82" />
            <text x="13" y="9" fill="#9bb1c8" fontSize="8">{label}</text>
          </g>
        ))}
      </svg>
    </>
  );
}
