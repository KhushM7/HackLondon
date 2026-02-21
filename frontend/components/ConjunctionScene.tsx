'use client';

import { Canvas } from '@react-three/fiber';

type Sample = { position_km: number[] };

function OrbitLine({ points, color }: { points: Sample[]; color: string }) {
  const scaled = points.map((p) => [p.position_km[0] / 2000, p.position_km[1] / 2000, p.position_km[2] / 2000]);
  return (
    <line>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={scaled.length}
          array={new Float32Array(scaled.flat())}
          itemSize={3}
        />
      </bufferGeometry>
      <lineBasicMaterial color={color} />
    </line>
  );
}

export function ConjunctionScene({
  defended,
  intruder
}: {
  defended: Sample[];
  intruder: Sample[];
}) {
  return (
    <div style={{ height: 420 }} className="panel">
      <Canvas camera={{ position: [0, 0, 6], fov: 50 }}>
        <ambientLight intensity={0.8} />
        <pointLight position={[6, 6, 8]} intensity={1.5} />
        <mesh>
          <sphereGeometry args={[3.2, 42, 42]} />
          <meshStandardMaterial color="#2a5b95" wireframe opacity={0.35} transparent />
        </mesh>
        <OrbitLine points={defended} color="#6de5c8" />
        <OrbitLine points={intruder} color="#ffb36d" />
      </Canvas>
    </div>
  );
}
