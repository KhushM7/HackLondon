'use client';

import { useEffect, useMemo, useRef, useState } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { SatellitePosition } from '../lib/api';

const EARTH_R_KM = 6371;
const S = 1 / EARTH_R_KM; // scene units: Earth radius = 1.0

// ECI TEME → Three.js (Y-up, north pole = +Y)
// ECI: x = vernal equinox, y = 90°E equatorial, z = north pole
function eciToScene(pos: [number, number, number]): [number, number, number] {
  return [pos[0] * S, pos[2] * S, pos[1] * S];
}

function riskColor(tier: string | null | undefined): string {
  if (tier === 'High') return '#ff6b6b';
  if (tier === 'Medium') return '#ffd56b';
  if (tier === 'Low') return '#76e68e';
  return '#3a6a9a';
}

function Controls() {
  const { camera, gl } = useThree();
  const ctrlRef = useRef<OrbitControls | null>(null);

  useEffect(() => {
    const ctrl = new OrbitControls(camera as THREE.PerspectiveCamera, gl.domElement);
    ctrl.enableDamping = true;
    ctrl.dampingFactor = 0.06;
    ctrl.minDistance = 1.4;
    ctrl.maxDistance = 14;
    ctrl.autoRotate = true;
    ctrl.autoRotateSpeed = 0.4;
    ctrlRef.current = ctrl;
    return () => ctrl.dispose();
  }, [camera, gl]);

  useFrame(() => {
    ctrlRef.current?.update();
  });

  return null;
}

function Earth() {
  const [texture, setTexture] = useState<THREE.Texture | null>(null);

  useEffect(() => {
    const loader = new THREE.TextureLoader();
    loader.load('/earth.jpg', (tex) => {
      tex.colorSpace = THREE.SRGBColorSpace;
      setTexture(tex);
    });
  }, []);

  return (
    <mesh>
      <sphereGeometry args={[1, 64, 64]} />
      <meshStandardMaterial
        map={texture ?? undefined}
        color={texture ? '#ffffff' : '#1a3d6e'}
        roughness={0.7}
        metalness={0.05}
      />
    </mesh>
  );
}

const GRID_R = 1.002; // slightly above surface so lines show over texture

function GridLines() {
  const latLines = useMemo(() => {
    const result: { key: string; arr: Float32Array }[] = [];
    for (let lat = -60; lat <= 60; lat += 30) {
      const pts: number[] = [];
      const latRad = (lat * Math.PI) / 180;
      const r = Math.cos(latRad) * GRID_R;
      const y = Math.sin(latRad) * GRID_R;
      for (let i = 0; i <= 64; i++) {
        const lon = (i / 64) * Math.PI * 2;
        pts.push(r * Math.cos(lon), y, r * Math.sin(lon));
      }
      result.push({ key: `lat${lat}`, arr: new Float32Array(pts) });
    }
    return result;
  }, []);

  const lonLines = useMemo(() => {
    const result: { key: string; arr: Float32Array }[] = [];
    for (let lon = 0; lon < 360; lon += 30) {
      const pts: number[] = [];
      const lonRad = (lon * Math.PI) / 180;
      for (let i = 0; i <= 64; i++) {
        const lat = ((i / 64) * 2 - 1) * Math.PI * 0.5;
        pts.push(
          Math.cos(lat) * Math.cos(lonRad) * GRID_R,
          Math.sin(lat) * GRID_R,
          Math.cos(lat) * Math.sin(lonRad) * GRID_R,
        );
      }
      result.push({ key: `lon${lon}`, arr: new Float32Array(pts) });
    }
    return result;
  }, []);

  return (
    <>
      {[...latLines, ...lonLines].map(({ key, arr }) => (
        <line key={key}>
          <bufferGeometry>
            <bufferAttribute attach="attributes-position" count={65} array={arr} itemSize={3} />
          </bufferGeometry>
          <lineBasicMaterial color="#ffffff" transparent opacity={0.15} />
        </line>
      ))}
    </>
  );
}

function Atmosphere() {
  return (
    <mesh>
      <sphereGeometry args={[1.04, 32, 32]} />
      <meshStandardMaterial color="#5bafd6" transparent opacity={0.07} side={THREE.BackSide} />
    </mesh>
  );
}

function SatDot({
  sat,
  selected,
  onClick,
}: {
  sat: SatellitePosition;
  selected: boolean;
  onClick: () => void;
}) {
  const pos = useMemo(() => eciToScene(sat.position_km), [sat.position_km]);
  const color = selected ? '#ffffff' : riskColor(sat.risk_tier);
  const size = selected ? 0.028 : sat.risk_tier ? 0.018 : 0.01;

  return (
    <mesh
      position={pos}
      onClick={(e) => {
        e.stopPropagation();
        onClick();
      }}
    >
      <sphereGeometry args={[size, 6, 6]} />
      <meshBasicMaterial color={color} />
    </mesh>
  );
}

function Scene({
  satellites,
  selectedId,
  onSelect,
  showAtRiskOnly,
}: {
  satellites: SatellitePosition[];
  selectedId: number | null;
  onSelect: (id: number | null) => void;
  showAtRiskOnly: boolean;
}) {
  const visible = showAtRiskOnly
    ? satellites.filter((s) => s.risk_tier === 'High' || s.risk_tier === 'Medium')
    : satellites;

  return (
    <>
      <ambientLight intensity={0.3} />
      <directionalLight position={[5, 3, 5]} intensity={1.5} />
      <pointLight position={[-8, -4, -6]} intensity={0.3} color="#3a6aaa" />
      <Controls />
      <Earth />
      <GridLines />
      <Atmosphere />
      {visible.map((sat) => (
        <SatDot
          key={sat.norad_id}
          sat={sat}
          selected={selectedId === sat.norad_id}
          onClick={() => onSelect(selectedId === sat.norad_id ? null : sat.norad_id)}
        />
      ))}
    </>
  );
}

export function GlobeView({
  satellites,
  selectedId,
  onSelect,
  showAtRiskOnly,
}: {
  satellites: SatellitePosition[];
  selectedId: number | null;
  onSelect: (id: number | null) => void;
  showAtRiskOnly: boolean;
}) {
  return (
    <div className="panel" style={{ padding: 0, overflow: 'hidden' }}>
      <div style={{ height: 520 }}>
        <Canvas camera={{ position: [0, 1.5, 3.8], fov: 45 }} gl={{ antialias: true }}>
          <Scene
            satellites={satellites}
            selectedId={selectedId}
            onSelect={onSelect}
            showAtRiskOnly={showAtRiskOnly}
          />
        </Canvas>
      </div>
      <div style={{ display: 'flex', gap: 16, padding: '0.6rem 1rem', fontSize: '0.8rem', color: 'var(--ink-muted)', flexWrap: 'wrap' }}>
        <span><span style={{ color: '#ff6b6b' }}>●</span> High risk</span>
        <span><span style={{ color: '#ffd56b' }}>●</span> Medium risk</span>
        <span><span style={{ color: '#76e68e' }}>●</span> Low risk</span>
        <span><span style={{ color: '#3a6a9a' }}>●</span> No conjunction</span>
        <span style={{ marginLeft: 'auto' }}>Drag to rotate · Scroll to zoom · Click satellite to select</span>
      </div>
    </div>
  );
}
