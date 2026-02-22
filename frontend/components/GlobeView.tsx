'use client';

import { useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { ConjunctionLink, SatellitePosition } from '../lib/api';

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

function riskLineColor(tier: string): string {
  if (tier === 'High') return '#ff7b7b';
  if (tier === 'Medium') return '#ffe08f';
  return '#9bf3b1';
}

function velocityToScene(vec: [number, number, number]): [number, number, number] {
  return [vec[0], vec[2], vec[1]];
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
  const matRef = useRef<THREE.MeshStandardMaterial>(null!);

  useEffect(() => {
    let active = true;
    const loader = new THREE.TextureLoader();
    loader.load('/earth.jpg', (tex) => {
      if (!active || !matRef.current) return;
      tex.colorSpace = THREE.SRGBColorSpace;
      matRef.current.map = tex;
      matRef.current.color.set(0xffffff);
      matRef.current.needsUpdate = true;
    });
    return () => { active = false; };
  }, []);

  return (
    <mesh>
      <sphereGeometry args={[1, 64, 64]} />
      <meshStandardMaterial ref={matRef} color="#1a3d6e" roughness={0.7} metalness={0.05} />
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

function DashedOrbit({ points, color = '#b8e9ff' }: { points: THREE.Vector3[]; color?: string }) {
  const lineRef = useRef<THREE.Line>(null);

  useEffect(() => {
    if (lineRef.current) {
      lineRef.current.computeLineDistances();
    }
  }, [points]);

  if (points.length < 2) return null;

  return (
    <line ref={lineRef}>
      <bufferGeometry>
        <bufferAttribute
          attach="attributes-position"
          count={points.length}
          array={new Float32Array(points.flatMap((p) => [p.x, p.y, p.z]))}
          itemSize={3}
        />
      </bufferGeometry>
      <lineDashedMaterial color={color} dashSize={0.04} gapSize={0.03} transparent opacity={0.75} />
    </line>
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
  links,
  orbitPath,
  relatedOrbits,
  selectedId,
  onSelect,
  showAtRiskOnly,
}: {
  satellites: SatellitePosition[];
  links: ConjunctionLink[];
  orbitPath: [number, number, number][];
  relatedOrbits: { norad_id: number; positions_km: [number, number, number][] }[];
  selectedId: number | null;
  onSelect: (id: number | null) => void;
  showAtRiskOnly: boolean;
}) {
  const visible = showAtRiskOnly
    ? satellites.filter((s) => s.risk_tier === 'High' || s.risk_tier === 'Medium')
    : satellites;

  const posById = useMemo(() => {
    const map = new Map<number, THREE.Vector3>();
    for (const sat of satellites) {
      const pos = eciToScene(sat.position_km);
      map.set(sat.norad_id, new THREE.Vector3(pos[0], pos[1], pos[2]));
    }
    return map;
  }, [satellites]);

  const velById = useMemo(() => {
    const map = new Map<number, THREE.Vector3>();
    for (const sat of satellites) {
      const vel = velocityToScene(sat.velocity_kms);
      const v = new THREE.Vector3(vel[0], vel[1], vel[2]).normalize();
      map.set(sat.norad_id, v);
    }
    return map;
  }, [satellites]);

  const orbitPoints = useMemo(() => {
    return orbitPath.map((pos) => {
      const p = eciToScene(pos);
      return new THREE.Vector3(p[0], p[1], p[2]);
    });
  }, [orbitPath]);

  const relatedOrbitPoints = useMemo(() => {
    return relatedOrbits.map((entry) => ({
      norad_id: entry.norad_id,
      points: entry.positions_km.map((pos) => {
        const p = eciToScene(pos);
        return new THREE.Vector3(p[0], p[1], p[2]);
      }),
    }));
  }, [relatedOrbits]);

  const selectedLinks = useMemo(() => {
    if (selectedId === null) return [];
    return links.filter(
      (link) =>
        link.defended_norad_id === selectedId || link.intruder_norad_id === selectedId
    );
  }, [links, selectedId]);

  const lineSegments = useMemo(() => {
    const segments: { key: string; curve: THREE.CatmullRomCurve3; color: string }[] = [];
    const radius = 1.0;
    const minHeight = 0.08;
    const maxHeight = 0.35;
    const steps = 24;

    for (const link of selectedLinks) {
      const a = posById.get(link.defended_norad_id);
      const b = posById.get(link.intruder_norad_id);
      if (!a || !b) continue;

      const aUnit = a.clone().normalize();
      const bUnit = b.clone().normalize();
      const dot = THREE.MathUtils.clamp(aUnit.dot(bUnit), -1, 1);
      const angle = Math.acos(dot);
      const sinAngle = Math.sin(angle);
      if (sinAngle < 1e-6) continue;

      const height = minHeight + (maxHeight - minHeight) * (angle / Math.PI);
      const points: THREE.Vector3[] = [];

      for (let i = 0; i <= steps; i++) {
        const t = i / steps;
        const s0 = Math.sin((1 - t) * angle) / sinAngle;
        const s1 = Math.sin(t * angle) / sinAngle;
        const dir = aUnit.clone().multiplyScalar(s0).add(bUnit.clone().multiplyScalar(s1));
        const scale = radius + height * Math.sin(Math.PI * t);
        const p = dir.multiplyScalar(scale);
        points.push(p);
      }

      segments.push({
        key: `${link.event_id}-${link.defended_norad_id}-${link.intruder_norad_id}`,
        curve: new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.5),
        color: riskLineColor(link.risk_tier),
      });
    }
    return segments;
  }, [selectedLinks, posById]);

  const arrowIds = useMemo(() => {
    const ids = new Set<number>();
    if (selectedId !== null) {
      ids.add(selectedId);
    }
    for (const link of selectedLinks) {
      ids.add(link.defended_norad_id);
      ids.add(link.intruder_norad_id);
    }
    return Array.from(ids);
  }, [selectedLinks, selectedId]);

  const relatedArrows = useMemo(() => {
    const arrows: { key: string; pos: THREE.Vector3; dir: THREE.Vector3 }[] = [];
    for (const entry of relatedOrbitPoints) {
      const pts = entry.points;
      if (pts.length < 2) continue;
      const indices = [Math.floor(pts.length * 0.33), Math.floor(pts.length * 0.66)];
      for (const idx of indices) {
        const i = Math.min(Math.max(idx, 0), pts.length - 2);
        const dir = pts[i + 1].clone().sub(pts[i]).normalize();
        arrows.push({
          key: `related-arrow-${entry.norad_id}-${i}`,
          pos: pts[i],
          dir,
        });
      }
    }
    return arrows;
  }, [relatedOrbitPoints]);

  return (
    <>
      <ambientLight intensity={0.3} />
      <directionalLight position={[5, 3, 5]} intensity={1.5} />
      <pointLight position={[-8, -4, -6]} intensity={0.3} color="#3a6aaa" />
      <Controls />
      <Earth />
      <GridLines />
      <Atmosphere />
      {lineSegments.map((seg) => (
        <mesh key={seg.key}>
          <tubeGeometry args={[seg.curve, 64, 0.006, 8, false]} />
          <meshBasicMaterial color={seg.color} transparent opacity={0.7} />
        </mesh>
      ))}
      {selectedId !== null && orbitPoints.length > 1 && (
        <line>
          <bufferGeometry>
            <bufferAttribute
              attach="attributes-position"
              count={orbitPoints.length}
              array={new Float32Array(orbitPoints.flatMap((p) => [p.x, p.y, p.z]))}
              itemSize={3}
            />
          </bufferGeometry>
          <lineBasicMaterial color="#7ad6ff" transparent opacity={0.6} />
        </line>
      )}
      {selectedId !== null &&
        relatedOrbitPoints.map((entry) => (
          <DashedOrbit key={`orbit-${entry.norad_id}`} points={entry.points} />
        ))}
      {selectedId !== null &&
        relatedArrows.map((arrow) => (
          <arrowHelper
            key={arrow.key}
            args={[arrow.dir, arrow.pos, 0.09, 0xb8e9ff, 0.03, 0.015]}
          />
        ))}
      {arrowIds.map((id) => {
        const pos = posById.get(id);
        const vel = velById.get(id);
        if (!pos || !vel) return null;
        return (
          <arrowHelper
            key={`arrow-${id}`}
            args={[vel, pos, 0.14, 0xffffff, 0.04, 0.02]}
          />
        );
      })}
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

export function GlobeView({ satellites, links, orbitPath, relatedOrbits, selectedId, onSelect, showAtRiskOnly }: {
  satellites: SatellitePosition[];
  links: ConjunctionLink[];
  orbitPath: [number, number, number][];
  relatedOrbits: { norad_id: number; positions_km: [number, number, number][] }[];
  selectedId: number | null;
  onSelect: (id: number | null) => void;
  showAtRiskOnly: boolean;
}) {
  return (
    <div style={{ position: 'absolute', inset: 0 }}>
      <Canvas camera={{ position: [0, 1.5, 3.8], fov: 45 }} gl={{ antialias: true }} onPointerMissed={() => onSelect(null)}>
        <Scene
          satellites={satellites}
          links={links}
          orbitPath={orbitPath}
          relatedOrbits={relatedOrbits}
          selectedId={selectedId}
          onSelect={onSelect}
          showAtRiskOnly={showAtRiskOnly}
        />
      </Canvas>
    </div>
  );
}
