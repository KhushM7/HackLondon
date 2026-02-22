'use client';

import { useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import type { ConjunctionLink, SatellitePosition } from '../lib/api';

const EARTH_R_KM = 6371;
const S = 1 / EARTH_R_KM; // scene units: Earth radius = 1.0
const MAX_SINGLE_TRACK_POINTS = 140;

// ECI TEME → Three.js (Y-up, north pole = +Y)
// ECI: x = vernal equinox, y = 90°E equatorial, z = north pole
function eciToScene(pos: [number, number, number]): [number, number, number] {
  return [pos[0] * S, pos[2] * S, pos[1] * S];
}

function llaToScene(latLonAlt: [number, number, number]): [number, number, number] {
  const [latDeg, lonDeg, altKm] = latLonAlt;
  const lat = (latDeg * Math.PI) / 180;
  const lon = (lonDeg * Math.PI) / 180;
  const r = (EARTH_R_KM + altKm) * S;
  const x = r * Math.cos(lat) * Math.cos(lon);
  const y = r * Math.cos(lat) * Math.sin(lon);
  const z = r * Math.sin(lat);
  return [x, z, y];
}

// Approximate sun direction in ECI (J2000-based) for real-time lighting.
function sunDirectionEci(date: Date): THREE.Vector3 {
  const dayMs = 1000 * 60 * 60 * 24;
  const j2000 = Date.UTC(2000, 0, 1, 12, 0, 0);
  const d = (date.getTime() - j2000) / dayMs;
  const toRad = Math.PI / 180;
  const g = (357.529 + 0.98560028 * d) * toRad; // mean anomaly
  const q = (280.459 + 0.98564736 * d) * toRad; // mean longitude
  const L = q + (1.915 * Math.sin(g) + 0.020 * Math.sin(2 * g)) * toRad; // ecliptic longitude
  const e = (23.439 - 0.00000036 * d) * toRad; // obliquity
  const x = Math.cos(L);
  const y = Math.cos(e) * Math.sin(L);
  const z = Math.sin(e) * Math.sin(L);
  return new THREE.Vector3(x, y, z).normalize();
}

function riskColor(tier: string | null | undefined): string {
  if (tier === 'High') return '#ff6b6b';
  if (tier === 'Medium') return '#ffd56b';
  if (tier === 'Low') return '#76e68e';
  return '#3a6a9a';
}

function velocityToScene(vec: [number, number, number]): [number, number, number] {
  return [vec[0], vec[2], vec[1]];
}

type AvoidancePathSample =
  | { t: string; position_km: [number, number, number] }
  | { t: string; lat_lon_alt: [number, number, number] };

function avoidanceSampleToScene(sample: AvoidancePathSample): [number, number, number] {
  if ('position_km' in sample) {
    return eciToScene(sample.position_km);
  }
  return llaToScene(sample.lat_lon_alt);
}

function orderedAvoidancePoints(path: AvoidancePathSample[]): THREE.Vector3[] {
  const rows = path
    .map((sample) => {
      const tMs = Date.parse(sample.t);
      const p = avoidanceSampleToScene(sample);
      return { tMs: Number.isFinite(tMs) ? tMs : Number.POSITIVE_INFINITY, p };
    })
    .filter(({ p }) => Number.isFinite(p[0]) && Number.isFinite(p[1]) && Number.isFinite(p[2]))
    .sort((a, b) => a.tMs - b.tMs);

  const points: THREE.Vector3[] = [];
  let lastTime = Number.NaN;
  for (const row of rows) {
    if (row.tMs === lastTime && points.length > 0) {
      points[points.length - 1] = new THREE.Vector3(row.p[0], row.p[1], row.p[2]);
    } else {
      points.push(new THREE.Vector3(row.p[0], row.p[1], row.p[2]));
      lastTime = row.tMs;
    }
  }
  if (points.length <= MAX_SINGLE_TRACK_POINTS) {
    return points;
  }
  return points.slice(0, MAX_SINGLE_TRACK_POINTS);
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

function OrbitTube({ points, color = '#7ad6ff', opacity = 0.6, radius = 0.01 }: { points: THREE.Vector3[]; color?: string; opacity?: number; radius?: number }) {
  const geometry = useMemo(() => {
    if (points.length < 2) return null;
    const curve = new THREE.CatmullRomCurve3(points, false, 'catmullrom', 0.5);
    const geom = new THREE.TubeGeometry(curve, 128, radius, 8, false);
    geom.computeBoundingSphere();
    return geom;
  }, [points, radius]);

  useEffect(() => {
    return () => {
      geometry?.dispose();
    };
  }, [geometry]);

  if (!geometry) return null;

  return (
    <mesh geometry={geometry} frustumCulled={false}>
      <meshBasicMaterial color={color} transparent opacity={opacity} />
    </mesh>
  );
}

function OrbitLine({ points, color = '#7ad6ff', opacity = 0.7 }: { points: THREE.Vector3[]; color?: string; opacity?: number }) {
  const geometry = useMemo(() => {
    if (points.length < 2) return null;
    const coords = new Float32Array(points.length * 3);
    for (let i = 0; i < points.length; i += 1) {
      coords[i * 3] = points[i].x;
      coords[i * 3 + 1] = points[i].y;
      coords[i * 3 + 2] = points[i].z;
    }
    const geom = new THREE.BufferGeometry();
    geom.setAttribute('position', new THREE.BufferAttribute(coords, 3));
    geom.computeBoundingSphere();
    return geom;
  }, [points]);

  useEffect(() => {
    return () => {
      geometry?.dispose();
    };
  }, [geometry]);

  if (!geometry) return null;

  return (
    <line geometry={geometry} frustumCulled={false}>
      <lineBasicMaterial color={color} transparent opacity={opacity} />
    </line>
  );
}

function OrbitTubeLinear({
  points,
  color = '#7ad6ff',
  opacity = 0.9,
  radius = 0.007,
}: {
  points: THREE.Vector3[];
  color?: string;
  opacity?: number;
  radius?: number;
}) {
  const geometry = useMemo(() => {
    if (points.length < 2) return null;
    const path = new THREE.CurvePath<THREE.Vector3>();
    for (let i = 0; i < points.length - 1; i += 1) {
      path.add(new THREE.LineCurve3(points[i], points[i + 1]));
    }
    const tubularSegments = Math.max(32, points.length * 2);
    const geom = new THREE.TubeGeometry(path, tubularSegments, radius, 10, false);
    geom.computeBoundingSphere();
    return geom;
  }, [points, radius]);

  useEffect(() => {
    return () => {
      geometry?.dispose();
    };
  }, [geometry]);

  if (!geometry) return null;

  return (
    <mesh geometry={geometry} frustumCulled={false}>
      <meshBasicMaterial color={color} transparent opacity={opacity} />
    </mesh>
  );
}

function SatDot({
  sat,
  position,
  selected,
  onClick,
}: {
  sat: SatellitePosition;
  position: [number, number, number];
  selected: boolean;
  onClick: () => void;
}) {
  const color = selected ? '#ffffff' : riskColor(sat.risk_tier);
  const size = selected ? 0.028 : sat.risk_tier ? 0.018 : 0.01;

  return (
    <mesh
      position={position}
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
  avoidanceCurrentPath,
  avoidanceDeviatedPath,
  selectedId,
  onSelect,
  showAtRiskOnly,
}: {
  satellites: SatellitePosition[];
  links: ConjunctionLink[];
  orbitPath: [number, number, number][];
  relatedOrbits: { norad_id: number; positions_km: [number, number, number][] }[];
  avoidanceCurrentPath: AvoidancePathSample[];
  avoidanceDeviatedPath: AvoidancePathSample[];
  selectedId: number | null;
  onSelect: (id: number | null) => void;
  showAtRiskOnly: boolean;
}) {
  const sunLightRef = useRef<THREE.DirectionalLight>(null);

  useFrame(() => {
    const light = sunLightRef.current;
    if (!light) return;
    const sunEci = sunDirectionEci(new Date());
    const [sx, sy, sz] = eciToScene([sunEci.x, sunEci.y, sunEci.z]);
    light.position.set(sx * 12, sy * 12, sz * 12);
  });

  const visible = showAtRiskOnly
    ? satellites.filter((s) => s.risk_tier === 'High' || s.risk_tier === 'Medium')
    : satellites;

  const displayPosById = useMemo(() => {
    const cellSize = 0.002;
    const ringRadius = 0.028;
    const normalLift = 0.004;
    const groups = new Map<string, { sat: SatellitePosition; base: THREE.Vector3 }[]>();

    for (const sat of visible) {
      const p = eciToScene(sat.position_km);
      const base = new THREE.Vector3(p[0], p[1], p[2]);
      const key = [
        Math.round(base.x / cellSize),
        Math.round(base.y / cellSize),
        Math.round(base.z / cellSize),
      ].join('|');
      const group = groups.get(key);
      if (group) {
        group.push({ sat, base });
      } else {
        groups.set(key, [{ sat, base }]);
      }
    }

    const result = new Map<number, [number, number, number]>();
    for (const group of groups.values()) {
      if (group.length === 1) {
        const p = group[0].base;
        result.set(group[0].sat.norad_id, [p.x, p.y, p.z]);
        continue;
      }

      const normal = group[0].base.clone().normalize();
      const refAxis = Math.abs(normal.y) > 0.9
        ? new THREE.Vector3(1, 0, 0)
        : new THREE.Vector3(0, 1, 0);
      const tangentA = new THREE.Vector3().crossVectors(normal, refAxis).normalize();
      const tangentB = new THREE.Vector3().crossVectors(normal, tangentA).normalize();

      group.forEach((entry, index) => {
        const angle = (2 * Math.PI * index) / group.length;
        const offset = tangentA
          .clone()
          .multiplyScalar(Math.cos(angle) * ringRadius)
          .add(tangentB.clone().multiplyScalar(Math.sin(angle) * ringRadius))
          .add(normal.clone().multiplyScalar(normalLift));
        const p = entry.base.clone().add(offset);
        result.set(entry.sat.norad_id, [p.x, p.y, p.z]);
      });
    }
    return result;
  }, [visible]);

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

  const avoidanceCurrentPoints = useMemo(() => {
    return orderedAvoidancePoints(avoidanceCurrentPath);
  }, [avoidanceCurrentPath]);

  const avoidanceDeviatedPoints = useMemo(() => {
    return orderedAvoidancePoints(avoidanceDeviatedPath);
  }, [avoidanceDeviatedPath]);
  const selectedTrack = useMemo(() => {
    if (selectedId === null) {
      return { points: [] as THREE.Vector3[], color: '#7ad6ff', opacity: 0.7 };
    }
    if (avoidanceDeviatedPoints.length > 1) {
      return { points: avoidanceDeviatedPoints, color: '#66bb6a', opacity: 0.9 };
    }
    if (avoidanceCurrentPoints.length > 1) {
      return { points: avoidanceCurrentPoints, color: '#4fc3f7', opacity: 0.75 };
    }
    if (orbitPoints.length > 1) {
      return { points: orbitPoints, color: '#7ad6ff', opacity: 0.7 };
    }
    return { points: [] as THREE.Vector3[], color: '#7ad6ff', opacity: 0.7 };
  }, [selectedId, avoidanceDeviatedPoints, avoidanceCurrentPoints, orbitPoints]);

  const selectedLinks = useMemo(() => {
    if (selectedId === null) return [];
    return links.filter(
      (link) =>
        link.defended_norad_id === selectedId || link.intruder_norad_id === selectedId
    );
  }, [links, selectedId]);

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
      <directionalLight ref={sunLightRef} position={[5, 3, 5]} intensity={1.5} />
      <pointLight position={[-8, -4, -6]} intensity={0.3} color="#3a6aaa" />
      <Controls />
      <Earth />
      <GridLines />
      <Atmosphere />
      {selectedId !== null && selectedTrack.points.length > 1 && (
        <OrbitTubeLinear points={selectedTrack.points} color={selectedTrack.color} opacity={selectedTrack.opacity} radius={0.008} />
      )}
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
          position={displayPosById.get(sat.norad_id) ?? eciToScene(sat.position_km)}
          selected={selectedId === sat.norad_id}
          onClick={() => onSelect(selectedId === sat.norad_id ? null : sat.norad_id)}
        />
      ))}
    </>
  );
}

export function GlobeView({ satellites, links, orbitPath, relatedOrbits, avoidanceCurrentPath, avoidanceDeviatedPath, selectedId, onSelect, showAtRiskOnly }: {
  satellites: SatellitePosition[];
  links: ConjunctionLink[];
  orbitPath: [number, number, number][];
  relatedOrbits: { norad_id: number; positions_km: [number, number, number][] }[];
  avoidanceCurrentPath: AvoidancePathSample[];
  avoidanceDeviatedPath: AvoidancePathSample[];
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
          avoidanceCurrentPath={avoidanceCurrentPath}
          avoidanceDeviatedPath={avoidanceDeviatedPath}
          selectedId={selectedId}
          onSelect={onSelect}
          showAtRiskOnly={showAtRiskOnly}
        />
      </Canvas>
    </div>
  );
}
