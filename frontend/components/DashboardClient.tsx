'use client';

import Link from 'next/link';
import { useEffect, useMemo, useRef, useState, type ReactNode } from 'react';

import {
  CatalogItem,
  Conjunction,
  ConjunctionLink,
  SatellitePosition,
  getCatalogPositions,
  getCongestion,
  getConjunctions,
  getOrbitPath,
  refreshIngest,
} from '../lib/api';
import { AddSatelliteModal } from './AddSatelliteModal';
import { CongestionChart } from './CongestionChart';
import { GlobeView } from './GlobeView';

// ── Draggable floating window ─────────────────────────────────────────────────

interface DraggableWindowProps {
  title: string;
  children: ReactNode;
  onClose: () => void;
  width: number;
  initialX: number;
  initialY: number;
}

function DraggableWindow({ title, children, onClose, width, initialX, initialY }: DraggableWindowProps) {
  const winRef = useRef<HTMLDivElement>(null);
  const s = useRef({ x: initialX, y: initialY, drag: false, ox: 0, oy: 0 });

  useEffect(() => {
    function onMove(e: MouseEvent) {
      if (!s.current.drag) return;
      s.current.x = e.clientX - s.current.ox;
      s.current.y = e.clientY - s.current.oy;
      if (winRef.current) {
        winRef.current.style.left = s.current.x + 'px';
        winRef.current.style.top = s.current.y + 'px';
      }
    }
    function onUp() {
      s.current.drag = false;
    }
    document.addEventListener('mousemove', onMove);
    document.addEventListener('mouseup', onUp);
    return () => {
      document.removeEventListener('mousemove', onMove);
      document.removeEventListener('mouseup', onUp);
    };
  }, []);

  function onMouseDown(e: React.MouseEvent) {
    s.current.drag = true;
    s.current.ox = e.clientX - s.current.x;
    s.current.oy = e.clientY - s.current.y;
    e.preventDefault();
  }

  return (
    <div
      ref={winRef}
      style={{
        position: 'fixed',
        left: initialX,
        top: initialY,
        width,
        zIndex: 60,
        background: 'rgba(4,10,22,0.92)',
        border: '1px solid var(--border)',
        borderRadius: 10,
        backdropFilter: 'blur(20px)',
        WebkitBackdropFilter: 'blur(20px)',
        boxShadow: '0 8px 32px rgba(0,0,0,0.6)',
      }}
    >
      {/* Title bar / drag handle */}
      <div
        onMouseDown={onMouseDown}
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
          padding: '10px 14px',
          borderBottom: '1px solid var(--border)',
          cursor: 'grab',
          userSelect: 'none',
        }}
      >
        <span style={{ color: 'var(--accent)', fontSize: '0.75rem', fontWeight: 700, letterSpacing: '0.1em', textTransform: 'uppercase' }}>
          {title}
        </span>
        <button
          onMouseDown={(e) => e.stopPropagation()}
          onClick={onClose}
          style={{
            background: 'transparent',
            border: 'none',
            color: 'var(--ink-muted)',
            cursor: 'pointer',
            fontSize: '1.1rem',
            lineHeight: 1,
            padding: 0,
            boxShadow: 'none',
          }}
        >
          ×
        </button>
      </div>

      {/* Content */}
      <div style={{ padding: '1rem', maxHeight: '70vh', overflowY: 'auto' }}>
        {children}
      </div>
    </div>
  );
}

// ── Dashboard ─────────────────────────────────────────────────────────────────

export function DashboardClient() {
  const [satellites, setSatellites] = useState<SatellitePosition[]>([]);
  const [links, setLinks] = useState<ConjunctionLink[]>([]);
  const [orbitPath, setOrbitPath] = useState<[number, number, number][]>([]);
  const [relatedOrbits, setRelatedOrbits] = useState<{ norad_id: number; positions_km: [number, number, number][] }[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [events, setEvents] = useState<Conjunction[]>([]);
  const [bands, setBands] = useState<{ altitude_band_km: string; object_count: number; conjunction_rate: number }[]>([]);
  const [showAtRiskOnly, setShowAtRiskOnly] = useState(false);
  const [showAddModal, setShowAddModal] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [showMenu, setShowMenu] = useState(false);
  const [showConjFloat, setShowConjFloat] = useState(false);
  const [showCongFloat, setShowCongFloat] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [refreshMessage, setRefreshMessage] = useState<string | null>(null);
  const refreshTimerRef = useRef<number | null>(null);

  const nameById = useMemo(() => new Map(satellites.map((s) => [s.norad_id, s.name])), [satellites]);
  const tierRank = useMemo(() => new Map([['High', 3], ['Medium', 2], ['Low', 1]]), []);

  const getOtherName = (ev: Conjunction) => {
    if (selectedId === ev.defended_norad_id) {
      return ev.intruder_name || nameById.get(ev.intruder_norad_id) || `NORAD ${ev.intruder_norad_id}`;
    }
    if (selectedId === ev.intruder_norad_id) {
      return ev.defended_name || nameById.get(ev.defended_norad_id) || `NORAD ${ev.defended_norad_id}`;
    }
    return ev.intruder_name || nameById.get(ev.intruder_norad_id) || `NORAD ${ev.intruder_norad_id}`;
  };

  const topRiskLink = useMemo(() => {
    if (selectedId === null) return null;
    const matches = links.filter(
      (link) =>
        link.defended_norad_id === selectedId || link.intruder_norad_id === selectedId
    );
    if (matches.length === 0) return null;
    return [...matches].sort((a, b) => {
      const tierDelta = (tierRank.get(b.risk_tier) || 0) - (tierRank.get(a.risk_tier) || 0);
      if (tierDelta !== 0) return tierDelta;
      return a.miss_distance_km - b.miss_distance_km;
    })[0];
  }, [links, selectedId, tierRank]);

  const getOtherNameFromLink = (link: ConjunctionLink) => {
    const otherId = link.defended_norad_id === selectedId ? link.intruder_norad_id : link.defended_norad_id;
    return nameById.get(otherId) || `NORAD ${otherId}`;
  };

  useEffect(() => {
    async function load() {
      try {
        const [positions, congestion] = await Promise.all([
          getCatalogPositions(500),
          getCongestion(),
        ]);
        setSatellites(positions.satellites);
        setLinks(positions.links);
        setBands(congestion.bands);
        setError(null);
      } catch {
        setError('API unreachable. Start backend on http://localhost:8000 and refresh.');
      } finally {
        setLoading(false);
      }
    }
    load();
  }, []);

  useEffect(() => {
    return () => {
      if (refreshTimerRef.current) {
        window.clearTimeout(refreshTimerRef.current);
        refreshTimerRef.current = null;
      }
    };
  }, []);

  useEffect(() => {
    if (selectedId !== null) {
      getConjunctions(selectedId).then(setEvents).catch(() => setEvents([]));
      getOrbitPath(selectedId)
        .then((data) => setOrbitPath(data.positions_km))
        .catch(() => setOrbitPath([]));
      const relatedIds = links
        .filter((link) => link.risk_tier === 'High' || link.risk_tier === 'Medium')
        .filter((link) => link.defended_norad_id === selectedId || link.intruder_norad_id === selectedId)
        .flatMap((link) => [link.defended_norad_id, link.intruder_norad_id])
        .filter((id) => id !== selectedId);
      if (relatedIds.length > 0) {
        const uniqueIds = Array.from(new Set(relatedIds));
        Promise.all(uniqueIds.map((id) => getOrbitPath(id)))
          .then((items) => setRelatedOrbits(items))
          .catch(() => setRelatedOrbits([]));
      } else {
        setRelatedOrbits([]);
      }
      setShowConjFloat(true);
    } else {
      setEvents([]);
      setOrbitPath([]);
      setRelatedOrbits([]);
    }
  }, [selectedId, links]);

  async function triggerRefresh() {
    try {
      if (refreshTimerRef.current) {
        window.clearTimeout(refreshTimerRef.current);
        refreshTimerRef.current = null;
      }
      setRefreshing(true);
      setRefreshMessage('Refreshing…');
      await refreshIngest();
      const [positions, congestion] = await Promise.all([getCatalogPositions(500), getCongestion()]);
      setSatellites(positions.satellites);
      setLinks(positions.links);
      setBands(congestion.bands);
      setError(null);
      setRefreshMessage('Refresh complete');
      refreshTimerRef.current = window.setTimeout(() => setRefreshMessage(null), 2000);
    } catch {
      setError('Manual refresh failed. Verify backend is running and reachable.');
      setRefreshMessage('Refresh failed');
      refreshTimerRef.current = window.setTimeout(() => setRefreshMessage(null), 3000);
    } finally {
      setRefreshing(false);
    }
  }

  async function handleSatelliteAdded(sat: CatalogItem) {
    try {
      const positions = await getCatalogPositions(500);
      setSatellites(positions.satellites);
      setLinks(positions.links);
      setSelectedId(sat.norad_id);
    } catch {
      // positions reload failed; globe will show on next full refresh
    }
  }

  const selectedSat = satellites.find((s) => s.norad_id === selectedId) ?? null;
  const atRiskCount = satellites.filter((s) => s.risk_tier === 'High' || s.risk_tier === 'Medium').length;

  return (
    <div style={{ position: 'fixed', inset: 0, background: '#000', overflow: 'hidden' }}>

      {/* ── Layer 1: Globe fills entire viewport ─────────────── */}
      <div style={{ position: 'absolute', inset: 0 }}>
        <GlobeView
          satellites={satellites}
          links={links}
          orbitPath={orbitPath}
          relatedOrbits={relatedOrbits}
          selectedId={selectedId}
          onSelect={setSelectedId}
          showAtRiskOnly={showAtRiskOnly}
        />
      </div>

      {/* ── Layer 2: Gradient vignettes (non-interactive) ─────── */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          height: 90,
          background: 'linear-gradient(to bottom, rgba(0,0,0,0.75) 0%, transparent 100%)',
          pointerEvents: 'none',
          zIndex: 5,
        }}
      />
      <div
        style={{
          position: 'absolute',
          bottom: 0,
          left: 0,
          right: 0,
          height: 100,
          background: 'linear-gradient(to top, rgba(0,0,0,0.55) 0%, transparent 100%)',
          pointerEvents: 'none',
          zIndex: 5,
        }}
      />

      {/* ── Layer 3: Top HUD bar ──────────────────────────────── */}
      <div
        style={{
          position: 'absolute',
          top: 0,
          left: 0,
          right: 0,
          zIndex: 20,
          display: 'flex',
          alignItems: 'center',
          gap: 10,
          padding: '14px 20px',
        }}
      >
        {/* Logo block */}
        <div style={{ display: 'flex', flexDirection: 'column', lineHeight: 1.2 }}>
          <span
            style={{
              color: 'var(--accent)',
              fontWeight: 700,
              fontSize: '1rem',
              letterSpacing: '0.2em',
              textTransform: 'uppercase',
            }}
          >
            ORBITGUARD
          </span>
          <span style={{ color: 'var(--ink-muted)', fontSize: '0.62rem', letterSpacing: '0.12em', textTransform: 'uppercase' }}>
            · &nbsp;LEO SCREENING
          </span>
        </div>

        {/* Spacer */}
        <div style={{ flex: 1 }} />

        {/* Stats chips */}
        <div
          style={{
            padding: '4px 10px',
            borderRadius: 20,
            fontSize: '0.7rem',
            letterSpacing: '0.06em',
            background: 'rgba(0,0,0,0.4)',
            border: '1px solid var(--border)',
            color: 'var(--ink)',
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}
        >
          <span style={{ color: 'var(--accent)' }}>●</span>
          {loading ? '…' : satellites.length} TRACKED
        </div>
        <div
          style={{
            padding: '4px 10px',
            borderRadius: 20,
            fontSize: '0.7rem',
            letterSpacing: '0.06em',
            background: 'rgba(0,0,0,0.4)',
            border: '1px solid var(--border)',
            color: atRiskCount > 0 ? 'var(--high)' : 'var(--ink-muted)',
            display: 'flex',
            alignItems: 'center',
            gap: 4,
          }}
        >
          <span>⚠</span>
          {loading ? '…' : atRiskCount} AT RISK
        </div>

        {/* Filter group */}
        <button
          className={!showAtRiskOnly ? 'btn-active' : undefined}
          onClick={() => setShowAtRiskOnly(false)}
        >
          ALL ({loading ? '…' : satellites.length})
        </button>
        <button
          className={showAtRiskOnly ? 'btn-active' : undefined}
          onClick={() => setShowAtRiskOnly(true)}
        >
          AT RISK ({loading ? '…' : atRiskCount})
        </button>

        {/* Add satellite CTA */}
        <button className="btn-accent" onClick={() => setShowAddModal(true)}>
          + SATELLITE
        </button>

        {/* Refresh */}
        <button onClick={triggerRefresh} disabled={refreshing}>
          {refreshing ? '↺ REFRESHING…' : '↺ REFRESH'}
        </button>
        {refreshMessage ? (
          <span className="hud-label" style={{ marginLeft: -6 }}>
            {refreshMessage}
          </span>
        ) : null}

        {/* Menu button + dropdown */}
        <div style={{ position: 'relative' }}>
          <button
            onClick={() => setShowMenu((v) => !v)}
            style={{ fontSize: '1rem', padding: '0.4rem 0.65rem' }}
          >
            ≡
          </button>

          {/* Transparent backdrop — closes menu on outside-click */}
          {showMenu && (
            <div
              style={{ position: 'fixed', inset: 0, zIndex: 48 }}
              onClick={() => setShowMenu(false)}
            />
          )}

          {/* Dropdown card */}
          {showMenu && (
            <div
              style={{
                position: 'absolute',
                top: 'calc(100% + 6px)',
                right: 0,
                zIndex: 50,
                background: 'rgba(4,10,22,0.97)',
                border: '1px solid var(--border)',
                borderRadius: 8,
                backdropFilter: 'blur(20px)',
                WebkitBackdropFilter: 'blur(20px)',
                boxShadow: '0 8px 24px rgba(0,0,0,0.5)',
                minWidth: 190,
                padding: '6px 0',
              }}
            >
              {/* Conjunctions item */}
              <button
                onClick={() => {
                  if (!showConjFloat) setShowConjFloat(true);
                  setShowMenu(false);
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  width: '100%',
                  padding: '9px 14px',
                  background: 'transparent',
                  border: 'none',
                  borderRadius: 0,
                  boxShadow: 'none',
                  textAlign: 'left',
                  fontSize: '0.8rem',
                  letterSpacing: '0.05em',
                  cursor: showConjFloat ? 'default' : 'pointer',
                  color: showConjFloat ? 'var(--ink-muted)' : 'var(--ink)',
                  opacity: showConjFloat ? 0.45 : 1,
                }}
              >
                <span>Conjunctions</span>
                {showConjFloat && (
                  <span style={{ fontSize: '0.62rem', letterSpacing: '0.1em', color: 'var(--accent)' }}>
                    OPEN
                  </span>
                )}
              </button>

              {/* Congestion Chart item */}
              <button
                onClick={() => {
                  if (!showCongFloat) setShowCongFloat(true);
                  setShowMenu(false);
                }}
                style={{
                  display: 'flex',
                  alignItems: 'center',
                  justifyContent: 'space-between',
                  width: '100%',
                  padding: '9px 14px',
                  background: 'transparent',
                  border: 'none',
                  borderRadius: 0,
                  boxShadow: 'none',
                  textAlign: 'left',
                  fontSize: '0.8rem',
                  letterSpacing: '0.05em',
                  cursor: showCongFloat ? 'default' : 'pointer',
                  color: showCongFloat ? 'var(--ink-muted)' : 'var(--ink)',
                  opacity: showCongFloat ? 0.45 : 1,
                }}
              >
                <span>Congestion Chart</span>
                {showCongFloat && (
                  <span style={{ fontSize: '0.62rem', letterSpacing: '0.1em', color: 'var(--accent)' }}>
                    OPEN
                  </span>
                )}
              </button>
            </div>
          )}
        </div>
      </div>

      {/* ── Layer 4: Bottom-left selected satellite card ───────── */}
      {selectedSat !== null && (
        <div
          style={{
            position: 'absolute',
            bottom: 24,
            left: 20,
            zIndex: 15,
            background: 'var(--bg-panel)',
            border: '1px solid var(--border)',
            borderRadius: 10,
            padding: '12px 14px',
            backdropFilter: 'blur(16px)',
            WebkitBackdropFilter: 'blur(16px)',
            minWidth: 200,
            maxWidth: 260,
          }}
        >
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 4 }}>
            <span style={{ color: 'var(--accent)', fontWeight: 600, fontSize: '0.9rem' }}>
              {selectedSat.name}
            </span>
            <button
              onClick={() => setSelectedId(null)}
              style={{
                background: 'transparent',
                border: 'none',
                color: 'var(--ink-muted)',
                cursor: 'pointer',
                padding: 0,
                fontSize: '1rem',
                lineHeight: 1,
                boxShadow: 'none',
              }}
            >
              ×
            </button>
          </div>
          <div className="hud-label" style={{ marginBottom: 6 }}>
            NORAD {selectedSat.norad_id}
          </div>
          {selectedSat.risk_tier ? (
            <span className={`badge ${selectedSat.risk_tier.toLowerCase()}`}>
              {selectedSat.risk_tier}
            </span>
          ) : null}
        </div>
      )}

      {/* ── Layer 5: Bottom-right legend ──────────────────────── */}
      <div
        style={{
          position: 'absolute',
          bottom: 24,
          right: 20,
          zIndex: 15,
          display: 'flex',
          gap: 12,
          fontSize: '0.68rem',
          letterSpacing: '0.07em',
          color: 'var(--ink-muted)',
          alignItems: 'center',
          background: 'rgba(0,0,0,0.4)',
          padding: '6px 12px',
          borderRadius: 20,
          border: '1px solid var(--border)',
        }}
      >
        <span><span style={{ color: '#ff4454' }}>●</span> HIGH</span>
        <span><span style={{ color: '#ffd32a' }}>●</span> MED</span>
        <span><span style={{ color: '#2ed573' }}>●</span> LOW</span>
        <span><span style={{ color: '#1a4060' }}>●</span> CLEAR</span>
        <span style={{ color: 'var(--border-bright)' }}>|</span>
        <span>DRAG · ZOOM · CLICK</span>
      </div>

      {/* ── Layer 6: Error toast ──────────────────────────────── */}
      {error && (
        <div
          style={{
            position: 'absolute',
            top: 68,
            left: '50%',
            transform: 'translateX(-50%)',
            background: 'rgba(255,60,60,0.1)',
            border: '1px solid rgba(255,60,60,0.35)',
            color: '#ff8080',
            padding: '8px 16px',
            borderRadius: 8,
            fontSize: '0.8rem',
            zIndex: 25,
            whiteSpace: 'nowrap',
          }}
        >
          {error}
        </div>
      )}

      {/* ── Layer 7: Add satellite modal ──────────────────────── */}
      {showAddModal && (
        <AddSatelliteModal
          onClose={() => setShowAddModal(false)}
          onAdded={handleSatelliteAdded}
        />
      )}

      {/* ── Floating: Conjunctions ────────────────────────────── */}
      {showConjFloat && (
        <DraggableWindow
          title="Conjunctions"
          onClose={() => setShowConjFloat(false)}
          width={440}
          initialX={80}
          initialY={80}
        >
          {selectedSat ? (
            <div>
              {/* Satellite mini-header */}
              <div style={{ marginBottom: 12 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 4 }}>
                  <span style={{ color: 'var(--accent)', fontWeight: 600, fontSize: '0.9rem' }}>
                    {selectedSat.name}
                  </span>
                  {selectedSat.risk_tier ? (
                    <span className={`badge ${selectedSat.risk_tier.toLowerCase()}`}>
                      {selectedSat.risk_tier}
                    </span>
                  ) : null}
                  {topRiskLink ? (
                    <div className="hud-label" style={{ marginTop: 6 }}>
                      Potential collision with {getOtherNameFromLink(topRiskLink)}
                    </div>
                  ) : null}
                </div>
                <span className="hud-label">NORAD {selectedSat.norad_id}</span>
              </div>

              {/* Conjunctions table */}
              <table>
                <thead>
                  <tr>
                    <th>Event</th>
                    <th>With</th>
                    <th>TCA (UTC)</th>
                    <th>Miss km</th>
                    <th>Risk</th>
                    <th>Pc</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((ev) => (
                    <tr key={ev.id}>
                      <td><Link href={`/conjunctions/${ev.id}`}>#{ev.id}</Link></td>
                      <td>
                        {getOtherName(ev)}
                      </td>
                      <td>{new Date(ev.tca_utc).toISOString().slice(0, 16).replace('T', ' ')}</td>
                      <td>{ev.miss_distance_km.toFixed(2)}</td>
                      <td>
                        <span className={`badge ${ev.risk_tier.toLowerCase()}`}>{ev.risk_tier}</span>
                      </td>
                      <td>{typeof ev.pc_foster === 'number' ? ev.pc_foster.toExponential(2) : '—'}</td>
                    </tr>
                  ))}
                  {events.length === 0 && (
                    <tr>
                      <td colSpan={6} className="disclaimer" style={{ padding: '0.75rem 0.5rem' }}>
                        No conjunctions recorded for this asset
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>

              {/* Link to full asset view */}
              <div style={{ marginTop: 14 }}>
                <Link href={`/assets/${selectedSat.norad_id}`}>
                  Full Asset View →
                </Link>
              </div>
            </div>
          ) : (
            <div className="hud-label" style={{ textAlign: 'center', marginTop: 40, marginBottom: 40 }}>
              SELECT A SATELLITE ON THE GLOBE
            </div>
          )}
        </DraggableWindow>
      )}

      {/* ── Floating: Congestion Chart ────────────────────────── */}
      {showCongFloat && (
        <DraggableWindow
          title="Altitude Congestion Index"
          onClose={() => setShowCongFloat(false)}
          width={700}
          initialX={120}
          initialY={100}
        >
          <CongestionChart bands={bands} />
        </DraggableWindow>
      )}
    </div>
  );
}
