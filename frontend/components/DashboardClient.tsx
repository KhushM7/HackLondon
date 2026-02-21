'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';

import {
  CatalogItem,
  Conjunction,
  SatellitePosition,
  getCatalogPositions,
  getCongestion,
  getConjunctions,
  refreshIngest,
} from '../lib/api';
import { AddSatelliteModal } from './AddSatelliteModal';
import { CongestionChart } from './CongestionChart';
import { GlobeView } from './GlobeView';

export function DashboardClient() {
  const [satellites, setSatellites] = useState<SatellitePosition[]>([]);
  const [selectedId, setSelectedId] = useState<number | null>(null);
  const [events, setEvents] = useState<Conjunction[]>([]);
  const [bands, setBands] = useState<{ altitude_band_km: string; object_count: number; conjunction_rate: number }[]>([]);
  const [showAtRiskOnly, setShowAtRiskOnly] = useState(false);
  const [showAddModal, setShowAddModal] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function load() {
      try {
        const [positions, congestion] = await Promise.all([
          getCatalogPositions(),
          getCongestion(),
        ]);
        setSatellites(positions);
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
    if (selectedId !== null) {
      getConjunctions(selectedId).then(setEvents).catch(() => setEvents([]));
    } else {
      setEvents([]);
    }
  }, [selectedId]);

  async function triggerRefresh() {
    try {
      await refreshIngest();
      const [positions, congestion] = await Promise.all([getCatalogPositions(), getCongestion()]);
      setSatellites(positions);
      setBands(congestion.bands);
      setError(null);
    } catch {
      setError('Manual refresh failed. Verify backend is running and reachable.');
    }
  }

  async function handleSatelliteAdded(sat: CatalogItem) {
    // Reload positions so the new satellite appears on the globe
    try {
      const positions = await getCatalogPositions();
      setSatellites(positions);
      // Auto-select the newly added satellite
      setSelectedId(sat.norad_id);
    } catch {
      // positions reload failed; globe will show on next full refresh
    }
  }

  const selectedSat = satellites.find((s) => s.norad_id === selectedId) ?? null;
  const atRiskCount = satellites.filter((s) => s.risk_tier === 'High' || s.risk_tier === 'Medium').length;

  return (
    <div className="grid" style={{ gap: '1rem' }}>
      {showAddModal && (
        <AddSatelliteModal
          onClose={() => setShowAddModal(false)}
          onAdded={handleSatelliteAdded}
        />
      )}
      <header className="panel">
        <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
          <div>
            <h1 style={{ margin: 0 }}>OrbitGuard</h1>
            <p style={{ margin: '4px 0 0' }}>LEO conjunction screening and 3D globe visualisation.</p>
            <p className="disclaimer" style={{ marginTop: 4 }}>
              Public TLE only · Screening-grade · No formal probability of collision.
            </p>
          </div>
          <div style={{ display: 'flex', gap: 8 }}>
            <button onClick={() => setShowAddModal(true)}>+ Add Satellite</button>
            <button onClick={triggerRefresh}>Refresh TLEs</button>
          </div>
        </div>
        {error ? <p style={{ color: '#ffb36d', margin: '8px 0 0' }}>{error}</p> : null}
      </header>

      {/* Filter bar */}
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
        <button
          style={!showAtRiskOnly ? { background: 'var(--accent)', color: '#08131f', border: 'none' } : {}}
          onClick={() => setShowAtRiskOnly(false)}
        >
          All satellites ({loading ? '…' : satellites.length})
        </button>
        <button
          style={showAtRiskOnly ? { background: '#ff6b6b', color: '#08131f', border: 'none' } : {}}
          onClick={() => setShowAtRiskOnly(true)}
        >
          At-risk only ({atRiskCount})
        </button>

        {selectedSat ? (
          <span style={{ marginLeft: 8, display: 'flex', alignItems: 'center', gap: 6 }}>
            <span style={{ color: 'var(--accent)', fontWeight: 600 }}>{selectedSat.name}</span>
            <span className="disclaimer">(NORAD {selectedSat.norad_id})</span>
            {selectedSat.risk_tier ? (
              <span className={`badge ${selectedSat.risk_tier.toLowerCase()}`}>{selectedSat.risk_tier}</span>
            ) : null}
            <button style={{ padding: '0.2rem 0.5rem' }} onClick={() => setSelectedId(null)}>×</button>
          </span>
        ) : (
          <span className="disclaimer" style={{ marginLeft: 8 }}>
            Click a satellite on the globe to select it
          </span>
        )}
      </div>

      {/* Main grid: globe + side panel */}
      <section className="grid grid-2" style={{ alignItems: 'start' }}>
        <GlobeView
          satellites={satellites}
          selectedId={selectedId}
          onSelect={setSelectedId}
          showAtRiskOnly={showAtRiskOnly}
        />

        <div className="grid" style={{ gap: '1rem' }}>
          {selectedSat ? (
            <div className="panel">
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: 8 }}>
                <h3 style={{ margin: 0 }}>{selectedSat.name}</h3>
                <Link href={`/assets/${selectedSat.norad_id}`}>Asset View →</Link>
              </div>
              <p className="disclaimer" style={{ margin: '0 0 10px' }}>NORAD {selectedSat.norad_id}</p>
              <table>
                <thead>
                  <tr>
                    <th>Event</th>
                    <th>TCA (UTC)</th>
                    <th>Miss (km)</th>
                    <th>Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {events.map((ev) => (
                    <tr key={ev.id}>
                      <td><Link href={`/conjunctions/${ev.id}`}>#{ev.id}</Link></td>
                      <td>{new Date(ev.tca_utc).toISOString().slice(0, 16).replace('T', ' ')}</td>
                      <td>{ev.miss_distance_km.toFixed(2)}</td>
                      <td><span className={`badge ${ev.risk_tier.toLowerCase()}`}>{ev.risk_tier}</span></td>
                    </tr>
                  ))}
                  {events.length === 0 && (
                    <tr>
                      <td colSpan={4} className="disclaimer" style={{ padding: '0.75rem 0.5rem' }}>
                        No conjunctions recorded for this asset
                      </td>
                    </tr>
                  )}
                </tbody>
              </table>
            </div>
          ) : null}

          <CongestionChart bands={bands} />
        </div>
      </section>
    </div>
  );
}
