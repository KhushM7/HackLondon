'use client';

import { useState } from 'react';

import { ConjunctionDetail, getConjunction, runAvoidance } from '../lib/api';
import { ConjunctionScene } from './ConjunctionScene';

export function EventReplayClient({ initialEvent }: { initialEvent: ConjunctionDetail }) {
  const [event, setEvent] = useState<ConjunctionDetail>(initialEvent);
  const [deltaV, setDeltaV] = useState(0.2);
  const [loading, setLoading] = useState(false);
  const [showPost, setShowPost] = useState(false);
  const [simStats, setSimStats] = useState<{ updated_miss_distance_km: number; delta_km: number } | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function onSimulate() {
    try {
      setLoading(true);
      const result = await runAvoidance(event.id, deltaV);
      const refreshed = await getConjunction(event.id);
      setEvent(refreshed);
      setSimStats(result);
      setShowPost(true);
      setError(null);
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Avoidance simulation failed.');
    } finally {
      setLoading(false);
    }
  }

  const defendedPath = showPost && event.post_trajectory ? event.post_trajectory : event.pre_trajectory;

  return (
    <div className="grid" style={{ gap: 12 }}>
      <ConjunctionScene defended={defendedPath} intruder={event.intruder_trajectory} />

      <div className="panel">
        <h3>Avoidance Simulation</h3>
        <p className="disclaimer">Simplified and non-authoritative along-track delta-v model.</p>
        <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
          <label>
            Delta-v (m/s)
            <input
              type="number"
              value={deltaV}
              min={0.01}
              max={5}
              step={0.01}
              onChange={(e) => setDeltaV(Number(e.target.value))}
              style={{ marginLeft: 8 }}
            />
          </label>
          <button onClick={onSimulate} disabled={loading}>{loading ? 'Running...' : 'Simulate Avoidance'}</button>
          <label>
            <input
              type="checkbox"
              checked={showPost}
              onChange={(e) => setShowPost(e.target.checked)}
              disabled={!event.post_trajectory}
              style={{ marginRight: 6 }}
            />
            Show post-manoeuvre path
          </label>
        </div>
        {simStats ? (
          <div style={{ marginTop: 10 }}>
            <div>Updated miss distance: {simStats.updated_miss_distance_km.toFixed(3)} km</div>
            <div>Change: {simStats.delta_km >= 0 ? '+' : ''}{simStats.delta_km.toFixed(3)} km</div>
          </div>
        ) : null}
        {error ? <p style={{ color: '#ffb36d', marginTop: 10 }}>{error}</p> : null}
      </div>
    </div>
  );
}
