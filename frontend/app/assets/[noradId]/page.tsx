'use client';

import Link from 'next/link';
import { useEffect, useState, useCallback } from 'react';

import { AvoidancePlan, Conjunction, getConjunctions, optimizeAvoidance, getAvoidancePlan } from '../../../lib/api';

export default function AssetView({ params }: { params: { noradId: string } }) {
  const noradId = Number(params.noradId);
  const [events, setEvents] = useState<Conjunction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [plan, setPlan] = useState<AvoidancePlan | null>(null);
  const [planLoading, setPlanLoading] = useState(false);
  const [planError, setPlanError] = useState<string | null>(null);

  useEffect(() => {
    getConjunctions(noradId)
      .then((data) => {
        setEvents(data);
        setError(null);
      })
      .catch((err: unknown) => {
        setError(err instanceof Error ? err.message : 'Failed to load conjunctions');
      })
      .finally(() => setLoading(false));

    // Try loading existing plan
    getAvoidancePlan(noradId)
      .then(setPlan)
      .catch(() => {}); // no plan yet is fine
  }, [noradId]);

  // Poll for plan status while running
  useEffect(() => {
    if (!plan || (plan.status !== 'pending' && plan.status !== 'running')) return;
    const interval = setInterval(() => {
      getAvoidancePlan(noradId).then((p) => {
        setPlan(p);
        if (p.status === 'completed' || p.status === 'failed') clearInterval(interval);
      }).catch(() => {});
    }, 2000);
    return () => clearInterval(interval);
  }, [plan?.status, noradId]);

  const startOptimization = useCallback(async () => {
    setPlanLoading(true);
    setPlanError(null);
    try {
      const p = await optimizeAvoidance(noradId);
      setPlan(p);
    } catch (err) {
      setPlanError(err instanceof Error ? err.message : 'Failed to start optimization');
    } finally {
      setPlanLoading(false);
    }
  }, [noradId]);

  return (
    <div className="grid" style={{ gap: 12 }}>
      <div className="panel">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
          <div>
            <h2 style={{ margin: 0 }}>Asset View: NORAD {noradId}</h2>
            <p className="disclaimer" style={{ margin: '4px 0 0' }}>
              Upcoming conjunctions within a 3-day screening window. Screening may take a moment.
            </p>
          </div>
          <Link href="/">← Back to Dashboard</Link>
        </div>
        {error && <p style={{ color: 'var(--warn)', marginTop: 8 }}>{error}</p>}
      </div>

      {/* Avoidance Recommendation Card */}
      <div className="panel">
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: 8 }}>
          <h3 style={{ margin: 0 }}>Collision Avoidance Plan</h3>
          <button onClick={startOptimization} disabled={planLoading || plan?.status === 'running'}>
            {planLoading || plan?.status === 'running' ? 'Computing avoidance plan…' : plan?.status === 'completed' ? 'Recompute Plan' : 'Optimize Avoidance'}
          </button>
        </div>
        {planError && <p style={{ color: 'var(--warn)', marginTop: 8 }}>{planError}</p>}

        {plan?.status === 'pending' || plan?.status === 'running' ? (
          <p className="disclaimer" style={{ marginTop: 8 }}>
            ⏳ Computing avoidance plan… This runs in the background and won&apos;t block the page.
          </p>
        ) : null}

        {plan?.status === 'failed' ? (
          <p style={{ color: '#ffb36d', marginTop: 8 }}>
            Optimization failed: {plan.error_message || 'Unknown error'}
          </p>
        ) : null}

        {plan?.status === 'completed' ? (
          <div style={{ marginTop: 10, display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(180px, 1fr))', gap: 12 }}>
            <div>
              <div className="disclaimer">Burn Direction</div>
              <div style={{ fontWeight: 600 }}>{plan.burn_direction?.replace('_', ' ') ?? '—'}</div>
            </div>
            <div>
              <div className="disclaimer">Burn Epoch (UTC)</div>
              <div style={{ fontWeight: 600 }}>{plan.burn_epoch ? new Date(plan.burn_epoch).toISOString().slice(0, 16).replace('T', ' ') : '—'}</div>
            </div>
            <div>
              <div className="disclaimer">Delta-v</div>
              <div style={{ fontWeight: 600 }}>{plan.burn_dv_mps?.toFixed(3) ?? '—'} m/s</div>
            </div>
            <div>
              <div className="disclaimer">RTN Vector</div>
              <div style={{ fontWeight: 600 }}>[{plan.burn_rtn_vector?.map(v => v.toFixed(3)).join(', ') ?? '—'}]</div>
            </div>
            <div>
              <div className="disclaimer">Miss Distance</div>
              <div style={{ fontWeight: 600 }}>
                {plan.pre_miss_distance_km?.toFixed(3) ?? '?'} → {plan.post_miss_distance_km?.toFixed(3) ?? '?'} km
                {plan.post_miss_distance_km != null && plan.pre_miss_distance_km != null && (
                  <span style={{ color: plan.post_miss_distance_km > plan.pre_miss_distance_km ? '#66bb6a' : '#ef5350', marginLeft: 6 }}>
                    ({plan.post_miss_distance_km > plan.pre_miss_distance_km ? '+' : ''}{(plan.post_miss_distance_km - plan.pre_miss_distance_km).toFixed(3)} km)
                  </span>
                )}
              </div>
            </div>
            <div>
              <div className="disclaimer">Pc Reduction</div>
              <div style={{ fontWeight: 600 }}>
                {typeof plan.pre_pc === 'number' ? plan.pre_pc.toExponential(2) : '?'} → {typeof plan.post_pc === 'number' ? plan.post_pc.toExponential(2) : '?'}
              </div>
            </div>
            <div>
              <div className="disclaimer">Fuel Cost</div>
              <div style={{ fontWeight: 600 }}>{plan.fuel_cost_kg?.toFixed(3) ?? '—'} kg</div>
            </div>
            <div>
              <div className="disclaimer">Compute Time</div>
              <div style={{ fontWeight: 600 }}>{plan.optimization_elapsed_s?.toFixed(1) ?? '—'} s ({plan.candidates_evaluated ?? '?'} candidates)</div>
            </div>
          </div>
        ) : null}
      </div>

      <div className="panel">
        {loading ? (
          <p className="disclaimer" style={{ padding: '0.5rem 0' }}>Screening for conjunctions — this may take a moment…</p>
        ) : events.length === 0 && !error ? (
          <p className="disclaimer">No conjunctions found within the 3-day window for this asset.</p>
        ) : (
          <table>
            <thead>
              <tr>
                <th>Event</th>
                <th>Intruder</th>
                <th>TCA (UTC)</th>
                <th>Miss Distance (km)</th>
                <th>Rel. Velocity (km/s)</th>
                <th>Risk</th>
                <th>Pc</th>
              </tr>
            </thead>
            <tbody>
              {events.map((event) => (
                <tr key={event.id}>
                  <td>#{event.id}</td>
                  <td>{event.intruder_name || `NORAD ${event.intruder_norad_id}`}</td>
                  <td>{new Date(event.tca_utc).toISOString().slice(0, 16).replace('T', ' ')}</td>
                  <td>{event.miss_distance_km.toFixed(3)}</td>
                  <td>{event.relative_velocity_kms.toFixed(3)}</td>
                  <td><span className={`badge ${event.risk_tier.toLowerCase()}`}>{event.risk_tier}</span></td>
                  <td>{typeof event.pc_foster === 'number' ? event.pc_foster.toExponential(2) : '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </div>
  );
}
