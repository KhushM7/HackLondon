'use client';

import Link from 'next/link';
import { useEffect, useState } from 'react';

import { Conjunction, getConjunctions } from '../../../lib/api';

export default function AssetView({ params }: { params: { noradId: string } }) {
  const noradId = Number(params.noradId);
  const [events, setEvents] = useState<Conjunction[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

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
                  <td><Link href={`/conjunctions/${event.id}`}>#{event.id}</Link></td>
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
