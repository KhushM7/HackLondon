'use client';

import { useState } from 'react';
import { addSatellite, CatalogItem } from '../lib/api';

export function AddSatelliteModal({
  onClose,
  onAdded,
}: {
  onClose: () => void;
  onAdded: (sat: CatalogItem) => void;
}) {
  const [noradId, setNoradId] = useState('');
  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [message, setMessage] = useState('');

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    const id = Number(noradId);
    if (!id || id < 1) {
      setStatus('error');
      setMessage('Enter a valid positive NORAD catalog number.');
      return;
    }
    setStatus('loading');
    setMessage('');
    try {
      const sat = await addSatellite(id);
      setStatus('success');
      setMessage(`Added: ${sat.name} (NORAD ${sat.norad_id})`);
      setTimeout(() => {
        onAdded(sat);
        onClose();
      }, 1200);
    } catch (err: unknown) {
      setStatus('error');
      setMessage(err instanceof Error ? err.message : 'Failed to add satellite');
    }
  }

  return (
    <div
      style={{
        position: 'fixed',
        inset: 0,
        background: 'rgba(0, 0, 0, 0.65)',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'center',
        zIndex: 1000,
      }}
      onClick={onClose}
    >
      <div
        className="panel"
        style={{ width: 400, padding: '1.5rem' }}
        onClick={(e) => e.stopPropagation()}
      >
        <h3 style={{ margin: '0 0 1rem' }}>Add Satellite by NORAD ID</h3>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
          <div>
            <label style={{ display: 'block', marginBottom: 4, fontSize: '0.9rem' }}>
              NORAD Catalog Number
            </label>
            <input
              type="number"
              placeholder="e.g. 25544 (ISS)"
              value={noradId}
              onChange={(e) => setNoradId(e.target.value)}
              style={{ width: '100%' }}
              min={1}
              disabled={status === 'loading' || status === 'success'}
              autoFocus
            />
          </div>

          <p className="disclaimer" style={{ margin: 0 }}>
            TLE data is fetched from CelesTrak and stored locally. Only LEO satellites (altitude &lt; 2000 km) are supported.
          </p>

          {message && (
            <p
              style={{
                margin: 0,
                color: status === 'error' ? 'var(--warn)' : '#76e68e',
                fontSize: '0.9rem',
              }}
            >
              {message}
            </p>
          )}

          <div style={{ display: 'flex', gap: 8, marginTop: 4 }}>
            <button type="submit" disabled={status === 'loading' || status === 'success'}>
              {status === 'loading' ? 'Fetching TLE…' : status === 'success' ? 'Done ✓' : 'Add Satellite'}
            </button>
            <button type="button" onClick={onClose}>
              Cancel
            </button>
          </div>
        </form>
      </div>
    </div>
  );
}
