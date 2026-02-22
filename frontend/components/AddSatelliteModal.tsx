'use client';

import { useMemo, useState } from 'react';
import { addCustomSatellite, addSatellite, CatalogItem, CustomSatelliteAddProgress } from '../lib/api';

type Mode = 'norad' | 'custom';

export function AddSatelliteModal({
  onClose,
  onAdded,
}: {
  onClose: () => void;
  onAdded: (sat: CatalogItem) => void;
}) {
  const [mode, setMode] = useState<Mode>('norad');

  const [noradId, setNoradId] = useState('');
  const [name, setName] = useState('');
  const [line1, setLine1] = useState('');
  const [line2, setLine2] = useState('');

  const [status, setStatus] = useState<'idle' | 'loading' | 'success' | 'error'>('idle');
  const [message, setMessage] = useState('');
  const [customProgress, setCustomProgress] = useState<CustomSatelliteAddProgress | null>(null);

  const submitLabel = useMemo(() => {
    if (status === 'loading') return mode === 'norad' ? 'Fetching TLE...' : 'Adding Satellite...';
    if (status === 'success') return 'Done';
    return mode === 'norad' ? 'Add from NORAD' : 'Add Custom Satellite';
  }, [status, mode]);

  async function handleSubmit(e: React.FormEvent) {
    e.preventDefault();
    setStatus('loading');
    setMessage('');
    setCustomProgress(null);

    try {
      let sat: CatalogItem;
      if (mode === 'norad') {
        const id = Number(noradId);
        if (!id || id < 1) {
          throw new Error('Enter a valid positive NORAD catalog number.');
        }
        sat = await addSatellite(id);
      } else {
        if (!name.trim()) {
          throw new Error('Enter a satellite name.');
        }
        if (!line1.trim().startsWith('1 ') || !line2.trim().startsWith('2 ')) {
          throw new Error("TLE format invalid: line 1 must start with '1 ' and line 2 with '2 '.");
        }
        sat = await addCustomSatellite({
          name: name.trim(),
          line1: line1.trim(),
          line2: line2.trim(),
        }, (progress) => {
          setCustomProgress(progress);
          if (progress.message) {
            setMessage(progress.message);
          }
        });
      }

      setStatus('success');
      if (mode === 'custom') {
        setCustomProgress({
          status: 'completed',
          stage: 'completed',
          progress_pct: 100,
          message: 'Satellite added successfully',
        });
      }
      setMessage(`Added: ${sat.name} (NORAD ${sat.norad_id}).`);
      setTimeout(() => {
        onAdded(sat);
        onClose();
      }, 900);
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
        style={{ width: 520, padding: '1.2rem' }}
        onClick={(e) => e.stopPropagation()}
      >
        <h3 style={{ margin: '0 0 0.8rem' }}>Add Satellite</h3>

        <div style={{ display: 'flex', gap: 8, marginBottom: 12 }}>
          <button
            type="button"
            style={mode === 'norad' ? { background: 'var(--accent)', color: '#08131f', border: 'none' } : {}}
            onClick={() => {
              setMode('norad');
              setStatus('idle');
              setMessage('');
              setCustomProgress(null);
            }}
          >
            NORAD
          </button>
          <button
            type="button"
            style={mode === 'custom' ? { background: 'var(--accent)', color: '#08131f', border: 'none' } : {}}
            onClick={() => {
              setMode('custom');
              setStatus('idle');
              setMessage('');
              setCustomProgress(null);
            }}
          >
            Custom TLE
          </button>
        </div>

        <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 10 }}>
          {mode === 'norad' ? (
            <>
              <label style={{ display: 'grid', gap: 4 }}>
                <span style={{ fontSize: '0.9rem' }}>NORAD Catalog Number</span>
                <input
                  type="number"
                  placeholder="e.g. 25544"
                  value={noradId}
                  onChange={(e) => setNoradId(e.target.value)}
                  min={1}
                  disabled={status === 'loading' || status === 'success'}
                  autoFocus
                />
              </label>
              <p className="disclaimer" style={{ margin: 0 }}>
                Fetches TLE from CelesTrak and stores locally.
              </p>
            </>
          ) : (
            <>
              <label style={{ display: 'grid', gap: 4 }}>
                <span style={{ fontSize: '0.9rem' }}>Satellite Name</span>
                <input
                  type="text"
                  placeholder="TestSat-A"
                  value={name}
                  onChange={(e) => setName(e.target.value)}
                  disabled={status === 'loading' || status === 'success'}
                  autoFocus
                />
              </label>
              <label style={{ display: 'grid', gap: 4 }}>
                <span style={{ fontSize: '0.9rem' }}>TLE Line 1</span>
                <textarea
                  placeholder="1 90001U 24001A   26052.50000000  .00010000  00000-0  15000-3 0  9997"
                  value={line1}
                  onChange={(e) => setLine1(e.target.value)}
                  rows={2}
                  disabled={status === 'loading' || status === 'success'}
                  style={{ width: '100%', background: 'var(--bg-soft)', color: 'var(--ink)', border: '1px solid rgba(255,255,255,0.14)', borderRadius: 8, padding: '0.5rem' }}
                />
              </label>
              <label style={{ display: 'grid', gap: 4 }}>
                <span style={{ fontSize: '0.9rem' }}>TLE Line 2</span>
                <textarea
                  placeholder="2 90001  51.6400 210.5000 0005000  75.0000 285.0000 15.50000000 12349"
                  value={line2}
                  onChange={(e) => setLine2(e.target.value)}
                  rows={2}
                  disabled={status === 'loading' || status === 'success'}
                  style={{ width: '100%', background: 'var(--bg-soft)', color: 'var(--ink)', border: '1px solid rgba(255,255,255,0.14)', borderRadius: 8, padding: '0.5rem' }}
                />
              </label>
              <p className="disclaimer" style={{ margin: 0 }}>
                Use this to add synthetic satellites for local collision-testing without API lookup.
              </p>
            </>
          )}

          {mode === 'custom' && status === 'loading' && customProgress && (
            <div style={{ display: 'grid', gap: 6 }}>
              <div style={{ width: '100%', height: 8, borderRadius: 99, background: 'rgba(255,255,255,0.14)', overflow: 'hidden' }}>
                <div
                  style={{
                    width: `${Math.max(3, Math.min(100, customProgress.progress_pct))}%`,
                    height: '100%',
                    background: 'linear-gradient(90deg, #4fc3f7, #66bb6a)',
                    transition: 'width 220ms ease',
                  }}
                />
              </div>
              <p className="disclaimer" style={{ margin: 0 }}>
                {customProgress.progress_pct}% · {customProgress.stage}
              </p>
            </div>
          )}

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

          <div style={{ display: 'flex', gap: 8, marginTop: 2 }}>
            <button type="submit" disabled={status === 'loading' || status === 'success'}>
              {submitLabel}
            </button>
            <button type="button" onClick={onClose}>Cancel</button>
          </div>
        </form>
      </div>
    </div>
  );
}
