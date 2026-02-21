import Link from 'next/link';

import { EventReplayClient } from '../../../components/EventReplayClient';
import { getConjunction } from '../../../lib/api';

export default async function ConjunctionView({ params }: { params: { eventId: string } }) {
  const eventId = Number(params.eventId);
  let event = null;
  let error: string | null = null;
  try {
    event = await getConjunction(eventId);
  } catch (err) {
    error = err instanceof Error ? err.message : 'Failed to load conjunction event.';
  }

  if (!event) {
    return (
      <div className="grid" style={{ gap: 12 }}>
        <div className="panel">
          <h2>Conjunction Event #{eventId}</h2>
          <p style={{ color: '#ffb36d' }}>{error || 'Event unavailable.'}</p>
          <Link href="/">Back to Dashboard</Link>
        </div>
      </div>
    );
  }

  return (
    <div className="grid" style={{ gap: 12 }}>
      <div className="panel">
        <h2>Conjunction Event #{event.id}</h2>
        <p>
          Defended {event.defended_norad_id} vs Intruder {event.intruder_norad_id} | Miss Distance {event.miss_distance_km.toFixed(3)} km
        </p>
        <p className="disclaimer">Closest approach replay with pre/post manoeuvre comparison.</p>
        <Link href={`/assets/${event.defended_norad_id}`}>Back to Asset</Link>
      </div>

      <EventReplayClient initialEvent={event} />
    </div>
  );
}
