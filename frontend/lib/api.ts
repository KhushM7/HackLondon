const configuredBase = process.env.API_BASE || process.env.NEXT_PUBLIC_API_BASE;
const API_BASE_CANDIDATES = Array.from(
  new Set(
    [configuredBase, 'http://127.0.0.1:8000', 'http://localhost:8000'].filter(
      (v): v is string => Boolean(v)
    )
  )
);

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const errors: string[] = [];
  for (const base of API_BASE_CANDIDATES) {
    try {
      const res = await fetch(`${base}${path}`, { ...init, cache: init?.cache ?? 'no-store' });
      const contentType = res.headers.get('content-type') || '';
      const raw = await res.text();

      if (!res.ok) {
        const detail = raw.slice(0, 220) || `HTTP ${res.status}`;
        throw new Error(`API ${res.status}: ${detail}`);
      }

      if (!raw) {
        throw new Error(`API returned empty response for ${path}`);
      }

      if (!contentType.includes('application/json')) {
        throw new Error(`API returned non-JSON response for ${path}: ${raw.slice(0, 220)}`);
      }

      return JSON.parse(raw) as T;
    } catch (err) {
      const message = err instanceof Error ? err.message : 'unknown error';
      errors.push(`${base}: ${message}`);
      // Try the next base URL on low-level network failures.
      if (!(err instanceof TypeError)) {
        throw err;
      }
    }
  }
  throw new Error(
    `Failed to reach OrbitGuard API for ${path}. Tried: ${errors.join(' | ')}`
  );
}

export type CatalogItem = {
  norad_id: number;
  name: string;
  inclination_deg: number;
  mean_motion: number;
  updated_at: string;
};

export type Conjunction = {
  id: number;
  defended_norad_id: number;
  intruder_norad_id: number;
  tca_utc: string;
  miss_distance_km: number;
  relative_velocity_kms: number;
  risk_tier: 'High' | 'Medium' | 'Low';
};

export type ConjunctionDetail = Conjunction & {
  pre_trajectory: { t: string; position_km: number[]; velocity_kms: number[] }[];
  intruder_trajectory: { t: string; position_km: number[]; velocity_kms: number[] }[];
  post_trajectory?: { t: string; position_km: number[]; velocity_kms: number[] }[];
  post_miss_distance_km?: number;
};

export type CongestionBand = {
  altitude_band_km: string;
  object_count: number;
  conjunction_rate: number;
};

export async function getCatalog(): Promise<CatalogItem[]> {
  return requestJson<CatalogItem[]>('/catalog');
}

export async function refreshIngest(): Promise<{ ingested: number }> {
  return requestJson<{ ingested: number }>('/ingest/refresh', { method: 'POST' });
}

export async function getConjunctions(noradId: number): Promise<Conjunction[]> {
  return requestJson<Conjunction[]>(`/assets/${noradId}/conjunctions`);
}

export async function getConjunction(eventId: number): Promise<ConjunctionDetail> {
  return requestJson<ConjunctionDetail>(`/conjunctions/${eventId}`);
}

export async function runAvoidance(eventId: number, deltaV: number): Promise<{ updated_miss_distance_km: number; delta_km: number }> {
  return requestJson<{ updated_miss_distance_km: number; delta_km: number }>(`/conjunctions/${eventId}/avoidance-sim`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ delta_v_mps: deltaV })
  });
}

export async function getCongestion(): Promise<{ bands: CongestionBand[] }> {
  return requestJson<{ bands: CongestionBand[] }>('/congestion');
}

export type SatellitePosition = {
  norad_id: number;
  name: string;
  position_km: [number, number, number];
  risk_tier: 'High' | 'Medium' | 'Low' | null;
};

export async function getCatalogPositions(limit = 500): Promise<SatellitePosition[]> {
  return requestJson<SatellitePosition[]>(`/catalog/positions?limit=${limit}`);
}

export async function addSatellite(noradId: number): Promise<CatalogItem> {
  return requestJson<CatalogItem>('/catalog/satellite', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({ norad_id: noradId }),
  });
}

export type CustomSatelliteResponse = {
  satellite: CatalogItem;
  conjunctions_found: number;
  events: {
    id: number;
    intruder_norad_id: number;
    tca_utc: string;
    miss_distance_km: number;
    risk_tier: string;
  }[];
};

export async function addCustomSatellite(payload: {
  name: string;
  line1: string;
  line2: string;
}): Promise<CatalogItem> {
  const resp = await requestJson<CustomSatelliteResponse>('/catalog/custom-satellite', {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify(payload),
  });
  return resp.satellite;
}
