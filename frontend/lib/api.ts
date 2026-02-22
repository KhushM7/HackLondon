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
  defended_name?: string | null;
  intruder_norad_id: number;
  intruder_name?: string | null;
  tca_utc: string;
  miss_distance_km: number;
  relative_velocity_kms: number;
  risk_tier: 'High' | 'Medium' | 'Low';
  pc_foster?: number | null;
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

const conjunctionsInFlight = new Map<number, Promise<Conjunction[]>>();

export async function getConjunctions(noradId: number): Promise<Conjunction[]> {
  const existing = conjunctionsInFlight.get(noradId);
  if (existing) {
    return existing;
  }

  const request = requestJson<Conjunction[]>(`/assets/${noradId}/conjunctions`).finally(() => {
    conjunctionsInFlight.delete(noradId);
  });
  conjunctionsInFlight.set(noradId, request);
  return request;
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
  velocity_kms: [number, number, number];
  risk_tier: 'High' | 'Medium' | 'Low' | null;
};

export type ConjunctionLink = {
  event_id: number;
  defended_norad_id: number;
  intruder_norad_id: number;
  risk_tier: 'High' | 'Medium' | 'Low';
  miss_distance_km: number;
};

export type CatalogPositionsResponse = {
  satellites: SatellitePosition[];
  links: ConjunctionLink[];
};

export type OrbitPathResponse = {
  norad_id: number;
  positions_km: [number, number, number][];
};

export async function getCatalogPositions(limit?: number, focusNoradId?: number): Promise<CatalogPositionsResponse> {
  const params = new URLSearchParams();
  if (typeof limit === 'number') {
    params.set('limit', String(limit));
  }
  if (typeof focusNoradId === 'number') {
    params.set('focus_norad_id', String(focusNoradId));
  }
  const qs = params.toString() ? `?${params.toString()}` : '';
  return requestJson<CatalogPositionsResponse>(`/catalog/positions${qs}`);
}

export async function getOrbitPath(noradId: number, orbits?: number): Promise<OrbitPathResponse> {
  const qs = typeof orbits === 'number' ? `?orbits=${orbits}` : '';
  return requestJson<OrbitPathResponse>(`/catalog/orbit/${noradId}${qs}`);
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

export type CustomSatelliteAddJob = {
  job_id: string;
  status: 'queued' | 'running' | 'completed' | 'failed';
  stage: string;
  progress_pct: number;
  message?: string | null;
  satellite?: CatalogItem | null;
  conjunctions_found?: number | null;
  error?: string | null;
};

export type CustomSatelliteAddProgress = {
  status: 'queued' | 'running' | 'completed' | 'failed';
  stage: string;
  progress_pct: number;
  message?: string | null;
};

const customSatelliteInFlight = new Map<string, Promise<CatalogItem>>();

function sleep(ms: number): Promise<void> {
  return new Promise((resolve) => setTimeout(resolve, ms));
}

export async function addCustomSatellite(payload: {
  name: string;
  line1: string;
  line2: string;
}, onProgress?: (progress: CustomSatelliteAddProgress) => void): Promise<CatalogItem> {
  const key = `${payload.name.trim().toLowerCase()}|${payload.line1.trim()}|${payload.line2.trim()}`;
  const existing = customSatelliteInFlight.get(key);
  if (existing) {
    return existing;
  }

  const request = (async () => {
    let state = await requestJson<CustomSatelliteAddJob>('/catalog/custom-satellite/submit', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify(payload),
    });
    onProgress?.({
      status: state.status,
      stage: state.stage,
      progress_pct: state.progress_pct,
      message: state.message,
    });

    while (state.status === 'queued' || state.status === 'running') {
      await sleep(650);
      state = await requestJson<CustomSatelliteAddJob>(`/catalog/custom-satellite/jobs/${state.job_id}`);
      onProgress?.({
        status: state.status,
        stage: state.stage,
        progress_pct: state.progress_pct,
        message: state.message,
      });
    }

    if (state.status !== 'completed' || !state.satellite) {
      throw new Error(state.error || 'Failed to add custom satellite');
    }
    return state.satellite;
  })().finally(() => {
    customSatelliteInFlight.delete(key);
  });

  customSatelliteInFlight.set(key, request);
  return request;
}

// --- Avoidance optimization ---

export type AvoidancePlan = {
  id: number;
  asset_norad_id: number;
  status: 'pending' | 'running' | 'completed' | 'failed' | 'cancelled';
  event_id?: number | null;
  burn_direction?: string | null;
  burn_dv_mps?: number | null;
  burn_rtn_vector?: number[] | null;
  burn_epoch?: string | null;
  pre_miss_distance_km?: number | null;
  post_miss_distance_km?: number | null;
  pre_pc?: number | null;
  post_pc?: number | null;
  fuel_cost_kg?: number | null;
  current_path?: ({ t: string; position_km: [number, number, number] } | { t: string; lat_lon_alt: [number, number, number] })[] | null;
  deviated_path?: ({ t: string; position_km: [number, number, number] } | { t: string; lat_lon_alt: [number, number, number] })[] | null;
  candidates_evaluated?: number | null;
  optimization_elapsed_s?: number | null;
  progress_stage?: string | null;
  progress_done?: number | null;
  progress_total?: number | null;
  progress_message?: string | null;
  heartbeat_at?: string | null;
  error_message?: string | null;
  created_at: string;
  completed_at?: string | null;
};

export async function optimizeAvoidance(
  noradId: number,
  opts?: { max_delta_v_mps?: number; burn_window_hours?: number; top_n_events?: number }
): Promise<AvoidancePlan> {
  return requestJson<AvoidancePlan>(`/assets/${noradId}/avoidance/optimize`, {
    method: 'POST',
    headers: { 'content-type': 'application/json' },
    body: JSON.stringify({
      max_delta_v_mps: opts?.max_delta_v_mps ?? 5.0,
      burn_window_hours: opts?.burn_window_hours ?? 48.0,
      top_n_events: opts?.top_n_events ?? 3,
    }),
  });
}

export async function getAvoidancePlan(noradId: number): Promise<AvoidancePlan> {
  return requestJson<AvoidancePlan>(`/assets/${noradId}/avoidance/plan`);
}
