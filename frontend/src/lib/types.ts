/* API + domain types for the Offgrid demo. Mirrors the /api/status contract. */

export interface ExampleRow {
  id?: string | number
  image?: string
  image_url?: string | null
  image_sha256?: string
  deployed?: 'candidate' | 'baseline' | string
  prediction_lat?: number
  prediction_lon?: number
  selected_lat?: number
  selected_lon?: number
  selected_km?: number
  baseline_lat?: number
  baseline_lon?: number
  baseline_km?: number
  true_lat?: number
  true_lon?: number
  sequence?: string
  [key: string]: unknown
}

export interface ReportModel {
  geoscore?: number
  'acc@25km'?: number
  'acc@1km'?: number
  'acc@200km'?: number
  'acc@750km'?: number
  'acc@2500km'?: number
  mean_km?: number
  median_km?: number
  [key: string]: unknown
}

export interface Report {
  accept_candidate?: boolean
  baseline?: ReportModel
  selected?: ReportModel
  decision?: string
  evaluation_queries?: number
  evaluation_sequences?: number
  mean_geoscore_delta?: number
  paired_sequence_bootstrap_95pct?: number[]
  paired_sequence_bootstrap_adjusted?: number[]
  familywise_confidence_percent?: number
  selected_top_n?: number
  selected_tower_weights?: Record<string, number>
  scope?: string
  [key: string]: unknown
}

export interface PredictionConfig {
  top_n?: number
  tower_weights?: Record<string, number>
  variant?: string
  calibrated?: boolean
  kind?: string
  model_revision?: string
  retrieval_k?: number
  towers?: string[]
  [key: string]: unknown
}

export interface StatusResponse {
  mode: 'recorded' | 'live'
  live_requested?: boolean
  cuda_available?: boolean
  can_predict?: boolean
  prediction_unavailable_reason?: string | null
  reason?: string | null
  report?: Report | null
  prediction_config?: PredictionConfig | null
  examples?: ExampleRow[]
  run_dir?: string
}

export interface PredictPayload {
  lat?: number
  lon?: number
  top_n?: number
  tower_weights?: Record<string, number>
  variant?: string
  calibrated?: boolean
  [key: string]: unknown
}

export interface GeoPoint {
  lat: number
  lon: number
}

export interface Prediction extends GeoPoint {
  source: 'live' | 'recorded'
  config?: PredictPayload
}

export interface Rect {
  x: number
  y: number
  width: number
  height: number
}

export type ToolType = 'cover' | 'crop'

export interface Operation {
  type: ToolType
  rect: Rect
}

export type ViewMode = 'original' | 'edited'

export interface MapPoint extends GeoPoint {
  key: 'original' | 'edited' | 'truth' | 'baseline'
  label: string
}

export type MetadataState = 'unchecked' | 'found' | 'absent'
