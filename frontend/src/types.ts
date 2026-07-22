export interface User {
  id: string;
  username: string;
  role: 'admin' | 'lawyer' | 'user' | 'guest';
}

export interface AuthState {
  user: User | null;
  token: string | null;
  isAuthenticated: boolean;
  isLoading: boolean;
}

export interface LegalSource {
  title?: string;
  document_title?: string;
  article_number?: string;
  text?: string;
  content?: string;
  url?: string;
  score?: number;
  collection?: string;
  [key: string]: any;
}

export interface AgentStepPayload {
  node?: string;
  step_index?: number;
  event_type?: string;
  payload?: any;
  status?: string;
  cached?: boolean;
}

export interface ChatMessage {
  id: string;
  role: 'user' | 'assistant' | 'system';
  content: string;
  timestamp: string;
  sources?: LegalSource[];
  route?: string;
  tool_calls?: any[];
  cached?: boolean;
  rating?: 'good' | 'bad';
  isStreaming?: boolean;
  traceSteps?: AgentStepPayload[];
}

export interface ServiceHealth {
  status: 'healthy' | 'unhealthy' | 'offline' | 'not_configured' | 'no_workers';
  label: string;
  details: string;
}

export interface DetailedHealth {
  status: 'healthy' | 'unhealthy';
  database: ServiceHealth;
  redis: ServiceHealth;
  qdrant: ServiceHealth;
  celery: ServiceHealth;
  ollama: ServiceHealth;
}

export interface ApprovalRequest {
  id: string;
  tool_name: string;
  arguments: any;
  status: 'pending' | 'approved' | 'rejected';
  created_at: string;
  requester_id?: string;
}

export interface AuditEntry {
  id: number;
  user_id?: string;
  action: string;
  resource: string;
  timestamp: string;
  ip?: string;
  payload?: any;
}
