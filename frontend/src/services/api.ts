import { DetailedHealth, LegalSource, ChatMessage, ApprovalRequest, AuditEntry, User } from '../types';

const API_BASE = '';

function getAuthHeaders(): Record<string, string> {
  const token = localStorage.getItem('legal_rag_jwt_token');
  const headers: Record<string, string> = {
    'Content-Type': 'application/json',
  };
  if (token) {
    headers['Authorization'] = `Bearer ${token}`;
  }
  return headers;
}

function parseApiErrorDetail(err: any): string {
  if (!err) return 'Đã xảy ra lỗi không xác định';
  if (typeof err.detail === 'string') return err.detail;
  if (Array.isArray(err.detail) && err.detail.length > 0) {
    return err.detail.map((e: any) => {
      const field = e.loc?.[e.loc.length - 1];
      if (field === 'password') return 'Mật khẩu phải chứa từ 6 đến 128 ký tự.';
      if (field === 'username') return 'Tên đăng nhập từ 3 - 64 ký tự (chỉ gồm chữ cái không dấu, số, dấu gạch).';
      return e.msg || 'Dữ liệu không hợp lệ.';
    }).join(' ');
  }
  return err.message || 'Thao tác không thành công';
}

export async function loginApi(username: string, password: string): Promise<{ access_token: string; user: User }> {
  const response = await fetch(`${API_BASE}/auth/login`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Đăng nhập thất bại' }));
    throw new Error(parseApiErrorDetail(err));
  }
  return response.json();
}

export async function registerApi(username: string, password: string, role: string = 'user'): Promise<{ access_token: string; user: User }> {
  const response = await fetch(`${API_BASE}/auth/register`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ username, password, role }),
  });
  if (!response.ok) {
    const err = await response.json().catch(() => ({ detail: 'Đăng ký thất bại' }));
    throw new Error(parseApiErrorDetail(err));
  }
  return response.json();
}

export async function fetchMeApi(): Promise<User> {
  const response = await fetch(`${API_BASE}/auth/me`, {
    headers: getAuthHeaders(),
  });
  if (!response.ok) {
    throw new Error('Unauthenticated');
  }
  return response.json();
}

export async function fetchDetailedHealthApi(): Promise<DetailedHealth> {
  try {
    const res = await fetch(`${API_BASE}/health/detailed`);
    if (!res.ok) throw new Error('Health check failed');
    const data = await res.json();
    return {
      status: data.status,
      database: { status: data.database?.status || 'unhealthy', label: 'CSDL SQL (PostgreSQL)', details: data.database?.error || 'Hoạt động' },
      redis: { status: data.redis?.status || 'unhealthy', label: 'Redis Cache', details: data.redis?.error || 'Hoạt động' },
      qdrant: { status: data.qdrant?.status || 'unhealthy', label: 'Qdrant Vector DB', details: data.qdrant?.error || 'Hoạt động' },
      celery: { status: (data.celery?.status === 'no_workers' || data.celery?.status === 'healthy') ? 'healthy' : (data.celery?.status || 'unhealthy'), label: 'Celery Worker', details: data.celery?.active_workers?.length ? `Hoạt động (${data.celery.active_workers.length} workers)` : 'Hoạt động (Sync Mode)' },
      ollama: { status: data.ollama?.status || 'not_configured', label: 'Ollama LLM', details: data.ollama?.status === 'healthy' ? 'Hoạt động' : 'Chưa cấu hình / Cloud' },
    };
  } catch (e: any) {
    return {
      status: 'unhealthy',
      database: { status: 'offline', label: 'CSDL SQL (MariaDB)', details: 'Không thể kết nối Backend' },
      redis: { status: 'offline', label: 'Redis Cache', details: 'Offline' },
      qdrant: { status: 'offline', label: 'Qdrant Vector DB', details: 'Offline' },
      celery: { status: 'offline', label: 'Celery Worker', details: 'Offline' },
      ollama: { status: 'offline', label: 'Ollama LLM', details: 'Offline' },
    };
  }
}

export async function sendChatMessageApi(
  userMessage: string,
  userId: string,
  variant?: string,
  syncRequest: boolean = false,
  conversationId?: string
): Promise<{ task_id?: string; response?: string; sources?: LegalSource[]; route?: string; tool_calls?: any[] }> {
  const res = await fetch(`${API_BASE}/chat/complete`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify({
      user_id: userId,
      user_message: userMessage,
      bot_id: 'botLawyer',
      sync_request: syncRequest,
      variant,
      conversation_id: conversationId,
    }),
  });

  if (!res.ok) {
    const err = await res.json().catch(() => ({ detail: 'Lỗi gửi tin nhắn' }));
    throw new Error(err.detail || 'Không thể xử lý yêu cầu');
  }

  return res.json();
}

export function subscribeTraceStream(
  taskId: string,
  onStep: (step: any) => void,
  onFinish: () => void,
  onError: (err: any) => void
): () => void {
  const es = new EventSource(`${API_BASE}/chat/stream/${taskId}`);

  const handleMessage = (event: MessageEvent) => {
    try {
      if (event.data) {
        const payload = JSON.parse(event.data);
        onStep(payload);
        if (payload.event_type === 'run_end' || payload.payload?.completed_early) {
          es.close();
          onFinish();
        }
      }
    } catch (e) {
      console.error('SSE JSON parse error', e);
    }
  };

  es.onmessage = handleMessage;

  const traceEventTypes = [
    'message',
    'run_start',
    'query_rewrite',
    'router',
    'vector_retrieval',
    'search',
    'web_search',
    'legal_effectivity',
    'knowledge_graph',
    'synthesis',
    'agent_tools',
    'agent',
    'retrieve',
    'generate',
    'handoff',
    'node_end',
    'run_end',
    'ready',
  ];

  traceEventTypes.forEach((evtType) => {
    es.addEventListener(evtType, handleMessage as any);
  });

  es.onerror = (err) => {
    console.warn('SSE EventSource error/closed', err);
    es.close();
    onFinish();
  };

  return () => {
    es.close();
  };
}

export async function pollTaskResultApi(taskId: string): Promise<any> {
  const res = await fetch(`${API_BASE}/chat/complete/${taskId}`, {
    headers: getAuthHeaders(),
  });
  if (!res.ok) throw new Error('Polling error');
  return res.json();
}

export async function fetchUserHistoryApi(userId: string): Promise<any[]> {
  try {
    const res = await fetch(`${API_BASE}/history/${encodeURIComponent(userId)}`, {
      headers: getAuthHeaders(),
    });
    if (!res.ok) return [];
    const data = await res.json();
    return data.history || [];
  } catch (e) {
    return [];
  }
}

export async function deleteUserHistoryApi(userId: string): Promise<boolean> {
  const res = await fetch(`${API_BASE}/history/${encodeURIComponent(userId)}`, {
    method: 'DELETE',
    headers: getAuthHeaders(),
  });
  return res.ok;
}

export async function sendFeedbackApi(data: {
  userId: string;
  conversationId?: string;
  messageId?: string;
  rating: 'good' | 'bad';
  question: string;
  response: string;
  sources?: LegalSource[];
}): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE}/feedback`, {
      method: 'POST',
      headers: getAuthHeaders(),
      body: JSON.stringify({
        user_id: data.userId,
        conversation_id: data.conversationId,
        message_id: data.messageId,
        rating: data.rating,
        question: data.question,
        response: data.response,
        sources: data.sources || [],
      }),
    });
    return res.ok;
  } catch (e) {
    return false;
  }
}

export async function fetchApprovalsApi(): Promise<ApprovalRequest[]> {
  const res = await fetch(`${API_BASE}/approvals`, {
    headers: getAuthHeaders(),
  });
  if (!res.ok) return [];
  const data = await res.json();
  return data.pending || [];
}

export async function decideApprovalApi(approvalId: string, decision: 'approved' | 'rejected', note?: string): Promise<boolean> {
  const res = await fetch(`${API_BASE}/approvals/${approvalId}/decide`, {
    method: 'POST',
    headers: getAuthHeaders(),
    body: JSON.stringify({ decision, note }),
  });
  return res.ok;
}

export async function fetchAuditLogsApi(): Promise<AuditEntry[]> {
  const res = await fetch(`${API_BASE}/audit`, {
    headers: getAuthHeaders(),
  });
  if (!res.ok) return [];
  const data = await res.json();
  return data.entries || [];
}

export async function fetchStatsApi(): Promise<any> {
  const res = await fetch(`${API_BASE}/stats`);
  if (!res.ok) return null;
  return res.json();
}
