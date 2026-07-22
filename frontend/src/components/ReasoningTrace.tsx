import React, { useState } from 'react';
import { AgentStepPayload } from '../types';
import { Brain, ChevronDown, ChevronUp, Search, Sparkles, Database, FileCheck, Layers } from 'lucide-react';

interface ReasoningTraceProps {
  traceSteps?: AgentStepPayload[];
  isStreaming?: boolean;
}

const pad2 = (n: number) => String(n).padStart(2, '0');

export const ReasoningTrace: React.FC<ReasoningTraceProps> = ({ traceSteps = [], isStreaming = false }) => {
  const [isOpen, setIsOpen] = useState<boolean>(true);

  if (traceSteps.length === 0 && !isStreaming) return null;

  const getNodeIcon = (node?: string) => {
    switch (node) {
      case 'query_rewrite':
      case 'router':
        return <Search className="w-3.5 h-3.5 text-ink" />;
      case 'vector_retrieval':
      case 'search':
        return <Database className="w-3.5 h-3.5 text-ink" />;
      case 'legal_effectivity':
        return <FileCheck className="w-3.5 h-3.5 text-ink" />;
      case 'knowledge_graph':
        return <Layers className="w-3.5 h-3.5 text-ink" />;
      default:
        return <Sparkles className="w-3.5 h-3.5 text-ink" />;
    }
  };

  const formatNodeName = (node?: string) => {
    switch (node) {
      case 'query_rewrite': return 'Chuẩn hóa & tối ưu câu hỏi';
      case 'router': return 'Phân tích lộ trình xử lý';
      case 'vector_retrieval': return 'Tra cứu CSDL văn bản (Qdrant)';
      case 'legal_effectivity': return 'Kiểm tra hiệu lực văn bản';
      case 'knowledge_graph': return 'Duyệt đồ thị tri thức (Neo4j)';
      case 'synthesis': return 'Tổng hợp & tạo câu trả lời';
      default: return node || 'Xử lý dữ liệu';
    }
  };

  return (
    <div className="border border-rule rounded-swiss overflow-hidden bg-paper-dim">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-3.5 py-2.5 flex items-center justify-between hover:bg-paper-tint transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <Brain className={`w-4 h-4 text-vn-500 ${isStreaming ? 'animate-pulse' : ''}`} />
          <span className="label-mono">Suy luận Agent · {traceSteps.length} bước</span>
          {isStreaming && (
            <span className="inline-flex items-center gap-1.5 ml-1 px-2 py-0.5 bg-vn-50 text-vn-700 border border-vn-100 rounded-swiss font-mono text-[10px] uppercase tracking-label">
              <span className="w-1.5 h-1.5 bg-vn-500 rounded-full animate-ping" />
              Đang chạy
            </span>
          )}
        </div>
        {isOpen ? <ChevronUp className="w-4 h-4 text-faint" /> : <ChevronDown className="w-4 h-4 text-faint" />}
      </button>

      {isOpen && (
        <div className="p-3.5 border-t border-rule space-y-2.5">
          {traceSteps.map((step, idx) => (
            <div key={idx} className="flex items-start gap-3">
              <span className="font-mono text-[10px] text-faint pt-1 tabular-nums shrink-0">
                {pad2(step.step_index ?? idx + 1)}
              </span>
              <div className="flex-1 min-w-0">
                <div className="flex items-center gap-2">
                  {getNodeIcon(step.node)}
                  <span className="text-sm font-medium text-ink">{formatNodeName(step.node)}</span>
                </div>
                {step.payload && typeof step.payload === 'object' && (
                  <pre className="mt-1.5 p-2.5 bg-ink text-paper font-mono text-[11px] overflow-x-auto max-h-28 rounded-swiss">
                    {JSON.stringify(step.payload, null, 2)}
                  </pre>
                )}
              </div>
            </div>
          ))}
          {isStreaming && traceSteps.length === 0 && (
            <div className="flex items-center gap-2 text-muted text-xs py-1">
              <div className="w-3 h-3 border-2 border-vn-500 border-t-transparent rounded-full animate-spin" />
              <span>Đang kết nối luồng SSE từ agent…</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};