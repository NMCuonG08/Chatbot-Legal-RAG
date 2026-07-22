import React, { useState } from 'react';
import { AgentStepPayload } from '../types';
import { Brain, ChevronDown, ChevronUp, Search, Sparkles, Database, FileCheck, Layers } from 'lucide-react';

interface ReasoningTraceProps {
  traceSteps?: AgentStepPayload[];
  isStreaming?: boolean;
}

export const ReasoningTrace: React.FC<ReasoningTraceProps> = ({ traceSteps = [], isStreaming = false }) => {
  const [isOpen, setIsOpen] = useState<boolean>(true);

  if (traceSteps.length === 0 && !isStreaming) return null;

  const getNodeIcon = (node?: string) => {
    switch (node) {
      case 'query_rewrite':
      case 'router':
        return <Search className="w-3.5 h-3.5 text-sky-400" />;
      case 'vector_retrieval':
      case 'search':
        return <Database className="w-3.5 h-3.5 text-legal-400" />;
      case 'legal_effectivity':
        return <FileCheck className="w-3.5 h-3.5 text-amber-400" />;
      case 'knowledge_graph':
        return <Layers className="w-3.5 h-3.5 text-purple-400" />;
      default:
        return <Sparkles className="w-3.5 h-3.5 text-emerald-400" />;
    }
  };

  const formatNodeName = (node?: string) => {
    switch (node) {
      case 'query_rewrite': return 'Chuẩn hóa & Tối ưu câu hỏi';
      case 'router': return 'Phân tích lộ trình xử lý';
      case 'vector_retrieval': return 'Tra cứu CSDL Văn bản Pháp luật (Qdrant)';
      case 'legal_effectivity': return 'Kiểm tra hiệu lực văn bản pháp lý';
      case 'knowledge_graph': return 'Duyệt Đồ thị Tri thức (Neo4j GraphRAG)';
      case 'synthesis': return 'Tổng hợp thông tin & Tạo câu trả lời';
      default: return node || 'Xử lý dữ liệu';
    }
  };

  return (
    <div className="mb-4 rounded-xl glass-panel border border-slate-800/80 overflow-hidden text-xs">
      <button
        onClick={() => setIsOpen(!isOpen)}
        className="w-full px-4 py-2.5 bg-slate-900/60 hover:bg-slate-900 flex items-center justify-between transition text-slate-300 font-medium"
      >
        <div className="flex items-center gap-2">
          <Brain className={`w-4 h-4 text-legal-400 ${isStreaming ? 'animate-bounce' : ''}`} />
          <span>Quá trình suy luận của Agent ({traceSteps.length} bước)</span>
          {isStreaming && (
            <span className="flex items-center gap-1.5 ml-2 px-2 py-0.5 rounded-full bg-legal-500/10 text-legal-400 text-[10px] border border-legal-500/20 font-mono">
              <span className="w-1.5 h-1.5 rounded-full bg-legal-400 animate-ping" />
              Đang suy luận...
            </span>
          )}
        </div>
        {isOpen ? <ChevronUp className="w-4 h-4 text-slate-400" /> : <ChevronDown className="w-4 h-4 text-slate-400" />}
      </button>

      {isOpen && (
        <div className="p-3 space-y-2 border-t border-slate-800/60 bg-slate-950/40">
          {traceSteps.map((step, idx) => (
            <div key={idx} className="flex items-start gap-2 text-slate-300 text-xs">
              <div className="mt-0.5 p-1 rounded-md bg-slate-900 border border-slate-800">
                {getNodeIcon(step.node)}
              </div>
              <div className="flex-1 min-w-0">
                <div className="flex items-center justify-between">
                  <span className="font-semibold text-slate-200">{formatNodeName(step.node)}</span>
                  <span className="text-[10px] font-mono text-slate-500">#{step.step_index ?? idx + 1}</span>
                </div>
                {step.payload && typeof step.payload === 'object' && (
                  <div className="mt-1 p-2 rounded bg-slate-900/80 font-mono text-[11px] text-slate-400 overflow-x-auto max-h-24">
                    {JSON.stringify(step.payload, null, 2)}
                  </div>
                )}
              </div>
            </div>
          ))}
          {isStreaming && traceSteps.length === 0 && (
            <div className="text-slate-400 text-xs py-2 flex items-center gap-2">
              <div className="w-3 h-3 border-2 border-legal-400 border-t-transparent rounded-full animate-spin" />
              <span>Đang kết nối luồng sự kiện SSE từ agent...</span>
            </div>
          )}
        </div>
      )}
    </div>
  );
};
