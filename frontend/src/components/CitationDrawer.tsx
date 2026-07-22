import React, { useState } from 'react';
import { LegalSource } from '../types';
import { BookOpen, FileText, ExternalLink, X, Scale } from 'lucide-react';

interface CitationDrawerProps {
  sources?: LegalSource[];
}

export const CitationDrawer: React.FC<CitationDrawerProps> = ({ sources = [] }) => {
  const [selectedSource, setSelectedSource] = useState<LegalSource | null>(null);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-4 border-t border-slate-800/80 pt-3">
      <div className="flex items-center gap-2 mb-2">
        <BookOpen className="w-4 h-4 text-legal-400" />
        <span className="text-xs font-semibold text-slate-300">
          Nguồn Trích Dẫn Pháp Lý ({sources.length} văn bản)
        </span>
      </div>

      {/* Sources Pills */}
      <div className="flex flex-wrap gap-2">
        {sources.map((src, idx) => {
          const docTitle = src.document_title || src.title || `Văn bản pháp luật #${idx + 1}`;
          const articleNum = src.article_number ? `Điều ${src.article_number}` : '';
          return (
            <button
              key={idx}
              onClick={() => setSelectedSource(src)}
              className="px-2.5 py-1.5 rounded-lg glass-panel-hover text-left text-xs text-slate-300 flex items-center gap-2 border border-slate-800 hover:border-legal-500/40"
            >
              <FileText className="w-3.5 h-3.5 text-legal-400 shrink-0" />
              <span className="font-medium text-slate-200 truncate max-w-[200px]">
                {articleNum ? `${articleNum} - ${docTitle}` : docTitle}
              </span>
              {src.score && (
                <span className="text-[10px] font-mono px-1.5 py-0.2 rounded bg-legal-950 text-legal-400 border border-legal-800">
                  {(src.score * 100).toFixed(0)}%
                </span>
              )}
            </button>
          );
        })}
      </div>

      {/* Source Detail Modal */}
      {selectedSource && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-sm animate-fade-in">
          <div className="relative w-full max-w-2xl glass-panel rounded-2xl p-6 shadow-2xl border border-slate-700 max-h-[85vh] flex flex-col">
            <button
              onClick={() => setSelectedSource(null)}
              className="absolute top-4 right-4 p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800 transition"
            >
              <X className="w-5 h-5" />
            </button>

            <div className="flex items-center gap-3 mb-4">
              <div className="w-10 h-10 rounded-xl bg-legal-600/20 border border-legal-500/30 flex items-center justify-center text-legal-400">
                <Scale className="w-5 h-5" />
              </div>
              <div>
                <h3 className="text-base font-bold text-white leading-tight">
                  {selectedSource.document_title || selectedSource.title || 'Chi Tiết Trích Dẫn Pháp Lý'}
                </h3>
                {selectedSource.article_number && (
                  <span className="text-xs font-semibold text-legal-400">
                    Điều {selectedSource.article_number}
                  </span>
                )}
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-4 rounded-xl bg-slate-900/90 border border-slate-800 text-sm text-slate-300 leading-relaxed font-serif">
              {selectedSource.text || selectedSource.content || 'Không có bản xem trước văn bản.'}
            </div>

            <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between text-xs text-slate-400">
              <span>Collection: {selectedSource.collection || 'llm'}</span>
              {selectedSource.url && (
                <a
                  href={selectedSource.url}
                  target="_blank"
                  rel="noreferrer"
                  className="flex items-center gap-1 text-legal-400 hover:underline font-medium"
                >
                  Xem văn bản gốc <ExternalLink className="w-3.5 h-3.5" />
                </a>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  );
};
