import React, { useState } from 'react';
import { LegalSource } from '../types';
import { BookOpen, ExternalLink, X, Scale } from 'lucide-react';

interface CitationDrawerProps {
  sources?: LegalSource[];
}

export const CitationDrawer: React.FC<CitationDrawerProps> = ({ sources = [] }) => {
  const [selectedSource, setSelectedSource] = useState<LegalSource | null>(null);

  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-2 pt-3 border-t border-rule">
      <div className="flex items-center gap-2 mb-2.5">
        <BookOpen className="w-3.5 h-3.5 text-vn-500" />
        <span className="label-mono">Nguồn trích dẫn · {sources.length} văn bản</span>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {sources.map((src, idx) => {
          const isWeb = src.collection === 'web' || (src.url && !src.article_number);
          const docTitle = src.document_title || src.title || `Nguồn #${idx + 1}`;
          const articleNum = src.article_number ? `Điều ${src.article_number}` : '';

          return (
            <button
              key={idx}
              onClick={() => setSelectedSource(src)}
              className={`inline-flex items-center gap-2 px-2.5 py-1.5 border rounded-swiss transition-all text-left ${
                isWeb
                  ? 'bg-sky-50/60 text-sky-900 border-sky-200 hover:border-sky-400 hover:bg-sky-100/80'
                  : 'bg-paper text-ink border-rule hover:border-ink hover:bg-paper-dim'
              }`}
            >
              {isWeb ? (
                <span className="font-mono text-[10px] uppercase font-bold text-sky-700 px-1.5 py-0.5 bg-sky-100 rounded border border-sky-200">
                  WEB
                </span>
              ) : articleNum ? (
                <span className="font-mono text-[11px] font-semibold text-vn-600 px-1.5 py-0.5 bg-vn-50 rounded border border-vn-200">
                  {articleNum}
                </span>
              ) : null}
              <span className="text-xs font-medium truncate max-w-[240px]">{docTitle}</span>
              {src.score && (
                <span className="font-mono text-[10px] opacity-70 tabular-nums">
                  {(src.score * 100).toFixed(0)}%
                </span>
              )}
              {src.url && <ExternalLink className="w-3 h-3 opacity-60 shrink-0" />}
            </button>
          );
        })}
      </div>

      {selectedSource && (
        <div
          onClick={() => setSelectedSource(null)}
          className="fixed inset-0 z-50 flex items-center justify-center p-4 backdrop-blur-md animate-fade-in"
          style={{ backgroundColor: 'rgba(0, 0, 0, 0.75)' }}
        >
          <div
            onClick={(e) => e.stopPropagation()}
            className="relative w-full max-w-2xl border border-rule rounded-2xl p-6 max-h-[85vh] flex flex-col shadow-2xl"
            style={{ backgroundColor: 'rgb(var(--c-paper))' }}
          >
            <button
              onClick={() => setSelectedSource(null)}
              className="absolute top-3 right-3 p-1.5 text-faint hover:text-ink hover:bg-paper-tint rounded-swiss transition-colors"
              aria-label="Đóng"
            >
              <X className="w-4 h-4" />
            </button>

            <div className="flex items-center gap-3 mb-4 pr-8">
              <div className="w-9 h-9 bg-ink flex items-center justify-center rounded-swiss shrink-0">
                <Scale className="w-4 h-4 text-vn-500" />
              </div>
              <div className="min-w-0">
                <h3 className="text-base font-semibold text-ink leading-tight">
                  {selectedSource.document_title || selectedSource.title || 'Chi tiết trích dẫn'}
                </h3>
                {selectedSource.article_number && (
                  <span className="font-mono text-xs font-semibold text-vn-600">
                    Điều {selectedSource.article_number}
                  </span>
                )}
              </div>
            </div>

            <div className="flex-1 overflow-y-auto p-4 bg-paper-dim border border-rule rounded-swiss text-sm text-ink leading-relaxed font-serif">
              {selectedSource.text || selectedSource.content || 'Không có bản xem trước văn bản.'}
            </div>

            <div className="mt-4 pt-3 border-t border-rule flex items-center justify-between text-xs">
              <span className="font-mono text-muted">
                Collection · {selectedSource.collection || 'llm'}
              </span>
              {selectedSource.url && (
                <a
                  href={selectedSource.url}
                  target="_blank"
                  rel="noreferrer"
                  className="inline-flex items-center gap-1.5 text-vn-600 hover:text-vn-700 font-medium"
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