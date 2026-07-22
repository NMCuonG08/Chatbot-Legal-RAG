import React from 'react';
import { LegalSource } from '../types';
import { BookOpen, ExternalLink } from 'lucide-react';

interface CitationDrawerProps {
  sources?: LegalSource[];
  onSelectSource?: (src: LegalSource) => void;
}

export const CitationDrawer: React.FC<CitationDrawerProps> = ({
  sources = [],
  onSelectSource,
}) => {
  if (!sources || sources.length === 0) return null;

  return (
    <div className="mt-2 pt-3 border-t border-rule">
      <div className="flex items-center gap-2 mb-2.5">
        <BookOpen className="w-3.5 h-3.5 text-vn-500" />
        <span className="label-mono">Nguồn trích dẫn · {sources.length} văn bản</span>
      </div>

      <div className="flex flex-wrap gap-1.5">
        {sources.map((src, idx) => {
          const isWeb =
            src.collection === 'web' ||
            (src as any).kind === 'web_search' ||
            (src as any).source_type === 'web';
          const docTitle = src.document_title || src.title || `Nguồn #${idx + 1}`;
          const articleNum = src.article_number ? `Điều ${src.article_number}` : '';

          return (
            <button
              key={idx}
              onClick={() => onSelectSource && onSelectSource(src)}
              className={`inline-flex items-center gap-2 px-2.5 py-1.5 border rounded-swiss transition-all text-left cursor-pointer ${
                isWeb
                  ? 'bg-sky-500/10 text-sky-700 dark:text-sky-300 border-sky-500/30 hover:border-sky-500 hover:bg-sky-500/20'
                  : 'bg-paper text-ink border-rule hover:border-ink hover:bg-paper-dim'
              }`}
            >
              {isWeb ? (
                <span className="font-mono text-[10px] uppercase font-bold text-sky-600 dark:text-sky-400 px-1.5 py-0.5 bg-sky-500/15 rounded border border-sky-500/30">
                  WEB
                </span>
              ) : articleNum ? (
                <span className="font-mono text-[11px] font-semibold text-vn-600 dark:text-vn-400 px-1.5 py-0.5 bg-vn-500/15 rounded border border-vn-500/30">
                  {articleNum}
                </span>
              ) : (
                <span className="font-mono text-[10px] uppercase font-bold text-vn-600 dark:text-vn-400 px-1.5 py-0.5 bg-vn-500/15 rounded border border-vn-500/30">
                  LUẬT
                </span>
              )}
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
    </div>
  );
};