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
          const kind = (src as any).kind || (src as any).source_type || '';
          const collection = src.collection || '';
          const url = src.url || '';
          const isWeb =
            collection === 'web' ||
            kind === 'web' ||
            kind === 'web_search' ||
            (url.startsWith('http') && !src.article_number && !(src as any).law_id && !src.document_title?.toLowerCase().includes('luật') && !src.document_title?.toLowerCase().includes('nghị định') && !src.document_title?.toLowerCase().includes('thông tư'));

          const docTitle = src.document_title || src.title || `Nguồn #${idx + 1}`;
          const articleNum = src.article_number ? `Điều ${src.article_number}` : '';

          return (
            <div
              key={idx}
              onClick={() => onSelectSource && onSelectSource(src)}
              className={`inline-flex items-center gap-2 px-2.5 py-1.5 border rounded-swiss transition-all text-left cursor-pointer group ${
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
              <span className="text-xs font-medium truncate max-w-[220px]">{docTitle}</span>
              {src.score && (
                <span className="font-mono text-[10px] opacity-70 tabular-nums">
                  {(src.score * 100).toFixed(0)}%
                </span>
              )}
              {src.url && (
                <a
                  href={src.url}
                  target="_blank"
                  rel="noopener noreferrer"
                  onClick={(e) => e.stopPropagation()}
                  title={`Mở liên kết web: ${src.url}`}
                  className="p-0.5 text-faint hover:text-sky-500 transition-colors shrink-0"
                >
                  <ExternalLink className="w-3.5 h-3.5" />
                </a>
              )}
            </div>
          );
        })}
      </div>
    </div>
  );
};