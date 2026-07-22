import React from 'react';
import { LegalSource } from '../types';
import { BookOpen, ExternalLink, X, Scale } from 'lucide-react';

interface RightCitationPanelProps {
  sources?: LegalSource[];
  selectedSource: LegalSource | null;
  onSelectSource: (src: LegalSource | null) => void;
  isOpen: boolean;
  onClose: () => void;
}

export const RightCitationPanel: React.FC<RightCitationPanelProps> = ({
  sources = [],
  selectedSource,
  onSelectSource,
  isOpen,
  onClose,
}) => {
  if (!isOpen) return null;

  const activeSrc = selectedSource || (sources.length > 0 ? sources[0] : null);

  return (
    <aside className="w-80 lg:w-96 h-full border-l border-rule bg-paper flex flex-col shrink-0 shadow-2xl transition-all relative z-20 animate-slide-left">
      {/* Panel Header */}
      <div className="px-4 py-3.5 border-b border-rule flex items-center justify-between bg-paper-dim">
        <div className="flex items-center gap-2 min-w-0 pr-2">
          <BookOpen className="w-4 h-4 text-vn-600 dark:text-vn-400 shrink-0" />
          <span className="label-mono font-bold text-ink truncate">Chi tiết nguồn trích dẫn</span>
        </div>
        <button
          onClick={onClose}
          className="p-1.5 text-faint hover:text-ink hover:bg-paper-tint rounded-swiss transition-colors shrink-0"
          title="Đóng bảng bên phải"
        >
          <X className="w-4 h-4" />
        </button>
      </div>

      {/* Sources List Pills (if multiple sources exist for switching) */}
      {sources.length > 1 && (
        <div className="p-3 border-b border-rule bg-paper flex flex-wrap gap-1.5 max-h-36 overflow-y-auto">
          {sources.map((src, idx) => {
            const isWeb =
              src.collection === 'web' ||
              (src as any).kind === 'web_search' ||
              (src as any).source_type === 'web';
            const docTitle = src.document_title || src.title || `Nguồn #${idx + 1}`;
            const articleNum = src.article_number ? `Điều ${src.article_number}` : '';
            const isSelected = activeSrc === src;

            return (
              <button
                key={idx}
                onClick={() => onSelectSource(src)}
                className={`inline-flex items-center gap-1.5 px-2 py-1 border rounded-swiss text-left transition-all cursor-pointer ${
                  isSelected
                    ? 'border-vn-600 bg-vn-500/15 text-ink font-semibold'
                    : isWeb
                    ? 'bg-sky-500/10 text-sky-700 dark:text-sky-300 border-sky-500/30 hover:border-sky-500'
                    : 'bg-paper text-ink border-rule hover:border-ink hover:bg-paper-dim'
                }`}
              >
                {isWeb ? (
                  <span className="font-mono text-[9px] uppercase font-bold text-sky-600 dark:text-sky-400 px-1 py-0.2 bg-sky-500/20 rounded">
                    WEB
                  </span>
                ) : articleNum ? (
                  <span className="font-mono text-[10px] font-semibold text-vn-600 dark:text-vn-400">
                    {articleNum}
                  </span>
                ) : (
                  <span className="font-mono text-[9px] uppercase font-bold text-vn-600 dark:text-vn-400 px-1 py-0.2 bg-vn-500/20 rounded">
                    LUẬT
                  </span>
                )}
                <span className="text-[11px] truncate max-w-[130px]">{docTitle}</span>
              </button>
            );
          })}
        </div>
      )}

      {/* Selected Source Document Details */}
      {activeSrc ? (
        <div className="flex-1 flex flex-col overflow-hidden p-4">
          <div className="flex items-center gap-2 mb-3">
            <div className="w-8 h-8 bg-vn-500/10 border border-vn-500/20 flex items-center justify-center rounded-swiss shrink-0">
              <Scale className="w-4 h-4 text-vn-600 dark:text-vn-400" />
            </div>
            <div className="min-w-0">
              <h4 className="text-xs font-semibold text-ink leading-tight truncate">
                {activeSrc.document_title || activeSrc.title || 'Chi tiết trích dẫn'}
              </h4>
              {activeSrc.article_number && (
                <span className="font-mono text-[11px] font-medium text-vn-600 dark:text-vn-400">
                  Điều {activeSrc.article_number}
                </span>
              )}
            </div>
          </div>

          <div className="flex-1 overflow-y-auto p-3.5 bg-paper-dim border border-rule rounded-swiss text-xs text-ink leading-relaxed font-serif whitespace-pre-wrap">
            {activeSrc.text || activeSrc.content || 'Không có bản xem trước văn bản.'}
          </div>

          <div className="mt-3 pt-2 border-t border-rule flex items-center justify-between text-[11px]">
            <span className="font-mono text-muted">
              Collection · {activeSrc.collection || 'llm'}
            </span>
            {activeSrc.url && (
              <a
                href={activeSrc.url}
                target="_blank"
                rel="noreferrer"
                className="inline-flex items-center gap-1 text-vn-600 dark:text-vn-400 hover:underline font-medium"
              >
                Mở link gốc <ExternalLink className="w-3 h-3" />
              </a>
            )}
          </div>
        </div>
      ) : (
        <div className="flex-1 flex items-center justify-center p-6 text-xs text-muted text-center">
          Nhấp vào một nguồn trích dẫn hoặc chỉ số [1], [2] để xem chi tiết văn bản ở đây.
        </div>
      )}
    </aside>
  );
};
