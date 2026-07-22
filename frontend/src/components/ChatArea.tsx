import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage } from '../types';
import { ReasoningTrace } from './ReasoningTrace';
import { CitationDrawer } from './CitationDrawer';
import { sendFeedbackApi } from '../services/api';
import { useAuth } from '../context/AuthContext';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import {
  ArrowUp,
  Copy,
  Check,
  ThumbsUp,
  ThumbsDown,
  CornerDownLeft,
  Lock,
  LogIn,
} from 'lucide-react';

interface ChatAreaProps {
  messages: ChatMessage[];
  onSendMessage: (text: string) => void;
  isLoading: boolean;
  userId: string;
  isAuthenticated: boolean;
  onSelectSource?: (src: LegalSource | null) => void;
}

const samplePrompts: { idx: string; title: string; subtitle: string; prompt: string }[] = [
  {
    idx: '01',
    title: 'Chuyển nhượng quyền sử dụng đất',
    subtitle: 'Theo Luật Đất đai 2024',
    prompt:
      'Cho tôi biết điều kiện để chuyển nhượng quyền sử dụng đất theo quy định của Luật Đất đai 2024?',
  },
  {
    idx: '02',
    title: 'Thời giờ làm việc & làm thêm giờ',
    subtitle: 'Bộ luật Lao động 2019',
    prompt:
      'Quy định về thời giờ làm việc bình thường và giới hạn làm thêm giờ theo Bộ luật Lao động 2019?',
  },
  {
    idx: '03',
    title: 'Thành lập doanh nghiệp tư nhân',
    subtitle: 'Hồ sơ & thủ tục theo Luật Doanh nghiệp',
    prompt:
      'Hồ sơ và thủ tục thành lập doanh nghiệp tư nhân theo quy định của Luật Doanh nghiệp bao gồm những gì?',
  },
  {
    idx: '04',
    title: 'Án lệ tranh chấp hợp đồng',
    subtitle: 'Mua bán tài sản · phương thức giải quyết',
    prompt:
      'Hãy tóm tắt các án lệ liên quan đến tranh chấp hợp đồng mua bán tài sản và phương thức giải quyết?',
  },
];

export const ChatArea: React.FC<ChatAreaProps> = ({
  messages,
  onSendMessage,
  isLoading,
  userId,
  isAuthenticated,
  onSelectSource,
}) => {
  const { setShowAuthModal } = useAuth();
  const [inputText, setInputText] = useState<string>('');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const [activeModalSource, setActiveModalSource] = useState<LegalSource | null>(null);

  const formatContentWithCitations = (content: string) => {
    if (!content) return '';
    return content.replace(/\[(\d+)\]/g, '[$1](#cite-$1)');
  };
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputText.trim() || isLoading || !isAuthenticated) return;
    onSendMessage(inputText.trim());
    setInputText('');
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSubmit();
    }
  };

  const handleCopy = (id: string, text: string) => {
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const handleRating = async (msg: ChatMessage, rating: 'good' | 'bad') => {
    msg.rating = rating;
    await sendFeedbackApi({
      userId,
      messageId: msg.id,
      rating,
      question: messages[messages.length - 2]?.content || '',
      response: msg.content,
      sources: msg.sources,
    });
  };

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-paper relative">
      <div className="flex-1 overflow-y-auto">
        {messages.length === 0 ? (
          <div className="max-w-3xl mx-auto px-6 py-16 md:py-24 animate-fade-in">
            <div className="flex items-center gap-3 mb-6">
              <span className="h-px w-10 bg-vn-500" />
              <span className="label-mono text-vn-600">Trợ lý AI · Vietnamese Legal RAG</span>
            </div>

            <h2 className="text-4xl md:text-5xl font-semibold tracking-display leading-[1.05] text-ink">
              Đặt câu hỏi pháp luật,
              <br />
              <span className="text-vn-600">nhận câu trả lời có trích dẫn.</span>
            </h2>

            {isAuthenticated ? (
              <>
                <p className="mt-5 text-base text-muted max-w-xl leading-relaxed">
                  Tra cứu văn bản quy phạm pháp luật & án lệ qua đồ thị tri thức RAG Agentic —
                  mỗi câu trả lời đi kèm điều luật và nguồn trích dẫn.
                </p>

                <div className="mt-10 border-t border-rule">
                  <div className="grid grid-cols-1 sm:grid-cols-2 divide-y sm:divide-y-0 sm:divide-x divide-rule border-b border-rule">
                    {samplePrompts.map((p) => (
                      <button
                        key={p.idx}
                        onClick={() => onSendMessage(p.prompt)}
                        className="group flex gap-4 px-1 py-5 text-left hover:bg-paper-dim transition-colors sm:px-5"
                      >
                        <span className="font-mono text-xs text-vn-500 pt-1 tabular-nums shrink-0">
                          {p.idx}
                        </span>
                        <div className="min-w-0">
                          <p className="font-medium text-sm text-ink group-hover:text-vn-700 transition-colors">
                            {p.title}
                          </p>
                          <p className="mt-1 text-xs text-muted leading-snug">{p.subtitle}</p>
                        </div>
                      </button>
                    ))}
                  </div>
                </div>
              </>
            ) : (
              /* Auth gate — empty state */
              <div className="mt-8 border-l-2 border-vn-500 pl-5 py-2 max-w-xl">
                <p className="text-base text-ink leading-relaxed">
                  Vui lòng đăng nhập để bắt đầu trò chuyện. Chế độ khách chỉ xem, không gửi câu hỏi.
                </p>
                <button
                  onClick={() => setShowAuthModal(true)}
                  className="btn-ink mt-4"
                >
                  <LogIn className="w-4 h-4" />
                  Đăng nhập để trò chuyện
                </button>
              </div>
            )}
          </div>
        ) : (
          <div className="max-w-3xl mx-auto px-4 md:px-8 py-8 space-y-10">
            {messages.map((msg) =>
              msg.role === 'user' ? (
                <div key={msg.id} className="flex justify-end animate-slide-up">
                  <div className="max-w-[80%]">
                    <div className="label-mono mb-1.5 text-right">Bạn</div>
                    <div className="bg-ink text-paper px-4 py-3 rounded-swiss text-sm leading-relaxed">
                      {msg.content}
                    </div>
                  </div>
                </div>
              ) : (
                <div key={msg.id} className="animate-slide-up">
                  <div className="label-mono mb-1.5 flex items-center gap-2">
                    <span className="w-1.5 h-1.5 bg-vn-500 rounded-full" />
                    Trợ lý Pháp Luật
                  </div>

                  <div className="border-l-2 border-vn-500 pl-5 space-y-4">
                    <ReasoningTrace traceSteps={msg.traceSteps} isStreaming={msg.isStreaming} />

                    {msg.content && (
                      <div className="prose-swiss max-w-none">
                        <ReactMarkdown
                          remarkPlugins={[remarkGfm]}
                          rehypePlugins={[rehypeKatex]}
                          components={{
                            a: ({ href, children }) => {
                              if (href?.startsWith('#cite-')) {
                                const idx = parseInt(href.replace('#cite-', ''), 10) - 1;
                                const src = msg.sources?.[idx];
                                return (
                                  <button
                                    type="button"
                                    onClick={() => src && onSelectSource && onSelectSource(src)}
                                    className="inline-flex items-center justify-center font-mono text-[11px] font-bold text-vn-600 dark:text-vn-400 hover:underline px-1 py-0.5 mx-0.5 bg-vn-500/10 hover:bg-vn-500/20 rounded border border-vn-500/30 transition-all cursor-pointer"
                                    title={src?.document_title || src?.title || `Nguồn #${idx + 1}`}
                                  >
                                    {children}
                                  </button>
                                );
                              }
                              return (
                                <a href={href} target="_blank" rel="noreferrer" className="text-vn-600 dark:text-vn-400 hover:underline">
                                  {children}
                                </a>
                              );
                            },
                          }}
                        >
                          {formatContentWithCitations(msg.content)}
                        </ReactMarkdown>
                      </div>
                    )}

                    <CitationDrawer
                      sources={msg.sources}
                      onSelectSource={onSelectSource}
                    />

                    <div className="flex items-center justify-between pt-3 border-t border-rule text-xs text-muted">
                      <div className="flex items-center gap-2">
                        {msg.route && (
                          <span className="font-mono text-[10px] uppercase tracking-label px-2 py-0.5 border border-rule text-ink">
                            Route · {msg.route}
                          </span>
                        )}
                        {msg.cached && (
                          <span className="font-mono text-[10px] uppercase tracking-label px-2 py-0.5 bg-vn-50 text-vn-700 border border-vn-100">
                            Cache HIT
                          </span>
                        )}
                      </div>

                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => handleCopy(msg.id, msg.content)}
                          title="Sao chép"
                          className="p-1.5 text-faint hover:text-ink hover:bg-paper-tint rounded-swiss transition-colors"
                        >
                          {copiedId === msg.id ? (
                            <Check className="w-4 h-4 text-vn-600" />
                          ) : (
                            <Copy className="w-4 h-4" />
                          )}
                        </button>
                        <button
                          onClick={() => handleRating(msg, 'good')}
                          title="Tốt"
                          className={`p-1.5 rounded-swiss transition-colors ${
                            msg.rating === 'good'
                              ? 'text-vn-600 bg-vn-50'
                              : 'text-faint hover:text-ink hover:bg-paper-tint'
                          }`}
                        >
                          <ThumbsUp className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleRating(msg, 'bad')}
                          title="Chưa chính xác"
                          className={`p-1.5 rounded-swiss transition-colors ${
                            msg.rating === 'bad'
                              ? 'text-vn-600 bg-vn-50'
                              : 'text-faint hover:text-ink hover:bg-paper-tint'
                          }`}
                        >
                          <ThumbsDown className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )
            )}
            <div ref={messagesEndRef} />
          </div>
        )}
      </div>

      {/* Composer */}
      <div className="border-t border-ink bg-paper-dim">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto px-4 md:px-8 py-4">
          {isAuthenticated ? (
            <div className="relative">
              <textarea
                value={inputText}
                onChange={(e) => setInputText(e.target.value)}
                onKeyDown={handleKeyDown}
                placeholder="Nhập câu hỏi pháp lý hoặc tra cứu điều luật…"
                rows={2}
                className="w-full bg-paper border border-rule text-ink text-sm placeholder-faint px-4 py-3 pr-14 rounded-swiss resize-none focus:outline-none focus:border-ink transition-colors"
              />
              <button
                type="submit"
                disabled={!inputText.trim() || isLoading}
                className="absolute right-2.5 bottom-2.5 p-2.5 bg-ink text-paper rounded-swiss hover:bg-vn-600 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                aria-label="Gửi"
              >
                {isLoading ? (
                  <div className="w-4 h-4 border-2 border-paper/30 border-t-paper rounded-full animate-spin" />
                ) : (
                  <ArrowUp className="w-4 h-4" />
                )}
              </button>
            </div>
          ) : (
            /* Locked composer — auth gate */
            <button
              type="button"
              onClick={() => setShowAuthModal(true)}
              className="w-full flex items-center gap-3 bg-paper border border-rule px-4 py-3.5 rounded-swiss text-left hover:border-ink transition-colors"
            >
              <Lock className="w-4 h-4 text-vn-500 shrink-0" />
              <span className="text-sm text-muted flex-1">Đăng nhập để gửi câu hỏi…</span>
              <LogIn className="w-4 h-4 text-ink shrink-0" />
            </button>
          )}

          <div className="flex items-center justify-between mt-2.5">
            <span className="label-mono flex items-center gap-1.5">
              <CornerDownLeft className="w-3 h-3" />
              Enter gửi · Shift+Enter xuống dòng
            </span>
            <p className="text-[11px] text-faint hidden sm:block">
              Tư vấn AI mang tính tham khảo — hỏi luật sư cho vụ việc thực tế.
            </p>
          </div>
        </form>
      </div>
    </div>
  );
};