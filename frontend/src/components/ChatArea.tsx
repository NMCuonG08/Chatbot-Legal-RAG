import React, { useState, useRef, useEffect } from 'react';
import { ChatMessage } from '../types';
import { ReasoningTrace } from './ReasoningTrace';
import { CitationDrawer } from './CitationDrawer';
import { sendFeedbackApi } from '../services/api';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import rehypeKatex from 'rehype-katex';
import {
  Scale,
  Send,
  User as UserIcon,
  Copy,
  Check,
  ThumbsUp,
  ThumbsDown,
  BookOpen,
  FileText,
  Briefcase,
} from 'lucide-react';

interface ChatAreaProps {
  messages: ChatMessage[];
  onSendMessage: (text: string) => void;
  isLoading: boolean;
  userId: string;
}

export const ChatArea: React.FC<ChatAreaProps> = ({ messages, onSendMessage, isLoading, userId }) => {
  const [inputText, setInputText] = useState<string>('');
  const [copiedId, setCopiedId] = useState<string | null>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const handleSubmit = (e?: React.FormEvent) => {
    if (e) e.preventDefault();
    if (!inputText.trim() || isLoading) return;
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

  const samplePrompts = [
    {
      icon: <FileText className="w-5 h-5 text-legal-400" />,
      title: 'Điều kiện chuyển nhượng đất',
      subtitle: 'Theo quy định mới nhất của Luật Đất đai 2024?',
      prompt: 'Cho tôi biết điều kiện để chuyển nhượng quyền sử dụng đất theo quy định của Luật Đất đai 2024?',
    },
    {
      icon: <Briefcase className="w-5 h-5 text-sky-400" />,
      title: 'Thời giờ làm việc & làm thêm',
      subtitle: 'Bộ luật Lao động quy định như thế nào?',
      prompt: 'Quy định về thời giờ làm việc bình thường và giới hạn làm thêm giờ theo Bộ luật Lao động 2019?',
    },
    {
      icon: <Scale className="w-5 h-5 text-amber-400" />,
      title: 'Đăng ký doanh nghiệp tư nhân',
      subtitle: 'Hồ sơ & thủ tục theo Luật Doanh nghiệp',
      prompt: 'Hồ sơ và thủ tục thành lập doanh nghiệp tư nhân theo quy định của Luật Doanh nghiệp bao gồm những gì?',
    },
    {
      icon: <BookOpen className="w-5 h-5 text-emerald-400" />,
      title: 'Tra cứu Án lệ tranh chấp',
      subtitle: 'Tranh chấp hợp đồng mua bán tài sản',
      prompt: 'Hãy tóm tắt các án lệ liên quan đến tranh chấp hợp đồng mua bán tài sản và phương thức giải quyết?',
    },
  ];

  return (
    <div className="flex-1 flex flex-col h-full overflow-hidden bg-slate-950 relative">
      {/* Scrollable Messages Container */}
      <div className="flex-1 overflow-y-auto px-4 py-6 md:px-8 space-y-6">
        {messages.length === 0 ? (
          /* Empty Welcome Hero */
          <div className="max-w-3xl mx-auto py-12 text-center space-y-8 animate-fade-in">
            <div className="inline-flex items-center justify-center w-20 h-20 rounded-3xl bg-gradient-to-tr from-legal-600 via-legal-500 to-sky-400 p-0.5 shadow-2xl shadow-legal-500/30">
              <div className="w-full h-full bg-slate-950 rounded-[22px] flex items-center justify-center">
                <Scale className="w-10 h-10 text-legal-400" />
              </div>
            </div>

            <div>
              <h2 className="text-3xl font-extrabold text-white tracking-tight sm:text-4xl">
                Trợ Lý AI Pháp Luật Việt Nam
              </h2>
              <p className="mt-3 text-base text-slate-400 max-w-xl mx-auto leading-relaxed">
                Hệ thống tra cứu, tư vấn văn bản quy phạm pháp luật & án lệ thông minh với đồ thị tri thức RAG Agentic
              </p>
            </div>

            {/* Quick Prompt Cards Grid */}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 max-w-2xl mx-auto pt-4 text-left">
              {samplePrompts.map((p, idx) => (
                <button
                  key={idx}
                  onClick={() => onSendMessage(p.prompt)}
                  className="p-4 rounded-2xl glass-panel glass-panel-hover text-left group border border-slate-800/80 hover:border-legal-500/40 transition duration-200"
                >
                  <div className="flex items-center gap-3 mb-2">
                    <div className="p-2 rounded-xl bg-slate-900 border border-slate-800 group-hover:scale-105 transition">
                      {p.icon}
                    </div>
                    <span className="font-semibold text-slate-200 text-sm group-hover:text-white">
                      {p.title}
                    </span>
                  </div>
                  <p className="text-xs text-slate-400 leading-snug">{p.subtitle}</p>
                </button>
              ))}
            </div>
          </div>
        ) : (
          /* Messages List */
          messages.map((msg) => (
            <div key={msg.id} className="max-w-3xl mx-auto space-y-3">
              {msg.role === 'user' ? (
                /* User Message Bubble */
                <div className="flex items-start justify-end gap-3">
                  <div className="p-4 rounded-2xl bg-legal-600 text-white shadow-lg shadow-legal-600/20 max-w-[85%] text-sm leading-relaxed">
                    {msg.content}
                  </div>
                  <div className="w-9 h-9 rounded-full bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300 font-bold text-xs shrink-0">
                    <UserIcon className="w-5 h-5" />
                  </div>
                </div>
              ) : (
                /* Assistant Message Bubble */
                <div className="flex items-start gap-3">
                  <div className="w-9 h-9 rounded-full bg-gradient-to-tr from-legal-600 to-legal-400 flex items-center justify-center text-white shadow-md shadow-legal-500/20 shrink-0 mt-1">
                    <Scale className="w-5 h-5" />
                  </div>

                  <div className="flex-1 glass-panel p-5 rounded-2xl border border-slate-800/90 text-slate-200 text-sm leading-relaxed space-y-3 max-w-[90%]">
                    {/* Reasoning Trace Accordion */}
                    <ReasoningTrace traceSteps={msg.traceSteps} isStreaming={msg.isStreaming} />

                    {/* Content Markdown */}
                    <div className="prose prose-invert prose-slate max-w-none prose-p:leading-relaxed prose-headings:text-legal-300 prose-a:text-legal-400">
                      <ReactMarkdown remarkPlugins={[remarkGfm]} rehypePlugins={[rehypeKatex]}>
                        {msg.content}
                      </ReactMarkdown>
                    </div>

                    {/* Citations Drawer */}
                    <CitationDrawer sources={msg.sources} />

                    {/* Bottom Message Action Bar */}
                    <div className="flex items-center justify-between pt-3 border-t border-slate-800/60 text-xs text-slate-400">
                      <div className="flex items-center gap-2">
                        {msg.route && (
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-slate-900 text-legal-400 border border-slate-800">
                            Route: {msg.route}
                          </span>
                        )}
                        {msg.cached && (
                          <span className="text-[10px] font-mono px-2 py-0.5 rounded bg-emerald-950 text-emerald-400 border border-emerald-800 flex items-center gap-1">
                            ⚡ Semantic Cache HIT
                          </span>
                        )}
                      </div>

                      <div className="flex items-center gap-1">
                        <button
                          onClick={() => handleCopy(msg.id, msg.content)}
                          title="Sao chép câu trả lời"
                          className="p-1.5 rounded-lg hover:bg-slate-800 text-slate-400 hover:text-white transition"
                        >
                          {copiedId === msg.id ? <Check className="w-4 h-4 text-emerald-400" /> : <Copy className="w-4 h-4" />}
                        </button>
                        <button
                          onClick={() => handleRating(msg, 'good')}
                          title="Câu trả lời tốt"
                          className={`p-1.5 rounded-lg hover:bg-slate-800 transition ${
                            msg.rating === 'good' ? 'text-emerald-400 bg-slate-800' : 'text-slate-400'
                          }`}
                        >
                          <ThumbsUp className="w-4 h-4" />
                        </button>
                        <button
                          onClick={() => handleRating(msg, 'bad')}
                          title="Chưa chính xác"
                          className={`p-1.5 rounded-lg hover:bg-slate-800 transition ${
                            msg.rating === 'bad' ? 'text-rose-400 bg-slate-800' : 'text-slate-400'
                          }`}
                        >
                          <ThumbsDown className="w-4 h-4" />
                        </button>
                      </div>
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))
        )}
        <div ref={messagesEndRef} />
      </div>

      {/* Floating Bottom Input Area */}
      <div className="p-4 md:px-8 border-t border-slate-800/80 bg-slate-950/90 backdrop-blur-md">
        <form onSubmit={handleSubmit} className="max-w-3xl mx-auto relative">
          <textarea
            value={inputText}
            onChange={(e) => setInputText(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Nhập câu hỏi pháp lý hoặc tra cứu điều luật tại đây... (Enter để gửi)"
            rows={2}
            className="w-full pl-4 pr-14 py-3 rounded-2xl glass-input text-white text-sm placeholder-slate-500 resize-none focus:ring-2 focus:ring-legal-500/50"
          />

          <button
            type="submit"
            disabled={!inputText.trim() || isLoading}
            className="absolute right-3 bottom-3.5 p-2.5 rounded-xl bg-gradient-to-r from-legal-600 to-legal-500 hover:from-legal-500 hover:to-legal-400 text-white shadow-lg shadow-legal-600/30 transition disabled:opacity-40"
          >
            {isLoading ? (
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : (
              <Send className="w-4 h-4" />
            )}
          </button>
        </form>

        <p className="text-[11px] text-slate-500 text-center mt-2">
          Thông tin tư vấn AI mang tính chất tham khảo trích dẫn văn bản pháp luật. Vui lòng hỏi luật sư đối với vụ việc thực tế.
        </p>
      </div>
    </div>
  );
};
