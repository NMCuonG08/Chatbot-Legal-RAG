import React from 'react';
import { useAuth } from '../context/AuthContext';
import { HealthWidget } from './HealthWidget';
import {
  Scale,
  Plus,
  MessageSquare,
  Trash2,
  LogOut,
  LogIn,
  Shield,
  Sliders,
} from 'lucide-react';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  onNewChat: () => void;
  history: any[];
  onSelectHistoryItem: (item: any) => void;
  onDeleteSingleHistory?: (conversationId: string) => void;
  onWipeHistory: () => void;
  selectedVariant: string;
  onSelectVariant: (variant: string) => void;
  activeTab: 'chat' | 'admin';
  setActiveTab: (tab: 'chat' | 'admin') => void;
  selectedHistoryId?: string | null;
}

const pad2 = (n: number) => String(n).padStart(2, '0');

export const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  onClose,
  onNewChat,
  history,
  onSelectHistoryItem,
  onDeleteSingleHistory,
  onWipeHistory,
  selectedVariant,
  onSelectVariant,
  activeTab,
  setActiveTab,
  selectedHistoryId,
}) => {
  const { user, isAuthenticated, logout, setShowAuthModal } = useAuth();
  const canAccessAdmin = isAuthenticated && (user?.role === 'admin' || user?.role === 'lawyer');

  return (
    <>
      {isOpen && <div onClick={onClose} className="fixed inset-0 z-40 bg-black/50 backdrop-blur-sm lg:hidden" />}

      <aside
        className={`fixed top-0 left-0 bottom-0 z-40 w-72 bg-paper border-r border-ink flex flex-col transition-transform duration-300 ease-in-out lg:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Branding */}
        <div className="px-5 pt-5 pb-4 border-b border-rule">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 bg-ink flex items-center justify-center rounded-swiss">
              <Scale className="w-4 h-4 text-vn-500" strokeWidth={2.25} />
            </div>
            <div className="leading-tight">
              <h1 className="font-semibold text-sm tracking-display uppercase">Pháp Luật VN</h1>
              <span className="label-mono">RAG Agentic · v2.0</span>
            </div>
          </div>
        </div>

        {/* Primary action */}
        <div className="px-4 pt-4 pb-3 space-y-2.5">
          <button
            onClick={() => {
              onNewChat();
              setActiveTab('chat');
            }}
            className="btn-ink w-full"
          >
            <Plus className="w-4 h-4" />
            Cuộc trò chuyện mới
          </button>

          {canAccessAdmin && (
            <div className="grid grid-cols-2 gap-1.5 pt-1">
              <button
                onClick={() => setActiveTab('chat')}
                className={`inline-flex items-center justify-center gap-1.5 py-2 px-2 rounded-swiss text-[11px] font-mono uppercase tracking-label transition-colors border ${
                  activeTab === 'chat'
                    ? 'bg-ink text-paper border-ink'
                    : 'bg-paper text-muted border-rule hover:border-ink hover:text-ink'
                }`}
              >
                <MessageSquare className="w-3.5 h-3.5" />
                Chat
              </button>
              <button
                onClick={() => setActiveTab('admin')}
                className={`inline-flex items-center justify-center gap-1.5 py-2 px-2 rounded-swiss text-[11px] font-mono uppercase tracking-label transition-colors border ${
                  activeTab === 'admin'
                    ? 'bg-ink text-paper border-ink'
                    : 'bg-paper text-muted border-rule hover:border-ink hover:text-ink'
                }`}
              >
                <Shield className="w-3.5 h-3.5" />
                Admin
              </button>
            </div>
          )}
        </div>

        {/* Model selector */}
        <div className="px-4 py-3 border-t border-rule">
          <label className="label-mono mb-1.5 flex items-center gap-1.5">
            <Sliders className="w-3 h-3 text-vn-500" />
            Mô hình LLM
          </label>
          <select
            value={selectedVariant}
            onChange={(e) => onSelectVariant(e.target.value)}
            className="field !py-2 !text-xs cursor-pointer"
          >
            <option value="gemma4:31b-cloud">Gemma 4 31B — Tiêu chuẩn</option>
            <option value="gemma4:cloud">Gemma 4 Fast — Siêu tốc</option>
            <option value="llama-3.1-8b-instant">Llama 3.1 8B — Instant</option>
            <option value="deep-reasoning">Deep Reasoning — GraphRAG</option>
          </select>
        </div>

        {/* History list — numbered */}
        <div className="flex-1 overflow-y-auto px-4 py-3 border-t border-rule">
          <div className="flex items-center justify-between mb-2">
            <span className="label-mono">Lịch sử · {history.length}</span>
            {history.length > 0 && (
              <button
                onClick={onWipeHistory}
                title="Xóa lịch sử"
                className="p-1 text-faint hover:text-vn-600 transition-colors"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          {history.length === 0 ? (
            <p className="py-8 text-center text-xs text-faint">Chưa có câu hỏi nào.</p>
          ) : (
            <ol className="space-y-1">
              {history.map((item, idx) => {
                const itemId = item.id || item.conversation_id;
                const isSelected = Boolean(selectedHistoryId && selectedHistoryId === itemId);
                return (
                  <li key={itemId || idx}>
                    <div
                      onClick={() => onSelectHistoryItem(item)}
                      className={`w-full flex items-center justify-between py-1.5 px-2 text-left text-[13px] transition-all group rounded-swiss border-l-2 cursor-pointer ${
                        isSelected
                          ? 'bg-paper-tint text-vn-600 font-semibold border-vn-500'
                          : 'border-transparent text-ink/80 hover:bg-paper-tint hover:text-ink'
                      }`}
                    >
                      <div className="flex items-center gap-2 min-w-0 pr-2">
                        <span className={`font-mono text-[10px] tabular-nums shrink-0 ${isSelected ? 'text-vn-600 font-bold' : 'text-faint'}`}>
                          {pad2(idx + 1)}
                        </span>
                        <span className="truncate">
                          {item.question || item.content || item.message?.question || `Hội thoại ${idx + 1}`}
                        </span>
                      </div>
                      
                      {onDeleteSingleHistory && (
                        <button
                          type="button"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteSingleHistory(itemId);
                          }}
                          title="Xóa cuộc trò chuyện này"
                          className="opacity-0 group-hover:opacity-100 p-1 text-faint hover:text-red-500 hover:bg-paper rounded transition-all shrink-0"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </button>
                      )}
                    </div>
                  </li>
                );
              })}
            </ol>
          )}
        </div>

        {/* Health */}
        <div className="px-4 py-2 border-t border-rule">
          <HealthWidget />
        </div>

        {/* User footer */}
        <div className="px-4 py-3 border-t border-ink bg-paper-dim">
          {isAuthenticated && user ? (
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5 min-w-0">
                <div className="w-7 h-7 bg-ink text-paper flex items-center justify-center font-semibold text-[11px] rounded-swiss shrink-0">
                  {user.username.charAt(0).toUpperCase()}
                </div>
                <div className="truncate leading-tight">
                  <p className="text-sm font-medium text-ink truncate">{user.username}</p>
                  <span className="font-mono text-[10px] uppercase tracking-label text-vn-600">
                    {user.role}
                  </span>
                </div>
              </div>
              <button
                onClick={logout}
                title="Đăng xuất"
                className="p-1.5 text-faint hover:text-vn-600 transition-colors"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <button
              onClick={() => setShowAuthModal(true)}
              className="btn-outline w-full !py-2 !text-xs"
            >
              <LogIn className="w-4 h-4" />
              Đăng nhập / Đăng ký
            </button>
          )}
        </div>
      </aside>
    </>
  );
};