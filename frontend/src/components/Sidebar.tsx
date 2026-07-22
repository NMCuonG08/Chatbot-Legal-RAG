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
  User as UserIcon,
  Shield,
  Sparkles,
  Sliders,
} from 'lucide-react';

interface SidebarProps {
  isOpen: boolean;
  onClose: () => void;
  onNewChat: () => void;
  history: any[];
  onSelectHistoryItem: (item: any) => void;
  onWipeHistory: () => void;
  selectedVariant: string;
  onSelectVariant: (variant: string) => void;
  activeTab: 'chat' | 'admin';
  setActiveTab: (tab: 'chat' | 'admin') => void;
}

export const Sidebar: React.FC<SidebarProps> = ({
  isOpen,
  onClose,
  onNewChat,
  history,
  onSelectHistoryItem,
  onWipeHistory,
  selectedVariant,
  onSelectVariant,
  activeTab,
  setActiveTab,
}) => {
  const { user, isAuthenticated, logout, setShowAuthModal } = useAuth();

  return (
    <>
      {/* Mobile Backdrop */}
      {isOpen && (
        <div
          onClick={onClose}
          className="fixed inset-0 z-40 bg-slate-950/80 backdrop-blur-sm lg:hidden"
        />
      )}

      {/* Sidebar Container */}
      <aside
        className={`fixed top-0 left-0 bottom-0 z-40 w-72 glass-panel border-r border-slate-800 flex flex-col transition-transform duration-300 ease-in-out lg:translate-x-0 ${
          isOpen ? 'translate-x-0' : '-translate-x-full'
        }`}
      >
        {/* Header Branding */}
        <div className="p-4 border-b border-slate-800/80 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-legal-600 to-legal-400 flex items-center justify-center shadow-md shadow-legal-500/20">
              <Scale className="w-6 h-6 text-white" />
            </div>
            <div>
              <h1 className="font-bold text-white text-base leading-tight tracking-tight">Legal AI Assistant</h1>
              <div className="flex items-center gap-1.5 mt-0.5">
                <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse" />
                <span className="text-[11px] font-medium text-slate-400">RAG Agentic v2.0</span>
              </div>
            </div>
          </div>
        </div>

        {/* Action Buttons */}
        <div className="p-4 space-y-2">
          <button
            onClick={() => {
              onNewChat();
              setActiveTab('chat');
            }}
            className="w-full py-2.5 px-4 rounded-xl bg-gradient-to-r from-legal-600 to-legal-500 hover:from-legal-500 hover:to-legal-400 text-white font-medium text-sm shadow-md shadow-legal-600/20 flex items-center justify-center gap-2 transition duration-150"
          >
            <Plus className="w-4 h-4" />
            Cuộc trò chuyện mới
          </button>

          {(user?.role === 'admin' || user?.role === 'lawyer') && (
            <div className="grid grid-cols-2 gap-1.5 pt-2">
              <button
                onClick={() => setActiveTab('chat')}
                className={`py-2 px-3 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 transition ${
                  activeTab === 'chat'
                    ? 'bg-slate-800 text-legal-400 border border-legal-500/30'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                }`}
              >
                <MessageSquare className="w-3.5 h-3.5" />
                Hỏi Đáp AI
              </button>
              <button
                onClick={() => setActiveTab('admin')}
                className={`py-2 px-3 rounded-lg text-xs font-medium flex items-center justify-center gap-1.5 transition ${
                  activeTab === 'admin'
                    ? 'bg-slate-800 text-legal-400 border border-legal-500/30'
                    : 'text-slate-400 hover:text-white hover:bg-slate-800/50'
                }`}
              >
                <Shield className="w-3.5 h-3.5" />
                Quản Trị Admin
              </button>
            </div>
          )}
        </div>

        {/* Model Selector Section */}
        <div className="px-4 py-2 border-t border-slate-800/60">
          <label className="block text-[11px] font-semibold text-slate-400 uppercase tracking-wider mb-1.5 flex items-center gap-1.5">
            <Sliders className="w-3 h-3 text-legal-400" />
            Mô hình LLM Agent
          </label>
          <select
            value={selectedVariant}
            onChange={(e) => onSelectVariant(e.target.value)}
            className="w-full px-3 py-2 rounded-lg glass-input text-xs text-slate-200 bg-slate-900 cursor-pointer border border-slate-800 focus:border-legal-500"
          >
            <option value="gemma4:31b-cloud">Gemma 4 31B (Tiêu chuẩn)</option>
            <option value="gemma4:cloud">Gemma 4 Fast (Siêu tốc)</option>
            <option value="llama-3.1-8b-instant">Llama 3.1 8B (Instant)</option>
            <option value="deep-reasoning">Deep Reasoning (GraphRAG)</option>
          </select>
        </div>

        {/* History List */}
        <div className="flex-1 overflow-y-auto px-4 py-2 space-y-1">
          <div className="flex items-center justify-between mb-2">
            <span className="text-[11px] font-semibold text-slate-400 uppercase tracking-wider">
              Lịch sử trò chuyện
            </span>
            {history.length > 0 && (
              <button
                onClick={onWipeHistory}
                title="Xóa lịch sử"
                className="p-1 rounded text-slate-500 hover:text-rose-400 hover:bg-slate-800/50 transition"
              >
                <Trash2 className="w-3.5 h-3.5" />
              </button>
            )}
          </div>

          {history.length === 0 ? (
            <div className="text-center py-6 text-slate-500 text-xs font-normal">
              Chưa có lịch sử câu hỏi
            </div>
          ) : (
            history.map((item, idx) => (
              <button
                key={idx}
                onClick={() => onSelectHistoryItem(item)}
                className="w-full p-2.5 rounded-lg text-left text-xs text-slate-300 hover:bg-slate-800/60 hover:text-white transition flex items-center gap-2 group truncate"
              >
                <MessageSquare className="w-3.5 h-3.5 text-slate-500 group-hover:text-legal-400 shrink-0" />
                <span className="truncate">{item.question || item.content || `Hội thoại ${idx + 1}`}</span>
              </button>
            ))
          )}
        </div>

        {/* Health Monitoring Widget */}
        <div className="px-4 py-2">
          <HealthWidget />
        </div>

        {/* User Profile Card Footer */}
        <div className="p-4 border-t border-slate-800/80 bg-slate-950/60">
          {isAuthenticated && user ? (
            <div className="flex items-center justify-between">
              <div className="flex items-center gap-2.5 min-w-0">
                <div className="w-8 h-8 rounded-full bg-legal-600/30 border border-legal-500/40 flex items-center justify-center text-legal-300 font-bold text-xs shrink-0">
                  {user.username.charAt(0).toUpperCase()}
                </div>
                <div className="truncate">
                  <p className="text-xs font-semibold text-white truncate">{user.username}</p>
                  <span className="inline-block text-[10px] px-1.5 py-0.2 rounded bg-slate-800 text-legal-400 font-mono capitalize">
                    {user.role}
                  </span>
                </div>
              </div>
              <button
                onClick={logout}
                title="Đăng xuất"
                className="p-2 rounded-lg text-slate-400 hover:text-rose-400 hover:bg-slate-800 transition"
              >
                <LogOut className="w-4 h-4" />
              </button>
            </div>
          ) : (
            <button
              onClick={() => setShowAuthModal(true)}
              className="w-full py-2 px-3 rounded-lg border border-slate-700 bg-slate-900/80 hover:bg-slate-800 text-slate-200 text-xs font-medium flex items-center justify-center gap-2 transition"
            >
              <LogIn className="w-4 h-4 text-legal-400" />
              Đăng nhập / Đăng ký
            </button>
          )}
        </div>
      </aside>
    </>
  );
};
