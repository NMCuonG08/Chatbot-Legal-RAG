import React from 'react';
import { useAuth } from '../context/AuthContext';
import { Menu, Scale, Shield, LogIn } from 'lucide-react';

interface HeaderProps {
  onToggleSidebar: () => void;
  selectedVariant: string;
  activeTab: 'chat' | 'admin';
  setActiveTab: (tab: 'chat' | 'admin') => void;
}

export const Header: React.FC<HeaderProps> = ({
  onToggleSidebar,
  selectedVariant,
  activeTab,
  setActiveTab,
}) => {
  const { user, isAuthenticated, setShowAuthModal } = useAuth();

  return (
    <header className="h-14 glass-panel border-b border-slate-800/80 px-4 flex items-center justify-between z-30 shrink-0">
      <div className="flex items-center gap-3">
        <button
          onClick={onToggleSidebar}
          className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/60 lg:hidden transition"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex items-center gap-2">
          <Scale className="w-5 h-5 text-legal-400" />
          <span className="font-bold text-white text-sm hidden sm:inline">Pháp Luật AI</span>
          <span className="text-[11px] font-mono px-2 py-0.5 rounded bg-slate-900 text-slate-400 border border-slate-800">
            {selectedVariant}
          </span>
        </div>
      </div>

      <div className="flex items-center gap-2">
        {(user?.role === 'admin' || user?.role === 'lawyer') && (
          <button
            onClick={() => setActiveTab(activeTab === 'chat' ? 'admin' : 'chat')}
            className={`px-3 py-1.5 rounded-lg text-xs font-semibold flex items-center gap-1.5 border transition ${
              activeTab === 'admin'
                ? 'bg-legal-600 text-white border-legal-500 shadow-md shadow-legal-600/20'
                : 'glass-panel text-slate-300 hover:text-white border-slate-800'
            }`}
          >
            <Shield className="w-3.5 h-3.5" />
            {activeTab === 'chat' ? 'Admin Board' : 'Về Trò Chuyện'}
          </button>
        )}

        {isAuthenticated && user ? (
          <div className="flex items-center gap-2 px-2.5 py-1 rounded-xl bg-slate-900 border border-slate-800 text-xs">
            <div className="w-6 h-6 rounded-full bg-legal-600/30 border border-legal-500/40 flex items-center justify-center text-legal-300 font-bold text-[10px]">
              {user.username.charAt(0).toUpperCase()}
            </div>
            <span className="text-slate-200 font-medium hidden sm:inline">{user.username}</span>
          </div>
        ) : (
          <button
            onClick={() => setShowAuthModal(true)}
            className="px-3 py-1.5 rounded-lg bg-legal-600 hover:bg-legal-500 text-white font-medium text-xs flex items-center gap-1.5 shadow-md shadow-legal-600/20 transition"
          >
            <LogIn className="w-3.5 h-3.5" />
            Đăng nhập
          </button>
        )}
      </div>
    </header>
  );
};
