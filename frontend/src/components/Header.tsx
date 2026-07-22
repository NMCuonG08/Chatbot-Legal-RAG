import React from 'react';
import { useAuth } from '../context/AuthContext';
import { useTheme } from '../context/ThemeContext';
import { Menu, Scale, LogIn, Sun, Moon, Shield } from 'lucide-react';

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
  const { theme, toggleTheme } = useTheme();
  const canAccessAdmin = isAuthenticated && (user?.role === 'admin' || user?.role === 'lawyer');

  return (
    <header className="h-14 bg-paper border-b border-ink px-4 flex items-center justify-between z-30 shrink-0">
      <div className="flex items-center gap-4">
        <button
          onClick={onToggleSidebar}
          className="p-1.5 -ml-1.5 text-ink hover:bg-paper-tint rounded-swiss lg:hidden transition-colors"
          aria-label="Mở menu"
        >
          <Menu className="w-5 h-5" />
        </button>

        <div className="flex items-baseline gap-2.5">
          <Scale className="w-4 h-4 text-vn-500 self-center" strokeWidth={2.25} />
          <span className="font-semibold text-sm tracking-display uppercase">Pháp Luật VN</span>
        </div>

        <span className="hidden sm:block h-4 w-px bg-rule" />

        <span className="hidden sm:inline font-mono text-[11px] text-muted px-2 py-0.5 border border-rule rounded-swiss bg-paper-dim">
          {selectedVariant}
        </span>
      </div>

      <div className="flex items-center gap-3">
        {/* Section tabs */}
        <nav className="hidden sm:flex items-center gap-1 font-mono text-[11px] uppercase tracking-label">
          <button
            onClick={() => setActiveTab('chat')}
            className={`px-2.5 py-1 transition-colors ${
              activeTab === 'chat' ? 'text-ink border-b-2 border-vn-500 -mb-px' : 'text-faint hover:text-ink'
            }`}
          >
            Chat
          </button>
          {canAccessAdmin && (
            <button
              onClick={() => setActiveTab('admin')}
              className={`px-2.5 py-1 transition-colors ${
                activeTab === 'admin' ? 'text-ink border-b-2 border-vn-500 -mb-px' : 'text-faint hover:text-ink'
              }`}
            >
              Admin
            </button>
          )}
        </nav>

        {canAccessAdmin && (
          <button
            onClick={() => setActiveTab(activeTab === 'chat' ? 'admin' : 'chat')}
            className="sm:hidden inline-flex items-center gap-1.5 px-2.5 py-1.5 border border-rule rounded-swiss text-[11px] font-mono uppercase tracking-label text-ink hover:bg-paper-tint transition-colors"
          >
            <Shield className="w-3.5 h-3.5 text-vn-500" />
            {activeTab === 'chat' ? 'Admin' : 'Chat'}
          </button>
        )}

        {/* Theme toggle */}
        <button
          onClick={toggleTheme}
          className="p-1.5 text-ink hover:bg-paper-tint rounded-swiss transition-colors"
          aria-label={theme === 'dark' ? 'Chuyển sáng' : 'Chuyển tối'}
          title={theme === 'dark' ? 'Chế độ sáng' : 'Chế độ tối'}
        >
          {theme === 'dark' ? <Sun className="w-4 h-4" /> : <Moon className="w-4 h-4" />}
        </button>

        {/* Auth / user */}
        {isAuthenticated && user ? (
          <div className="flex items-center gap-2 pl-2 border-l border-rule">
            <div className="w-7 h-7 bg-ink text-paper flex items-center justify-center font-semibold text-[11px] rounded-swiss">
              {user.username.charAt(0).toUpperCase()}
            </div>
            <span className="hidden sm:inline text-sm font-medium text-ink">{user.username}</span>
          </div>
        ) : (
          <button onClick={() => setShowAuthModal(true)} className="btn-ink !py-1.5 !px-3 text-xs">
            <LogIn className="w-3.5 h-3.5" />
            Đăng nhập
          </button>
        )}
      </div>
    </header>
  );
};