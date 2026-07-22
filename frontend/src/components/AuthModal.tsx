import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { Lock, User as UserIcon, X, AlertCircle, LogIn, UserPlus, ShieldCheck, Scale } from 'lucide-react';

export const AuthModal: React.FC = () => {
  const { showAuthModal, setShowAuthModal, login, register } = useAuth();
  const [isLoginTab, setIsLoginTab] = useState<boolean>(true);
  const [username, setUsername] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [role, setRole] = useState<string>('user');
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

  if (!showAuthModal) return null;

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError(null);

    if (!username.trim() || !password.trim()) {
      setError('Vui lòng nhập tên đăng nhập và mật khẩu.');
      return;
    }

    setIsSubmitting(true);
    try {
      if (isLoginTab) {
        await login(username.trim(), password);
      } else {
        await register(username.trim(), password, role);
      }
    } catch (err: any) {
      setError(err.message || 'Đã xảy ra lỗi, vui lòng thử lại.');
    } finally {
      setIsSubmitting(false);
    }
  };

  return (
    <div
      onClick={() => setShowAuthModal(false)}
      className="fixed inset-0 z-50 flex items-center justify-center p-4 backdrop-blur-md animate-fade-in"
      style={{ backgroundColor: 'rgba(0, 0, 0, 0.75)' }}
    >
      <div
        onClick={(e) => e.stopPropagation()}
        className="relative w-full max-w-md border border-rule rounded-2xl p-6 sm:p-8 shadow-2xl transition-all"
        style={{ backgroundColor: 'rgb(var(--c-paper))' }}
      >
        {/* Close Button */}
        <button
          onClick={() => setShowAuthModal(false)}
          className="absolute top-4 right-4 p-2 text-muted hover:text-ink hover:bg-paper-tint rounded-lg transition-colors"
          aria-label="Đóng"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header Branding */}
        <div className="flex items-center gap-3 mb-6">
          <div
            className="w-10 h-10 border border-vn-500/20 flex items-center justify-center rounded-xl shrink-0"
            style={{ backgroundColor: '#FFF1F3', color: '#D7263D' }}
          >
            <Scale className="w-5 h-5" />
          </div>
          <div>
            <h2 className="text-xl font-bold text-ink leading-tight">
              {isLoginTab ? 'Đăng nhập' : 'Tạo tài khoản mới'}
            </h2>
            <p className="text-xs text-muted mt-0.5">Trợ Lý Pháp Luật AI · Tra cứu & Trí nhớ RAG</p>
          </div>
        </div>

        {/* Segmented Tab Switcher */}
        <div
          className="flex p-1 border border-rule rounded-xl mb-6"
          style={{ backgroundColor: 'rgb(var(--c-paper-dim))' }}
        >
          <button
            type="button"
            onClick={() => { setIsLoginTab(true); setError(null); }}
            className={`flex-1 flex items-center justify-center gap-2 py-2.5 text-xs font-semibold rounded-lg transition-all ${
              isLoginTab
                ? 'text-ink shadow-sm border border-rule/50 font-bold'
                : 'text-muted hover:text-ink font-medium'
            }`}
            style={isLoginTab ? { backgroundColor: 'rgb(var(--c-paper))' } : undefined}
          >
            <LogIn className="w-4 h-4 text-vn-500" />
            Đăng nhập
          </button>
          <button
            type="button"
            onClick={() => { setIsLoginTab(false); setError(null); }}
            className={`flex-1 flex items-center justify-center gap-2 py-2.5 text-xs font-semibold rounded-lg transition-all ${
              !isLoginTab
                ? 'text-ink shadow-sm border border-rule/50 font-bold'
                : 'text-muted hover:text-ink font-medium'
            }`}
            style={!isLoginTab ? { backgroundColor: 'rgb(var(--c-paper))' } : undefined}
          >
            <UserPlus className="w-4 h-4 text-vn-500" />
            Đăng ký
          </button>
        </div>

        {/* Error Alert */}
        {error && (
          <div
            className="flex items-start gap-2.5 p-3.5 mb-5 border rounded-xl text-xs"
            style={{ backgroundColor: '#FFF1F3', borderColor: '#FFE0E5', color: '#9B1233' }}
          >
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" style={{ color: '#D7263D' }} />
            <span>{error}</span>
          </div>
        )}

        {/* Form Inputs */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-muted mb-1.5">
              Tên đăng nhập
            </label>
            <div className="relative flex items-center">
              <UserIcon className="absolute left-3.5 w-4 h-4 text-muted pointer-events-none" />
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Nhập tên đăng nhập"
                className="w-full border border-rule text-ink placeholder:text-faint text-sm rounded-xl pl-10 pr-4 py-2.5 focus:outline-none focus:border-vn-500 transition-all"
                style={{ backgroundColor: 'rgb(var(--c-paper-dim))', color: 'rgb(var(--c-ink))' }}
                required
              />
            </div>
          </div>

          <div>
            <label className="block text-xs font-semibold uppercase tracking-wider text-muted mb-1.5">
              Mật khẩu
            </label>
            <div className="relative flex items-center">
              <Lock className="absolute left-3.5 w-4 h-4 text-muted pointer-events-none" />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="w-full border border-rule text-ink placeholder:text-faint text-sm rounded-xl pl-10 pr-4 py-2.5 focus:outline-none focus:border-vn-500 transition-all"
                style={{ backgroundColor: 'rgb(var(--c-paper-dim))', color: 'rgb(var(--c-ink))' }}
                required
              />
            </div>
          </div>

          {!isLoginTab && (
            <div>
              <label className="block text-xs font-semibold uppercase tracking-wider text-muted mb-1.5">
                Vai trò
              </label>
              <div className="relative flex items-center">
                <ShieldCheck className="absolute left-3.5 w-4 h-4 text-muted pointer-events-none" />
                <select
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                  className="w-full border border-rule text-ink text-sm rounded-xl pl-10 pr-4 py-2.5 focus:outline-none focus:border-vn-500 transition-all cursor-pointer"
                  style={{ backgroundColor: 'rgb(var(--c-paper-dim))', color: 'rgb(var(--c-ink))' }}
                >
                  <option value="user" style={{ backgroundColor: 'rgb(var(--c-paper))', color: 'rgb(var(--c-ink))' }}>
                    Người dùng (User)
                  </option>
                  <option value="lawyer" style={{ backgroundColor: 'rgb(var(--c-paper))', color: 'rgb(var(--c-ink))' }}>
                    Luật sư / Thư ký pháp lý (Lawyer)
                  </option>
                </select>
              </div>
            </div>
          )}

          {/* Submit Button with GUARANTEED Red Background & White Text */}
          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full mt-2 py-3 px-4 font-semibold text-sm rounded-xl shadow-md hover:opacity-90 active:opacity-100 transition-all flex items-center justify-center gap-2 cursor-pointer disabled:opacity-50"
            style={{ backgroundColor: '#D7263D', color: '#FFFFFF' }}
          >
            {isSubmitting ? (
              <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : isLoginTab ? (
              'Đăng nhập'
            ) : (
              'Tạo tài khoản'
            )}
          </button>
        </form>

        {/* Footer Guest Link */}
        <div className="mt-6 pt-4 border-t border-rule text-center">
          <button
            onClick={() => setShowAuthModal(false)}
            className="text-xs text-muted hover:text-ink transition-colors font-medium"
          >
            Tiếp tục với chế độ Khách (chỉ xem)
          </button>
        </div>
      </div>
    </div>
  );
};