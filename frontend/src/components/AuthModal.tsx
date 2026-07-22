import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { Lock, User as UserIcon, X, AlertCircle, LogIn, UserPlus, ShieldCheck } from 'lucide-react';

export const AuthModal: React.FC = () => {
  const { showAuthModal, setShowAuthModal, login, register } = useAuth();
  const [isLoginTab, setIsLoginTab] = useState<boolean>(true);
  const [username, setUsername] = useState<string>('');
  const [password, setPassword] = useState<string>('');
  const [role, setRole] = useState<string>('user');
  const [error, setError] = useState<string | null>(null);
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

  if (!showAuthModal) return null;

  // Minimal validation for testing: non-empty only, no password-length rule
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
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-ink/40 animate-fade-in">
      <div className="relative w-full max-w-md bg-paper border border-ink rounded-swiss p-6 md:p-8">
        <button
          onClick={() => setShowAuthModal(false)}
          className="absolute top-3 right-3 p-1.5 text-faint hover:text-ink hover:bg-paper-tint rounded-swiss transition-colors"
          aria-label="Đóng"
        >
          <X className="w-4 h-4" />
        </button>

        <div className="mb-6">
          <div className="flex items-center gap-3 mb-3">
            <span className="h-px w-8 bg-vn-500" />
            <span className="label-mono text-vn-600">Tài khoản</span>
          </div>
          <h2 className="text-2xl font-semibold tracking-display text-ink">
            {isLoginTab ? 'Đăng nhập' : 'Tạo tài khoản'}
          </h2>
          <p className="text-sm text-muted mt-1.5">Lưu lịch sử trò chuyện & trí nhớ dài hạn.</p>
        </div>

        <div className="grid grid-cols-2 border border-rule rounded-swiss mb-5 overflow-hidden">
          <button
            type="button"
            onClick={() => { setIsLoginTab(true); setError(null); }}
            className={`flex items-center justify-center gap-2 py-2.5 text-sm font-medium transition-colors ${
              isLoginTab ? 'bg-ink text-paper' : 'bg-paper text-muted hover:text-ink'
            }`}
          >
            <LogIn className="w-4 h-4" />
            Đăng nhập
          </button>
          <button
            type="button"
            onClick={() => { setIsLoginTab(false); setError(null); }}
            className={`flex items-center justify-center gap-2 py-2.5 text-sm font-medium transition-colors border-l border-rule ${
              !isLoginTab ? 'bg-ink text-paper' : 'bg-paper text-muted hover:text-ink'
            }`}
          >
            <UserPlus className="w-4 h-4" />
            Đăng ký
          </button>
        </div>

        {error && (
          <div className="flex items-start gap-2.5 p-3 mb-5 bg-vn-50 border-l-2 border-vn-500 text-vn-700 text-sm">
            <AlertCircle className="w-4 h-4 shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="label-mono mb-1.5 block">Tên đăng nhập</label>
            <div className="relative">
              <UserIcon className="absolute left-3 top-3 w-4 h-4 text-faint" />
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="nhập tên đăng nhập"
                className="field !pl-9"
                required
              />
            </div>
          </div>

          <div>
            <label className="label-mono mb-1.5 block">Mật khẩu</label>
            <div className="relative">
              <Lock className="absolute left-3 top-3 w-4 h-4 text-faint" />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="••••••••"
                className="field !pl-9"
                required
              />
            </div>
          </div>

          {!isLoginTab && (
            <div>
              <label className="label-mono mb-1.5 block">Vai trò</label>
              <div className="relative">
                <ShieldCheck className="absolute left-3 top-3 w-4 h-4 text-faint" />
                <select
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                  className="field !pl-9 cursor-pointer"
                >
                  <option value="user">Người dùng (User)</option>
                  <option value="lawyer">Luật sư / Thư ký pháp lý (Lawyer)</option>
                </select>
              </div>
            </div>
          )}

          <button type="submit" disabled={isSubmitting} className="btn-ink w-full mt-2">
            {isSubmitting ? (
              <div className="w-4 h-4 border-2 border-paper/30 border-t-paper rounded-full animate-spin" />
            ) : isLoginTab ? (
              'Đăng nhập'
            ) : (
              'Tạo tài khoản'
            )}
          </button>
        </form>

        <div className="mt-5 pt-4 border-t border-rule text-center">
          <button
            onClick={() => setShowAuthModal(false)}
            className="text-xs text-muted hover:text-vn-600 underline underline-offset-2 transition-colors"
          >
            Tiếp tục với chế độ Khách (chỉ xem)
          </button>
        </div>
      </div>
    </div>
  );
};