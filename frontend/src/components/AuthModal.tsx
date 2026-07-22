import React, { useState } from 'react';
import { useAuth } from '../context/AuthContext';
import { Scale, Lock, User as UserIcon, X, AlertCircle, LogIn, UserPlus, ShieldCheck } from 'lucide-react';

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
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-slate-950/80 backdrop-blur-md animate-fade-in">
      <div className="relative w-full max-w-md glass-panel rounded-2xl p-6 md:p-8 shadow-2xl border border-slate-700/50">
        {/* Close Button */}
        <button
          onClick={() => setShowAuthModal(false)}
          className="absolute top-4 right-4 p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/60 transition"
        >
          <X className="w-5 h-5" />
        </button>

        {/* Header Logo */}
        <div className="flex flex-col items-center mb-6">
          <div className="w-14 h-14 rounded-2xl bg-gradient-to-tr from-legal-600 to-legal-400 flex items-center justify-center shadow-lg shadow-legal-500/25 mb-3">
            <Scale className="w-8 h-8 text-white" />
          </div>
          <h2 className="text-2xl font-bold text-white tracking-tight">Trợ Lý Pháp Luật AI</h2>
          <p className="text-sm text-slate-400 mt-1">Đăng nhập để lưu lịch sử & trí nhớ dài hạn</p>
        </div>

        {/* Tabs */}
        <div className="grid grid-cols-2 gap-1 p-1 bg-slate-900/80 rounded-xl mb-6 border border-slate-800">
          <button
            type="button"
            onClick={() => { setIsLoginTab(true); setError(null); }}
            className={`flex items-center justify-center gap-2 py-2.5 rounded-lg font-medium text-sm transition ${
              isLoginTab
                ? 'bg-legal-600 text-white shadow-md'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <LogIn className="w-4 h-4" />
            Đăng nhập
          </button>
          <button
            type="button"
            onClick={() => { setIsLoginTab(false); setError(null); }}
            className={`flex items-center justify-center gap-2 py-2.5 rounded-lg font-medium text-sm transition ${
              !isLoginTab
                ? 'bg-legal-600 text-white shadow-md'
                : 'text-slate-400 hover:text-white'
            }`}
          >
            <UserPlus className="w-4 h-4" />
            Đăng ký
          </button>
        </div>

        {/* Error Alert */}
        {error && (
          <div className="flex items-start gap-3 p-3.5 mb-5 bg-rose-500/10 border border-rose-500/30 rounded-xl text-rose-400 text-sm">
            <AlertCircle className="w-5 h-5 shrink-0 mt-0.5" />
            <span>{error}</span>
          </div>
        )}

        {/* Form */}
        <form onSubmit={handleSubmit} className="space-y-4">
          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
              Tên đăng nhập
            </label>
            <div className="relative">
              <UserIcon className="absolute left-3.5 top-3 w-4 h-4 text-slate-400" />
              <input
                type="text"
                value={username}
                onChange={(e) => setUsername(e.target.value)}
                placeholder="Ví dụ: lawyer_user, user123..."
                minLength={3}
                maxLength={64}
                pattern="^[A-Za-z0-9_.-]+$"
                title="Chỉ gồm chữ cái không dấu, số, dấu gạch dưới hoặc gạch ngang (tối thiểu 3 ký tự)."
                className="w-full pl-10 pr-4 py-2.5 rounded-xl glass-input text-white text-sm placeholder-slate-500"
                required
              />
            </div>
            <p className="text-[10px] text-slate-500 mt-1">Viết liền không dấu, từ 3-64 ký tự (chỉ gồm a-z, 0-9, _, -)</p>
          </div>

          <div>
            <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
              Mật khẩu
            </label>
            <div className="relative">
              <Lock className="absolute left-3.5 top-3 w-4 h-4 text-slate-400" />
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                placeholder="Tối thiểu 6 ký tự..."
                minLength={6}
                maxLength={128}
                className="w-full pl-10 pr-4 py-2.5 rounded-xl glass-input text-white text-sm placeholder-slate-500"
                required
              />
            </div>
            <p className="text-[10px] text-slate-500 mt-1">Mật khẩu tối thiểu 6 ký tự</p>
          </div>

          {!isLoginTab && (
            <div>
              <label className="block text-xs font-semibold text-slate-300 uppercase tracking-wider mb-1.5">
                Vai trò (Role)
              </label>
              <div className="relative">
                <ShieldCheck className="absolute left-3.5 top-3 w-4 h-4 text-slate-400" />
                <select
                  value={role}
                  onChange={(e) => setRole(e.target.value)}
                  className="w-full pl-10 pr-4 py-2.5 rounded-xl glass-input text-white text-sm bg-slate-900 cursor-pointer"
                >
                  <option value="user">Người dùng (User)</option>
                  <option value="lawyer">Luật sư / Thư ký pháp lý (Lawyer)</option>
                </select>
              </div>
            </div>
          )}

          <button
            type="submit"
            disabled={isSubmitting}
            className="w-full py-3 rounded-xl bg-gradient-to-r from-legal-600 to-legal-500 hover:from-legal-500 hover:to-legal-400 text-white font-semibold text-sm shadow-lg shadow-legal-600/30 transition duration-200 disabled:opacity-50 flex items-center justify-center gap-2 mt-6"
          >
            {isSubmitting ? (
              <div className="w-5 h-5 border-2 border-white/30 border-t-white rounded-full animate-spin" />
            ) : isLoginTab ? (
              'Đăng Nhập Ngay'
            ) : (
              'Tạo Tài Khoản Mới'
            )}
          </button>
        </form>

        {/* Guest Mode footer */}
        <div className="mt-6 text-center">
          <button
            onClick={() => setShowAuthModal(false)}
            className="text-xs text-slate-400 hover:text-legal-400 underline transition"
          >
            Tiếp tục với chế độ Khách (Guest Mode)
          </button>
        </div>
      </div>
    </div>
  );
};
