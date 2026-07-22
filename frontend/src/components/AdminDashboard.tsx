import React, { useEffect, useState } from 'react';
import { ApprovalRequest, AuditEntry } from '../types';
import { fetchApprovalsApi, decideApprovalApi, fetchAuditLogsApi, fetchStatsApi } from '../services/api';
import { Shield, CheckCircle, XCircle, Clock, FileText, RefreshCw, BarChart2 } from 'lucide-react';

export const AdminDashboard: React.FC = () => {
  const [activeTab, setActiveTab] = useState<'approvals' | 'audit' | 'stats'>('approvals');
  const [approvals, setApprovals] = useState<ApprovalRequest[]>([]);
  const [auditLogs, setAuditLogs] = useState<AuditEntry[]>([]);
  const [stats, setStats] = useState<any>(null);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const loadData = async () => {
    setIsLoading(true);
    try {
      if (activeTab === 'approvals') {
        const data = await fetchApprovalsApi();
        setApprovals(data);
      } else if (activeTab === 'audit') {
        const data = await fetchAuditLogsApi();
        setAuditLogs(data);
      } else if (activeTab === 'stats') {
        const data = await fetchStatsApi();
        setStats(data);
      }
    } catch (e) {
      console.error(e);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    loadData();
  }, [activeTab]);

  const handleDecide = async (id: string, decision: 'approved' | 'rejected') => {
    await decideApprovalApi(id, decision);
    loadData();
  };

  return (
    <div className="max-w-6xl mx-auto p-4 md:p-8 space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between border-b border-slate-800 pb-4">
        <div className="flex items-center gap-3">
          <div className="w-12 h-12 rounded-2xl bg-legal-600/20 border border-legal-500/30 flex items-center justify-center text-legal-400">
            <Shield className="w-6 h-6" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-white tracking-tight">Trung Tâm Quản Trị Admin & Security</h2>
            <p className="text-xs text-slate-400">Phê duyệt Tool Approvals, Kiểm tra Audit Logs & Thống kê hệ thống</p>
          </div>
        </div>

        <button
          onClick={loadData}
          disabled={isLoading}
          className="p-2.5 rounded-xl glass-panel hover:bg-slate-800 text-slate-300 transition flex items-center gap-2 text-xs font-medium"
        >
          <RefreshCw className={`w-4 h-4 ${isLoading ? 'animate-spin' : ''}`} />
          Làm mới
        </button>
      </div>

      {/* Tabs */}
      <div className="flex gap-2 border-b border-slate-800">
        <button
          onClick={() => setActiveTab('approvals')}
          className={`pb-3 px-4 font-semibold text-sm transition border-b-2 flex items-center gap-2 ${
            activeTab === 'approvals'
              ? 'border-legal-500 text-legal-400'
              : 'border-transparent text-slate-400 hover:text-white'
          }`}
        >
          <Clock className="w-4 h-4" />
          Tool Approvals Phê Duyệt ({approvals.length})
        </button>
        <button
          onClick={() => setActiveTab('audit')}
          className={`pb-3 px-4 font-semibold text-sm transition border-b-2 flex items-center gap-2 ${
            activeTab === 'audit'
              ? 'border-legal-500 text-legal-400'
              : 'border-transparent text-slate-400 hover:text-white'
          }`}
        >
          <FileText className="w-4 h-4" />
          Nhật Ký Hệ Thống (Audit Trail)
        </button>
        <button
          onClick={() => setActiveTab('stats')}
          className={`pb-3 px-4 font-semibold text-sm transition border-b-2 flex items-center gap-2 ${
            activeTab === 'stats'
              ? 'border-legal-500 text-legal-400'
              : 'border-transparent text-slate-400 hover:text-white'
          }`}
        >
          <BarChart2 className="w-4 h-4" />
          Thống Kê Cache & Router
        </button>
      </div>

      {/* Content */}
      {activeTab === 'approvals' && (
        <div className="space-y-4">
          {approvals.length === 0 ? (
            <div className="glass-panel p-8 rounded-2xl text-center text-slate-400 text-sm">
              Không có yêu cầu phê duyệt công cụ nào đang chờ.
            </div>
          ) : (
            approvals.map((app) => (
              <div key={app.id} className="glass-panel p-5 rounded-2xl border border-slate-800 flex items-center justify-between">
                <div>
                  <div className="flex items-center gap-2">
                    <span className="font-bold text-white text-base">{app.tool_name}</span>
                    <span className="px-2 py-0.5 rounded text-[10px] font-mono uppercase bg-amber-500/20 text-amber-300 border border-amber-500/30">
                      {app.status}
                    </span>
                  </div>
                  <pre className="mt-2 p-3 rounded-lg bg-slate-900 font-mono text-xs text-slate-300 overflow-x-auto max-w-xl">
                    {JSON.stringify(app.arguments, null, 2)}
                  </pre>
                </div>
                <div className="flex items-center gap-2">
                  <button
                    onClick={() => handleDecide(app.id, 'approved')}
                    className="px-4 py-2 rounded-xl bg-emerald-600 hover:bg-emerald-500 text-white font-semibold text-xs flex items-center gap-1.5 shadow-md shadow-emerald-600/20 transition"
                  >
                    <CheckCircle className="w-4 h-4" />
                    Chấp thuận
                  </button>
                  <button
                    onClick={() => handleDecide(app.id, 'rejected')}
                    className="px-4 py-2 rounded-xl bg-rose-600 hover:bg-rose-500 text-white font-semibold text-xs flex items-center gap-1.5 shadow-md shadow-rose-600/20 transition"
                  >
                    <XCircle className="w-4 h-4" />
                    Từ chối
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {activeTab === 'audit' && (
        <div className="glass-panel rounded-2xl overflow-hidden border border-slate-800">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs text-slate-300">
              <thead className="bg-slate-900/80 text-slate-400 uppercase font-semibold text-[10px]">
                <tr>
                  <th className="p-3">Thời gian</th>
                  <th className="p-3">User ID</th>
                  <th className="p-3">Hành động</th>
                  <th className="p-3">Tài nguyên</th>
                  <th className="p-3">IP Address</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-slate-800">
                {auditLogs.map((log) => (
                  <tr key={log.id} className="hover:bg-slate-900/40">
                    <td className="p-3 font-mono text-[11px] text-slate-400">{log.timestamp}</td>
                    <td className="p-3 font-medium text-slate-200">{log.user_id || 'anonymous'}</td>
                    <td className="p-3 font-semibold text-legal-400">{log.action}</td>
                    <td className="p-3 text-slate-400">{log.resource}</td>
                    <td className="p-3 font-mono text-slate-500">{log.ip || '-'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}

      {activeTab === 'stats' && (
        <div className="glass-panel p-6 rounded-2xl border border-slate-800">
          <pre className="p-4 rounded-xl bg-slate-900 font-mono text-xs text-slate-300 overflow-x-auto">
            {JSON.stringify(stats, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
};
