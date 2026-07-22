import React, { useEffect, useState } from 'react';
import { ApprovalRequest, AuditEntry } from '../types';
import { fetchApprovalsApi, decideApprovalApi, fetchAuditLogsApi, fetchStatsApi } from '../services/api';
import { Shield, Check, X, Clock, FileText, RefreshCw, BarChart2 } from 'lucide-react';

const pad2 = (n: number) => String(n).padStart(2, '0');

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

  const tabs: { key: typeof activeTab; label: string; icon: React.ReactNode; count?: number }[] = [
    { key: 'approvals', label: 'Phê duyệt', icon: <Clock className="w-4 h-4" />, count: approvals.length },
    { key: 'audit', label: 'Audit Trail', icon: <FileText className="w-4 h-4" /> },
    { key: 'stats', label: 'Cache & Router', icon: <BarChart2 className="w-4 h-4" /> },
  ];

  return (
    <div className="max-w-6xl mx-auto p-5 md:p-8">
      {/* Header */}
      <div className="flex items-end justify-between border-b border-ink pb-4">
        <div>
          <div className="flex items-center gap-2 mb-2">
            <Shield className="w-3.5 h-3.5 text-vn-500" />
            <span className="label-mono text-vn-600">Admin · Security</span>
          </div>
          <h2 className="text-2xl font-semibold tracking-display text-ink">Trung tâm Quản trị</h2>
          <p className="text-xs text-muted mt-1">
            Phê duyệt tool · Audit logs · Thống kê cache & router
          </p>
        </div>

        <button onClick={loadData} disabled={isLoading} className="btn-outline !py-2 !px-3 !text-xs">
          <RefreshCw className={`w-3.5 h-3.5 ${isLoading ? 'animate-spin' : ''}`} />
          Làm mới
        </button>
      </div>

      {/* Tabs */}
      <nav className="flex gap-0 border-b border-rule mt-6">
        {tabs.map((t, i) => (
          <button
            key={t.key}
            onClick={() => setActiveTab(t.key)}
            className={`inline-flex items-center gap-2 py-3 px-4 text-sm font-medium transition-colors border-b-2 -mb-px ${
              activeTab === t.key
                ? 'border-vn-500 text-ink'
                : 'border-transparent text-faint hover:text-ink'
            }`}
          >
            <span className="font-mono text-[10px] text-faint tabular-nums">{pad2(i + 1)}</span>
            {t.icon}
            {t.label}
            {t.count !== undefined && (
              <span className="font-mono text-[10px] text-muted">({t.count})</span>
            )}
          </button>
        ))}
      </nav>

      {/* Approvals */}
      {activeTab === 'approvals' && (
        <div className="mt-6 space-y-3">
          {approvals.length === 0 ? (
            <div className="border border-rule p-10 text-center text-muted text-sm">
              Không có yêu cầu phê duyệt nào đang chờ.
            </div>
          ) : (
            approvals.map((app, idx) => (
              <div
                key={app.id}
                className="border border-rule rounded-swiss p-5 flex items-start justify-between gap-4"
              >
                <div className="min-w-0 flex-1">
                  <div className="flex items-center gap-3 mb-2">
                    <span className="font-mono text-[10px] text-faint tabular-nums">{pad2(idx + 1)}</span>
                    <span className="font-semibold text-ink text-base">{app.tool_name}</span>
                    <span className="font-mono text-[10px] uppercase tracking-label px-2 py-0.5 bg-vn-50 text-vn-700 border border-vn-100 rounded-swiss">
                      {app.status}
                    </span>
                  </div>
                  <pre className="p-3 bg-ink text-paper font-mono text-xs overflow-x-auto rounded-swiss">
                    {JSON.stringify(app.arguments, null, 2)}
                  </pre>
                </div>
                <div className="flex flex-col gap-2 shrink-0">
                  <button
                    onClick={() => handleDecide(app.id, 'approved')}
                    className="btn-ink !py-2 !px-3 !text-xs"
                  >
                    <Check className="w-3.5 h-3.5" />
                    Chấp thuận
                  </button>
                  <button
                    onClick={() => handleDecide(app.id, 'rejected')}
                    className="btn-outline !py-2 !px-3 !text-xs hover:!border-vn-600 hover:!text-vn-600"
                  >
                    <X className="w-3.5 h-3.5" />
                    Từ chối
                  </button>
                </div>
              </div>
            ))
          )}
        </div>
      )}

      {/* Audit */}
      {activeTab === 'audit' && (
        <div className="mt-6 border border-rule rounded-swiss overflow-hidden overflow-x-auto">
          <table className="w-full text-left text-xs">
            <thead className="bg-paper-dim text-muted uppercase font-mono text-[10px] tracking-label">
              <tr>
                <th className="p-3 border-b border-rule">Thời gian</th>
                <th className="p-3 border-b border-rule">User ID</th>
                <th className="p-3 border-b border-rule">Hành động</th>
                <th className="p-3 border-b border-rule">Tài nguyên</th>
                <th className="p-3 border-b border-rule">IP</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-rule">
              {auditLogs.map((log) => (
                <tr key={log.id} className="hover:bg-paper-dim transition-colors">
                  <td className="p-3 font-mono text-[11px] text-muted">{log.timestamp}</td>
                  <td className="p-3 font-medium text-ink">{log.user_id || 'anonymous'}</td>
                  <td className="p-3 font-mono text-vn-600">{log.action}</td>
                  <td className="p-3 text-muted">{log.resource}</td>
                  <td className="p-3 font-mono text-faint">{log.ip || '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Stats */}
      {activeTab === 'stats' && (
        <div className="mt-6 border border-rule rounded-swiss p-5">
          <pre className="p-4 bg-ink text-paper font-mono text-xs overflow-x-auto rounded-swiss">
            {JSON.stringify(stats, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
};