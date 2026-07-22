import React, { useEffect, useState } from 'react';
import { DetailedHealth } from '../types';
import { fetchDetailedHealthApi } from '../services/api';
import { Activity, Server, Database, Layers, Cpu, CheckCircle2, AlertTriangle, XCircle } from 'lucide-react';

export const HealthWidget: React.FC = () => {
  const [health, setHealth] = useState<DetailedHealth | null>(null);
  const [isExpanded, setIsExpanded] = useState<boolean>(false);

  useEffect(() => {
    let isMounted = true;
    async function loadHealth() {
      const data = await fetchDetailedHealthApi();
      if (isMounted) setHealth(data);
    }
    loadHealth();
    const interval = setInterval(loadHealth, 15000);
    return () => {
      isMounted = false;
      clearInterval(interval);
    };
  }, []);

  if (!health) {
    return (
      <div className="flex items-center gap-2 text-xs text-muted py-1">
        <Activity className="w-3.5 h-3.5 text-vn-500 animate-pulse" />
        <span>Kiểm tra hệ thống…</span>
      </div>
    );
  }

  const StatusIcon = ({ status }: { status: string }) => {
    if (status === 'healthy') return <CheckCircle2 className="w-3.5 h-3.5 text-ink shrink-0" />;
    if (status === 'not_configured') return <AlertTriangle className="w-3.5 h-3.5 text-vn-500 shrink-0" />;
    return <XCircle className="w-3.5 h-3.5 text-vn-600 shrink-0" />;
  };

  const rows: { icon: React.ReactNode; label: string; status: string }[] = [
    { icon: <Server className="w-3.5 h-3.5 text-muted" />, label: 'FastAPI Backend', status: 'healthy' },
    { icon: <Database className="w-3.5 h-3.5 text-muted" />, label: 'SQL · MariaDB', status: health.database.status },
    { icon: <Layers className="w-3.5 h-3.5 text-muted" />, label: 'Qdrant Vector', status: health.qdrant.status },
    { icon: <Cpu className="w-3.5 h-3.5 text-muted" />, label: 'Celery Worker', status: health.celery.status },
  ];

  return (
    <div className="border border-rule rounded-swiss overflow-hidden">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full px-3 py-2 flex items-center justify-between hover:bg-paper-tint transition-colors text-left"
      >
        <div className="flex items-center gap-2">
          <span className={`w-2 h-2 rounded-full ${health.status === 'healthy' ? 'bg-ink' : 'bg-vn-500'}`} />
          <span className="label-mono">Hệ thống</span>
        </div>
        <span className="font-mono text-[10px] uppercase tracking-label text-muted">
          {health.status === 'healthy' ? 'OK' : 'WARN'}
        </span>
      </button>

      {isExpanded && (
        <div className="px-3 pb-3 pt-2 border-t border-rule space-y-2">
          {rows.map((r, i) => (
            <div key={i} className="flex items-center justify-between">
              <div className="flex items-center gap-1.5 text-xs text-ink">
                {r.icon}
                <span>{r.label}</span>
              </div>
              <StatusIcon status={r.status} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
};