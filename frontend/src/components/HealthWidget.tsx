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
      <div className="p-3 glass-panel rounded-xl text-xs text-slate-400 flex items-center justify-between">
        <div className="flex items-center gap-2">
          <Activity className="w-4 h-4 text-legal-400 animate-pulse" />
          <span>Kiểm tra hệ thống...</span>
        </div>
      </div>
    );
  }

  const getIcon = (status: string) => {
    if (status === 'healthy') return <CheckCircle2 className="w-3.5 h-3.5 text-emerald-400 shrink-0" />;
    if (status === 'not_configured') return <AlertTriangle className="w-3.5 h-3.5 text-amber-400 shrink-0" />;
    return <XCircle className="w-3.5 h-3.5 text-rose-400 shrink-0" />;
  };

  return (
    <div className="glass-panel rounded-xl overflow-hidden text-xs transition-all border border-slate-800">
      <button
        onClick={() => setIsExpanded(!isExpanded)}
        className="w-full p-3 flex items-center justify-between hover:bg-slate-800/40 transition text-left"
      >
        <div className="flex items-center gap-2">
          <div className={`w-2 h-2 rounded-full ${health.status === 'healthy' ? 'bg-emerald-400 shadow-sm shadow-emerald-400' : 'bg-rose-400 shadow-sm shadow-rose-400'}`} />
          <span className="font-semibold text-slate-200">Trạng Thái Hệ Thống</span>
        </div>
        <span className="text-[10px] uppercase font-mono px-2 py-0.5 rounded-full bg-slate-900 text-slate-400 border border-slate-800">
          {health.status === 'healthy' ? 'Sẵn sàng' : 'Cảnh báo'}
        </span>
      </button>

      {isExpanded && (
        <div className="px-3 pb-3 pt-1 space-y-2 border-t border-slate-800/60 bg-slate-950/40">
          <div className="flex items-center justify-between text-slate-300">
            <div className="flex items-center gap-1.5">
              <Server className="w-3.5 h-3.5 text-slate-400" />
              <span>FastAPI Backend</span>
            </div>
            {getIcon('healthy')}
          </div>

          <div className="flex items-center justify-between text-slate-300">
            <div className="flex items-center gap-1.5">
              <Database className="w-3.5 h-3.5 text-slate-400" />
              <span>SQL (MariaDB)</span>
            </div>
            {getIcon(health.database.status)}
          </div>

          <div className="flex items-center justify-between text-slate-300">
            <div className="flex items-center gap-1.5">
              <Layers className="w-3.5 h-3.5 text-slate-400" />
              <span>Qdrant Vector DB</span>
            </div>
            {getIcon(health.qdrant.status)}
          </div>

          <div className="flex items-center justify-between text-slate-300">
            <div className="flex items-center gap-1.5">
              <Cpu className="w-3.5 h-3.5 text-slate-400" />
              <span>Celery Worker</span>
            </div>
            {getIcon(health.celery.status)}
          </div>
        </div>
      )}
    </div>
  );
};
