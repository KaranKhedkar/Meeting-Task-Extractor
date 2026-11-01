import React from 'react';
import { CheckCircle2, AlertCircle, Info, X } from 'lucide-react';

export default function Toast({ toasts, onDismiss }) {
  if (!toasts || toasts.length === 0) return null;

  return (
    <div className="toast-container">
      {toasts.map((t) => (
        <div key={t.id} className={`toast ${t.type || 'info'}`}>
          {t.type === 'success' && <CheckCircle2 size={16} color="var(--success-text)" />}
          {t.type === 'error' && <AlertCircle size={16} color="var(--danger-text)" />}
          {t.type === 'info' && <Info size={16} color="var(--primary-500)" />}
          <span style={{ flex: 1, color: 'var(--text-primary)' }}>{t.message}</span>
          <button
            type="button"
            className="btn-icon"
            style={{ padding: '0.15rem' }}
            onClick={() => onDismiss(t.id)}
          >
            <X size={13} />
          </button>
        </div>
      ))}
    </div>
  );
}
