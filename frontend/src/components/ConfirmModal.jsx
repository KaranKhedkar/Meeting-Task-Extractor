import React, { useEffect } from 'react';
import { AlertTriangle, Trash2, X } from 'lucide-react';

export default function ConfirmModal({
  isOpen,
  title = 'Confirm Deletion',
  message = 'Are you sure you want to delete this item? This action cannot be undone.',
  confirmLabel = 'Delete',
  cancelLabel = 'Cancel',
  isDanger = true,
  onConfirm,
  onCancel,
}) {
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (!isOpen) return;
      if (e.key === 'Escape') onCancel();
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isOpen, onCancel]);

  if (!isOpen) return null;

  return (
    <div className="modal-backdrop" onClick={onCancel} style={{ zIndex: 1100 }}>
      <div
        className="modal-dialog animate-fade-in"
        onClick={(e) => e.stopPropagation()}
        style={{ maxWidth: '420px', padding: 0, overflow: 'hidden' }}
      >
        <div style={{ padding: '1.35rem 1.35rem 1.15rem 1.35rem', display: 'flex', gap: '0.85rem' }}>
          <div
            style={{
              width: '38px',
              height: '38px',
              borderRadius: 'var(--radius-md)',
              backgroundColor: isDanger ? 'var(--danger-bg)' : 'var(--warning-bg)',
              color: isDanger ? 'var(--danger-text)' : 'var(--warning-text)',
              display: 'flex',
              alignItems: 'center',
              justifyContent: 'center',
              flexShrink: 0,
            }}
          >
            {isDanger ? <Trash2 size={18} /> : <AlertTriangle size={18} />}
          </div>
          <div style={{ flex: 1 }}>
            <h3
              style={{
                fontSize: '0.96rem',
                fontWeight: 600,
                color: 'var(--text-primary)',
                marginBottom: '0.35rem',
              }}
            >
              {title}
            </h3>
            <p
              style={{
                fontSize: '0.82rem',
                color: 'var(--text-secondary)',
                lineHeight: 1.45,
                margin: 0,
              }}
            >
              {message}
            </p>
          </div>
          <button
            type="button"
            className="btn-icon"
            onClick={onCancel}
            title="Close"
            style={{ alignSelf: 'flex-start', color: 'var(--text-muted)' }}
          >
            <X size={15} />
          </button>
        </div>

        <div
          style={{
            padding: '0.85rem 1.35rem',
            backgroundColor: 'var(--bg-subtle)',
            borderTop: '1px solid var(--border-default)',
            display: 'flex',
            justifyContent: 'flex-end',
            gap: '0.5rem',
          }}
        >
          <button
            type="button"
            className="btn-secondary"
            style={{ fontSize: '0.82rem', padding: '0.4rem 0.85rem' }}
            onClick={onCancel}
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            style={{
              fontSize: '0.82rem',
              padding: '0.4rem 0.95rem',
              backgroundColor: isDanger ? 'var(--danger-text)' : 'var(--primary-600)',
              color: '#ffffff',
              border: 'none',
              borderRadius: 'var(--radius-md)',
              fontWeight: 500,
              cursor: 'pointer',
              display: 'inline-flex',
              alignItems: 'center',
              gap: '0.35rem',
              transition: 'background-color 0.15s ease',
            }}
            onClick={onConfirm}
          >
            {isDanger && <Trash2 size={13} />}
            {confirmLabel}
          </button>
        </div>
      </div>
    </div>
  );
}
