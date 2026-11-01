import React from 'react';
import { 
  X, 
  MessageSquare, 
  Calendar, 
  User, 
  ShieldCheck, 
  AlertTriangle 
} from 'lucide-react';

export default function TaskDetailModal({ task, onClose }) {
  if (!task) return null;

  const confPercent = Math.round((task.confidence || 0.8) * 100);

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <div>
            <h3 className="modal-title">Linguistic Evidence & Extraction Details</h3>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
              Deterministic NLP parsing attribution and confidence breakdown.
            </p>
          </div>
          <button className="btn-icon" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        <div className="modal-body">
          {/* Source Evidence Box */}
          <div>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '0.35rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}>
              <MessageSquare size={13} color="var(--primary-500)" />
              Original Transcript Utterance
            </div>
            <div className="evidence-quote-box">
              "{task.evidence}"
            </div>
          </div>

          {/* Extracted Core Action */}
          <div>
            <div style={{ fontSize: '0.75rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', letterSpacing: '0.04em', marginBottom: '0.25rem' }}>
              Extracted Action Predicate
            </div>
            <div style={{ fontSize: '0.92rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              {task.description}
            </div>
          </div>

          {/* Assignee & Deadline Resolution Grid */}
          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.85rem' }}>
            <div style={{ padding: '0.75rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-default)', backgroundColor: '#fafbfc' }}>
              <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                <User size={12} />
                Assignee Resolution
              </div>
              <div style={{ marginTop: '0.35rem', fontSize: '0.88rem', fontWeight: 500, color: task.assignee ? 'var(--text-primary)' : 'var(--warning-text)' }}>
                {task.assignee || 'Unassigned (No definitive owner found)'}
              </div>
            </div>

            <div style={{ padding: '0.75rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-default)', backgroundColor: '#fafbfc' }}>
              <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                <Calendar size={12} />
                Deadline Resolution
              </div>
              <div style={{ marginTop: '0.35rem', fontSize: '0.88rem', fontWeight: 500, color: task.deadline ? 'var(--text-primary)' : 'var(--text-dim)' }}>
                {task.deadline || 'No target date'}
                {task.deadline_text && (
                  <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', fontWeight: 400 }}>
                    Extracted phrase: "{task.deadline_text}"
                  </div>
                )}
              </div>
            </div>
          </div>

          {/* Confidence and Review Alert */}
          <div style={{ padding: '0.75rem', borderRadius: 'var(--radius-md)', border: '1px solid var(--border-default)', backgroundColor: '#fafbfc' }}>
            <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.25rem' }}>
              <div style={{ fontSize: '0.72rem', fontWeight: 600, color: 'var(--text-muted)', textTransform: 'uppercase', display: 'flex', alignItems: 'center', gap: '0.3rem' }}>
                <ShieldCheck size={12} />
                Calibrated Confidence
              </div>
              <span style={{ fontSize: '0.84rem', fontWeight: 600, color: confPercent >= 80 ? 'var(--success-text)' : 'var(--warning-text)' }}>
                {confPercent}%
              </span>
            </div>

            <p style={{ fontSize: '0.78rem', color: 'var(--text-secondary)' }}>
              Calculated from classifier prediction probability minus penalties for indefinite pronouns, relative ambiguity, or fallback predicate segmentation.
            </p>

            {task.needs_review && (
              <div style={{ 
                marginTop: '0.65rem', 
                backgroundColor: 'var(--warning-bg)', 
                border: '1px solid var(--warning-border)',
                padding: '0.55rem 0.75rem',
                borderRadius: 'var(--radius-sm)',
                display: 'flex',
                alignItems: 'flex-start',
                gap: '0.45rem',
                fontSize: '0.78rem',
                color: 'var(--warning-text)'
              }}>
                <AlertTriangle size={14} style={{ flexShrink: 0, marginTop: '0.1rem' }} />
                <div>
                  <strong>Human Review Flagged:</strong> Either the assignee was not definitively identified, the deadline expression lacks an exact target calendar date, or confidence score is below 70%.
                </div>
              </div>
            )}
          </div>
        </div>

        <div className="modal-footer">
          <button type="button" className="btn-secondary" onClick={onClose}>
            Close
          </button>
        </div>
      </div>
    </div>
  );
}
