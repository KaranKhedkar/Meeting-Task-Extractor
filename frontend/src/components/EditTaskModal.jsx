import React, { useState } from 'react';
import { X, Check, Trash2 } from 'lucide-react';

export default function EditTaskModal({ task, onSave, onDelete, onClose }) {
  if (!task) return null;

  const [description, setDescription] = useState(task.description || '');
  const [assignee, setAssignee] = useState(task.assignee || '');
  const [deadline, setDeadline] = useState(task.deadline || '');
  const [deadlineText, setDeadlineText] = useState(task.deadline_text || '');
  const [status, setStatus] = useState(task.status || 'pending');
  const [needsReview, setNeedsReview] = useState(task.needs_review || false);

  const handleSubmit = (e) => {
    e.preventDefault();
    if (!description.trim()) return;

    onSave(task.id, {
      description: description.trim(),
      assignee: assignee.trim() ? assignee.trim() : null,
      deadline: deadline || null,
      deadline_text: deadlineText.trim() ? deadlineText.trim() : null,
      status,
      needs_review: needsReview,
    });
    onClose();
  };

  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-dialog" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3 className="modal-title">Edit Task Details</h3>
          <button type="button" className="btn-icon" onClick={onClose}>
            <X size={16} />
          </button>
        </div>

        <form onSubmit={handleSubmit}>
          <div className="modal-body">
            <div className="form-group">
              <label className="form-label">Action Description</label>
              <textarea
                className="form-control"
                style={{ height: '75px', resize: 'vertical' }}
                value={description}
                onChange={(e) => setDescription(e.target.value)}
                required
              />
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.85rem' }}>
              <div className="form-group">
                <label className="form-label">Assignee</label>
                <input
                  type="text"
                  className="form-control"
                  placeholder="e.g. Rahul (or blank)"
                  value={assignee}
                  onChange={(e) => setAssignee(e.target.value)}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Status</label>
                <select
                  className="form-control"
                  value={status}
                  onChange={(e) => setStatus(e.target.value)}
                >
                  <option value="pending">Pending</option>
                  <option value="confirmed">Confirmed</option>
                  <option value="completed">Completed</option>
                  <option value="cancelled">Cancelled</option>
                </select>
              </div>
            </div>

            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '0.85rem' }}>
              <div className="form-group">
                <label className="form-label">Target Date (ISO)</label>
                <input
                  type="date"
                  className="form-control"
                  value={deadline}
                  onChange={(e) => setDeadline(e.target.value)}
                />
              </div>

              <div className="form-group">
                <label className="form-label">Raw Deadline Phrase</label>
                <input
                  type="text"
                  className="form-control"
                  placeholder="e.g. by Friday"
                  value={deadlineText}
                  onChange={(e) => setDeadlineText(e.target.value)}
                />
              </div>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', paddingTop: '0.25rem' }}>
              <input
                type="checkbox"
                id="reviewCheckbox"
                checked={needsReview}
                onChange={(e) => setNeedsReview(e.target.checked)}
                style={{ width: '15px', height: '15px', accentColor: 'var(--primary-500)', cursor: 'pointer' }}
              />
              <label htmlFor="reviewCheckbox" style={{ fontSize: '0.84rem', color: 'var(--text-secondary)', cursor: 'pointer', margin: 0 }}>
                Flag this task as needing human review
              </label>
            </div>
          </div>

          <div className="modal-footer" style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            {onDelete ? (
              <button
                type="button"
                className="btn-danger-outline"
                onClick={() => {
                  onClose();
                  onDelete(task);
                }}
              >
                <Trash2 size={13} />
                Delete Task
              </button>
            ) : <div />}
            <div style={{ display: 'flex', gap: '0.5rem' }}>
              <button type="button" className="btn-secondary" onClick={onClose}>
                Cancel
              </button>
              <button type="submit" className="btn-primary">
                <Check size={14} />
                Save Changes
              </button>
            </div>
          </div>
        </form>
      </div>
    </div>
  );
}
