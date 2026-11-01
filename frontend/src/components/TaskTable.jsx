import React, { useState, useMemo } from 'react';
import { 
  Search, 
  Calendar, 
  AlertTriangle, 
  Edit3, 
  Trash2, 
  Eye, 
  Download, 
  Save, 
  Check, 
  User,
  Filter,
  CheckCircle2
} from 'lucide-react';
import { api } from '../services/api';

export default function TaskTable({ 
  meetingData, 
  onSaveMeeting, 
  onUpdateTask, 
  onDeleteTask, 
  onOpenDetails, 
  onOpenEdit, 
  showToast 
}) {
  const [searchTerm, setSearchTerm] = useState('');
  const [assigneeFilter, setAssigneeFilter] = useState('ALL');
  const [statusFilter, setStatusFilter] = useState('ALL');
  const [onlyReviewNeeded, setOnlyReviewNeeded] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  const tasks = meetingData.tasks || [];

  // Compute unique assignees
  const uniqueAssignees = useMemo(() => {
    const set = new Set();
    tasks.forEach((t) => {
      if (t.assignee) set.add(t.assignee);
    });
    return Array.from(set).sort();
  }, [tasks]);

  // Filter tasks
  const filteredTasks = useMemo(() => {
    return tasks.filter((task) => {
      const matchesSearch = 
        task.description.toLowerCase().includes(searchTerm.toLowerCase()) ||
        (task.assignee && task.assignee.toLowerCase().includes(searchTerm.toLowerCase())) ||
        (task.evidence && task.evidence.toLowerCase().includes(searchTerm.toLowerCase()));
      if (!matchesSearch) return false;

      if (assigneeFilter === 'UNASSIGNED' && task.assignee) return false;
      if (assigneeFilter !== 'ALL' && assigneeFilter !== 'UNASSIGNED' && task.assignee !== assigneeFilter) return false;

      if (statusFilter !== 'ALL' && task.status !== statusFilter) return false;

      if (onlyReviewNeeded && !task.needs_review) return false;

      return true;
    });
  }, [tasks, searchTerm, assigneeFilter, statusFilter, onlyReviewNeeded]);

  const handleSave = async () => {
    setIsSaving(true);
    try {
      await onSaveMeeting();
    } finally {
      setIsSaving(false);
    }
  };

  const cycleStatus = (task) => {
    const sequence = ['pending', 'confirmed', 'completed', 'cancelled'];
    const currentIndex = sequence.indexOf(task.status);
    const nextStatus = sequence[(currentIndex + 1) % sequence.length];
    onUpdateTask(task.id, { status: nextStatus });
    showToast(`Task status updated to ${nextStatus}`, 'info');
  };

  const handleQuickConfirm = (task) => {
    onUpdateTask(task.id, { status: 'confirmed', needs_review: false });
    showToast('Task confirmed', 'success');
  };

  const handleExport = async (format) => {
    try {
      await api.exportTasks({
        meetingId: meetingData.id,
        format,
        title: meetingData.title,
        tasks,
      });
      showToast(`Exported tasks as ${format.toUpperCase()}`, 'success');
    } catch (err) {
      showToast(`Export failed: ${err.message}`, 'error');
    }
  };

  return (
    <div className="card-panel animate-fade-in">
      {/* Meeting Header & Actions */}
      <div className="card-header">
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <h2 style={{ fontSize: '1.05rem', fontWeight: 600, color: 'var(--text-primary)' }}>
              {meetingData.title}
            </h2>
            <span className={`badge ${meetingData.extractionMode === 'v2_transformer' ? 'badge-info' : 'badge-mode'}`}>
              {meetingData.extractionMode === 'v2_transformer' ? 'V2 DistilBERT Transformer' : (meetingData.extractionMode || 'hybrid')}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.2rem', fontSize: '0.78rem', color: 'var(--text-muted)' }}>
            <span style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <Calendar size={13} />
              Meeting Date: <strong style={{ color: 'var(--text-secondary)' }}>{meetingData.meetingDate}</strong>
            </span>
            <span>•</span>
            <span><strong>{tasks.length}</strong> tasks extracted</span>
          </div>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
          <button className="btn-secondary" onClick={() => handleExport('csv')} title="Export CSV">
            <Download size={13} />
            Export CSV
          </button>
          <button className="btn-secondary" onClick={() => handleExport('json')} title="Export JSON">
            <Download size={13} />
            Export JSON
          </button>
          <button 
            className="btn-primary" 
            onClick={handleSave} 
            disabled={isSaving}
          >
            <Save size={14} />
            {isSaving ? 'Saving...' : meetingData.id ? 'Save Updates' : 'Save Meeting'}
          </button>
        </div>
      </div>

      <div className="card-body">
        {/* Filters Toolbar */}
        <div className="filter-toolbar">
          <div className="search-field">
            <Search size={14} className="search-icon" />
            <input
              type="text"
              className="form-control"
              style={{ fontSize: '0.82rem', padding: '0.4rem 0.65rem 0.4rem 1.9rem' }}
              placeholder="Search description, assignee, dialogue..."
              value={searchTerm}
              onChange={(e) => setSearchTerm(e.target.value)}
            />
          </div>

          <div style={{ minWidth: '150px' }}>
            <select
              className="form-control"
              style={{ fontSize: '0.82rem', padding: '0.4rem 0.65rem' }}
              value={assigneeFilter}
              onChange={(e) => setAssigneeFilter(e.target.value)}
            >
              <option value="ALL">All Assignees</option>
              <option value="UNASSIGNED">Unassigned Only</option>
              {uniqueAssignees.map((name) => (
                <option key={name} value={name}>{name}</option>
              ))}
            </select>
          </div>

          <div style={{ minWidth: '135px' }}>
            <select
              className="form-control"
              style={{ fontSize: '0.82rem', padding: '0.4rem 0.65rem' }}
              value={statusFilter}
              onChange={(e) => setStatusFilter(e.target.value)}
            >
              <option value="ALL">All Statuses</option>
              <option value="pending">Pending</option>
              <option value="confirmed">Confirmed</option>
              <option value="completed">Completed</option>
              <option value="cancelled">Cancelled</option>
            </select>
          </div>

          <button
            type="button"
            className={`btn-secondary ${onlyReviewNeeded ? 'active' : ''}`}
            style={{
              padding: '0.4rem 0.65rem',
              fontSize: '0.8rem',
              backgroundColor: onlyReviewNeeded ? 'var(--warning-bg)' : undefined,
              borderColor: onlyReviewNeeded ? 'var(--warning-border)' : undefined,
              color: onlyReviewNeeded ? 'var(--warning-text)' : undefined,
            }}
            onClick={() => setOnlyReviewNeeded(!onlyReviewNeeded)}
          >
            <AlertTriangle size={13} />
            Review Needed
          </button>
        </div>

        {/* Structured Table */}
        {filteredTasks.length === 0 ? (
          <div className="empty-state" style={{ padding: '2.5rem 1rem' }}>
            <p className="empty-desc">No tasks match the active filters or search term.</p>
          </div>
        ) : (
          <div className="table-wrapper" style={{ border: '1px solid var(--border-default)', borderRadius: 'var(--radius-md)' }}>
            <table className="clean-table">
              <thead>
                <tr>
                  <th style={{ width: '40%' }}>Task Description</th>
                  <th style={{ width: '16%' }}>Assignee</th>
                  <th style={{ width: '16%' }}>Deadline</th>
                  <th style={{ width: '11%' }}>Status</th>
                  <th style={{ width: '9%' }}>Confidence</th>
                  <th style={{ width: '8%', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {filteredTasks.map((task) => {
                  const confPercent = Math.round((task.confidence || 0.8) * 100);
                  const confBg = confPercent >= 85 ? 'var(--success-text)' : confPercent >= 70 ? 'var(--primary-500)' : 'var(--warning-text)';

                  return (
                    <tr key={task.id}>
                      {/* Description */}
                      <td>
                        <div style={{ display: 'flex', flexDirection: 'column' }}>
                          <span
                            style={{ fontWeight: 500, color: 'var(--text-primary)', cursor: 'pointer' }}
                            onClick={() => onOpenEdit(task)}
                            title="Click to edit task"
                          >
                            {task.description}
                          </span>
                          {task.needs_review && (
                            <div>
                              <span className="badge-review" title="Flagged for human confirmation">
                                <AlertTriangle size={10} />
                                Needs Review
                              </span>
                            </div>
                          )}
                        </div>
                      </td>

                      {/* Assignee */}
                      <td>
                        {task.assignee ? (
                          <div className="assignee-cell">
                            <span className="avatar-initial">
                              {task.assignee.charAt(0).toUpperCase()}
                            </span>
                            <span>{task.assignee}</span>
                          </div>
                        ) : (
                          <span className="unassigned-label">
                            Unassigned
                          </span>
                        )}
                      </td>

                      {/* Deadline */}
                      <td>
                        {task.deadline ? (
                          <div style={{ display: 'flex', flexDirection: 'column' }}>
                            <span style={{ fontSize: '0.84rem', fontWeight: 500, color: 'var(--text-primary)' }}>
                              {task.deadline}
                            </span>
                            {task.deadline_text && task.deadline_text !== task.deadline && (
                              <span style={{ fontSize: '0.72rem', color: 'var(--text-muted)' }}>
                                "{task.deadline_text}"
                              </span>
                            )}
                          </div>
                        ) : task.deadline_text ? (
                          <span style={{ fontSize: '0.8rem', color: 'var(--warning-text)' }}>
                            "{task.deadline_text}"
                          </span>
                        ) : (
                          <span style={{ fontSize: '0.78rem', color: 'var(--text-dim)' }}>
                            None
                          </span>
                        )}
                      </td>

                      {/* Status */}
                      <td>
                        <span
                          className={`badge badge-status ${task.status}`}
                          onClick={() => cycleStatus(task)}
                          title="Click to cycle status"
                        >
                          {task.status}
                        </span>
                      </td>

                      {/* Confidence */}
                      <td>
                        <div className="confidence-indicator" title={`Calibrated Confidence: ${confPercent}%`}>
                          <div className="confidence-track">
                            <div
                              className="confidence-fill"
                              style={{ width: `${confPercent}%`, backgroundColor: confBg }}
                            />
                          </div>
                          <span className="confidence-text">{confPercent}%</span>
                        </div>
                      </td>

                      {/* Actions */}
                      <td style={{ textAlign: 'right' }}>
                        <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.2rem' }}>
                          {task.status !== 'confirmed' && (
                            <button
                              className="btn-icon"
                              style={{ color: 'var(--success-text)' }}
                              title="Confirm Task"
                              onClick={() => handleQuickConfirm(task)}
                            >
                              <Check size={14} />
                            </button>
                          )}
                          <button
                            className="btn-icon"
                            title="View Linguistic Evidence"
                            onClick={(e) => {
                              e.stopPropagation();
                              onOpenDetails(task);
                            }}
                          >
                            <Eye size={14} />
                          </button>
                          <button
                            className="btn-icon"
                            title="Edit Task"
                            onClick={(e) => {
                              e.stopPropagation();
                              onOpenEdit(task);
                            }}
                          >
                            <Edit3 size={14} />
                          </button>
                          <button
                            className="btn-icon danger"
                            title="Delete Task"
                            onClick={(e) => {
                              e.stopPropagation();
                              onDeleteTask(task);
                            }}
                          >
                            <Trash2 size={14} />
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
