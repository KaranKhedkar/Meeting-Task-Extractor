import React from 'react';
import { 
  FileText, 
  Clock, 
  CheckCircle2, 
  AlertTriangle, 
  Plus, 
  Download, 
  Trash2, 
  ExternalLink,
  FolderOpen
} from 'lucide-react';
import { api } from '../services/api';

export default function Dashboard({ 
  meetings, 
  onOpenMeeting, 
  onNewMeeting, 
  onDeleteMeeting, 
  showToast 
}) {
  const totalMeetings = meetings.length;
  const totalTasks = meetings.reduce((acc, m) => acc + (m.task_count || 0), 0);
  const totalPending = meetings.reduce((acc, m) => acc + (m.pending_count || 0), 0);
  const totalReview = meetings.reduce((acc, m) => acc + (m.review_needed_count || 0), 0);
  const totalConfirmed = Math.max(0, totalTasks - totalPending);

  const handleExport = async (meeting, format) => {
    try {
      await api.exportTasks({ 
        meetingId: meeting.id, 
        format, 
        title: meeting.title 
      });
      showToast(`Exported "${meeting.title}" as ${format.toUpperCase()}`, 'success');
    } catch (err) {
      showToast(`Export failed: ${err.message}`, 'error');
    }
  };

  return (
    <div className="animate-fade-in">
      {/* Page Header */}
      <div className="page-header">
        <div>
          <h1 className="page-title">Workspace Overview</h1>
          <p className="page-subtitle">
            Meeting extraction statistics and recorded task commitments.
          </p>
        </div>

        <button className="btn-primary" onClick={onNewMeeting}>
          <Plus size={15} />
          New Meeting
        </button>
      </div>

      {/* Metrics Row */}
      <div className="metrics-row">
        <div className="metric-card">
          <div>
            <div className="metric-label">Total Extracted Tasks</div>
            <div className="metric-number">{totalTasks}</div>
          </div>
          <div className="metric-icon-box indigo">
            <FileText size={17} />
          </div>
        </div>

        <div className="metric-card">
          <div>
            <div className="metric-label">Pending Confirmation</div>
            <div className="metric-number">{totalPending}</div>
          </div>
          <div className="metric-icon-box amber">
            <Clock size={17} />
          </div>
        </div>

        <div className="metric-card">
          <div>
            <div className="metric-label">Confirmed & Ready</div>
            <div className="metric-number">{totalConfirmed}</div>
          </div>
          <div className="metric-icon-box green">
            <CheckCircle2 size={17} />
          </div>
        </div>

        <div className="metric-card">
          <div>
            <div className="metric-label">Needs Human Review</div>
            <div className="metric-number">{totalReview}</div>
          </div>
          <div className="metric-icon-box slate">
            <AlertTriangle size={17} />
          </div>
        </div>
      </div>

      {/* Recent Meetings Panel */}
      <div className="card-panel">
        <div className="card-header">
          <div className="card-header-title">
            <span>Recent Processed Meetings</span>
            <span style={{ fontSize: '0.78rem', color: 'var(--text-muted)', fontWeight: 400 }}>
              ({totalMeetings})
            </span>
          </div>

          {totalMeetings > 0 && (
            <button 
              className="btn-secondary" 
              style={{ padding: '0.35rem 0.65rem', fontSize: '0.78rem' }}
              onClick={onNewMeeting}
            >
              <Plus size={13} />
              Process New
            </button>
          )}
        </div>

        {totalMeetings === 0 ? (
          <div className="empty-state">
            <div className="empty-icon-wrap">
              <FolderOpen size={20} />
            </div>
            <div className="empty-title">No meetings recorded yet</div>
            <p className="empty-desc">
              Paste or upload a meeting transcript to extract actionable tasks, owners, and deadlines.
            </p>
            <button className="btn-primary" onClick={onNewMeeting}>
              <Plus size={15} />
              Start First Extraction
            </button>
          </div>
        ) : (
          <div className="table-wrapper">
            <table className="clean-table">
              <thead>
                <tr>
                  <th style={{ width: '36%' }}>Title</th>
                  <th style={{ width: '13%' }}>Date</th>
                  <th style={{ width: '13%' }}>Engine</th>
                  <th style={{ width: '9%' }}>Tasks</th>
                  <th style={{ width: '12%' }}>Status</th>
                  <th style={{ width: '17%', textAlign: 'right' }}>Actions</th>
                </tr>
              </thead>
              <tbody>
                {meetings.map((m) => (
                  <tr key={m.id}>
                    <td>
                      <span
                        style={{ fontWeight: 500, color: 'var(--primary-600)', cursor: 'pointer' }}
                        onClick={() => onOpenMeeting(m.id)}
                        title="Open meeting details"
                      >
                        {m.title}
                      </span>
                    </td>
                    <td style={{ color: 'var(--text-secondary)' }}>
                      {m.meeting_date}
                    </td>
                    <td>
                      <span className="badge badge-mode">
                        {m.extraction_mode || 'hybrid'}
                      </span>
                    </td>
                    <td>
                      <span style={{ fontWeight: 600 }}>{m.task_count}</span>
                    </td>
                    <td>
                      {m.review_needed_count > 0 ? (
                        <span className="badge-review" title="Has tasks flagged for human review">
                          <AlertTriangle size={11} />
                          {m.review_needed_count} review
                        </span>
                      ) : (
                        <span style={{ fontSize: '0.78rem', color: 'var(--success-text)', fontWeight: 500 }}>
                          Verified
                        </span>
                      )}
                    </td>
                    <td style={{ textAlign: 'right' }}>
                      <div style={{ display: 'inline-flex', alignItems: 'center', gap: '0.25rem' }}>
                        <button
                          className="btn-icon"
                          title="Open Meeting Tasks"
                          onClick={(e) => {
                            e.stopPropagation();
                            onOpenMeeting(m.id);
                          }}
                        >
                          <ExternalLink size={14} />
                        </button>
                        <button
                          className="btn-icon"
                          title="Download CSV"
                          onClick={(e) => {
                            e.stopPropagation();
                            handleExport(m, 'csv');
                          }}
                        >
                          <Download size={14} />
                        </button>
                        <button
                          className="btn-icon"
                          title="Download JSON"
                          style={{ fontSize: '0.65rem', fontWeight: 700, padding: '0.25rem 0.45rem' }}
                          onClick={(e) => {
                            e.stopPropagation();
                            handleExport(m, 'json');
                          }}
                        >
                          JSON
                        </button>
                        <button
                          className="btn-icon danger"
                          title="Delete Meeting"
                          onClick={(e) => {
                            e.stopPropagation();
                            onDeleteMeeting(m);
                          }}
                        >
                          <Trash2 size={14} />
                        </button>
                      </div>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
