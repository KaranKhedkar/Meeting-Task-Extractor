const API_BASE_URL = import.meta.env.VITE_API_URL || '/api';

/**
 * Handles HTTP fetch calls and parses JSON response or raises descriptive Error.
 */
async function request(endpoint, options = {}) {
  const url = `${API_BASE_URL}${endpoint}`;
  const headers = { ...options.headers };

  // Only set Content-Type for requests that send a JSON body
  if (options.body && !(options.body instanceof FormData) && !headers['Content-Type']) {
    headers['Content-Type'] = 'application/json';
  }

  let response;
  try {
    response = await fetch(url, {
      ...options,
      headers,
    });
  } catch (netErr) {
    // If relative proxy failed or offline, try direct backend URL as fallback
    if (url.startsWith('/api')) {
      const fallbackUrl = `http://127.0.0.1:8000${url}`;
      try {
        response = await fetch(fallbackUrl, {
          ...options,
          headers,
        });
      } catch {
        throw new Error(`Network error connecting to backend: ${netErr.message}`);
      }
    } else {
      throw new Error(`Network error connecting to backend: ${netErr.message}`);
    }
  }

  if (response.status === 204) {
    return null;
  }

  if (!response.ok) {
    let errorDetail = 'An unexpected error occurred.';
    try {
      const errorData = await response.json();
      errorDetail = errorData.detail || errorDetail;
    } catch {
      errorDetail = `Request failed with status ${response.status}: ${response.statusText}`;
    }
    throw new Error(errorDetail);
  }

  return response.json();
}

/**
 * Programmatically triggers a file download in the browser using a Blob and temporary anchor element.
 */
export function triggerBlobDownload(blob, filename) {
  const url = window.URL.createObjectURL(blob);
  const a = document.createElement('a');
  a.style.display = 'none';
  a.href = url;
  a.download = filename;
  document.body.appendChild(a);
  a.click();
  setTimeout(() => {
    document.body.removeChild(a);
    window.URL.revokeObjectURL(url);
  }, 250);
}

/**
 * Generates and downloads a client-side CSV or JSON file from the given task array.
 */
export function downloadClientExport(tasks = [], format = 'csv', title = 'meeting') {
  const safeTitle = (title || 'meeting').replace(/[^a-zA-Z0-9_-]/g, '_').replace(/_+/g, '_');
  const filename = `${safeTitle}_tasks.${format}`;

  if (format === 'json') {
    const jsonStr = JSON.stringify(tasks, null, 2);
    const blob = new Blob([jsonStr], { type: 'application/json;charset=utf-8;' });
    triggerBlobDownload(blob, filename);
  } else {
    // RFC-4180 compliant CSV generator with UTF-8 BOM for spreadsheet compatibility
    const headers = ['ID', 'Description', 'Assignee', 'Deadline', 'Deadline Text', 'Status', 'Confidence', 'Needs Review', 'Evidence'];
    const rows = [headers.join(',')];

    tasks.forEach((t) => {
      const esc = (val) => {
        if (val === null || val === undefined) return '""';
        return `"${String(val).replace(/"/g, '""')}"`;
      };

      rows.push([
        esc(t.id || ''),
        esc(t.description || ''),
        esc(t.assignee || 'Unassigned'),
        esc(t.deadline || ''),
        esc(t.deadline_text || ''),
        esc(t.status || 'pending'),
        esc(typeof t.confidence === 'number' ? t.confidence.toFixed(2) : t.confidence || '0.80'),
        esc(t.needs_review ? 'Yes' : 'No'),
        esc(t.evidence || ''),
      ].join(','));
    });

    // \uFEFF BOM ensures Excel opens UTF-8 encoded characters properly
    const csvContent = '\uFEFF' + rows.join('\r\n');
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    triggerBlobDownload(blob, filename);
  }
}

export const api = {
  // Check backend health
  async checkHealth() {
    try {
      const res = await fetch('/health');
      if (res.ok) return res.json();
    } catch {
      // fallback to direct backend host
    }
    const directRes = await fetch('http://127.0.0.1:8000/health');
    if (!directRes.ok) throw new Error('Backend health check failed');
    return directRes.json();
  },

  // Extract tasks from transcript text
  async extractTasks({ transcript, meeting_date, extraction_mode = 'hybrid' }) {
    return request('/meetings/extract', {
      method: 'POST',
      body: JSON.stringify({
        transcript,
        meeting_date,
        extraction_mode,
      }),
    });
  },

  // Upload TXT or VTT file
  async uploadTranscript(file) {
    const formData = new FormData();
    formData.append('file', file);
    return request('/meetings/upload', {
      method: 'POST',
      body: formData,
    });
  },

  // Transcribe audio file using local Faster-Whisper (V2 Audio Pipeline)
  async transcribeAudio(file) {
    const formData = new FormData();
    formData.append('file', file);
    return request('/meetings/transcribe-audio', {
      method: 'POST',
      body: formData,
    });
  },

  // Get V1 vs V2 model benchmark report
  async getBenchmarkReport() {
    return request('/benchmark', {
      method: 'GET',
    });
  },

  // Save meeting & tasks to database
  async saveMeeting(payload) {
    return request('/meetings', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
  },

  // List all saved meetings
  async getMeetings() {
    return request('/meetings', {
      method: 'GET',
    });
  },

  // Get single meeting by ID
  async getMeeting(meetingId) {
    return request(`/meetings/${meetingId}`, {
      method: 'GET',
    });
  },

  // Delete meeting
  async deleteMeeting(meetingId) {
    return request(`/meetings/${meetingId}`, {
      method: 'DELETE',
    });
  },

  // Update task fields / status
  async updateTask(taskId, updates) {
    return request(`/tasks/${taskId}`, {
      method: 'PATCH',
      body: JSON.stringify(updates),
    });
  },

  // Delete task
  async deleteTask(taskId) {
    return request(`/tasks/${taskId}`, {
      method: 'DELETE',
    });
  },

  // Get direct export download URL
  getExportUrl(meetingId, format = 'csv') {
    return `${API_BASE_URL}/meetings/${meetingId}/export?format=${format}`;
  },

  /**
   * Downloads export file reliably via fetch and blob download.
   * If meeting is saved, fetches from backend; falls back to client-side generator.
   */
  async exportTasks({ meetingId, format = 'csv', title = 'meeting', tasks = [] }) {
    const safeTitle = (title || 'meeting').replace(/[^a-zA-Z0-9_-]/g, '_').replace(/_+/g, '_');
    const filename = `${safeTitle}_tasks.${format}`;

    if (meetingId) {
      try {
        const url = this.getExportUrl(meetingId, format);
        let res;
        try {
          res = await fetch(url);
        } catch {
          // Fallback to direct backend URL
          res = await fetch(`http://127.0.0.1:8000/api/meetings/${meetingId}/export?format=${format}`);
        }

        if (res && res.ok) {
          const blob = await res.blob();
          triggerBlobDownload(blob, filename);
          return;
        }
      } catch (err) {
        console.warn('Backend export failed, falling back to client-side generation:', err);
      }
    }

    // Client-side fallback
    downloadClientExport(tasks, format, title);
  },
};
