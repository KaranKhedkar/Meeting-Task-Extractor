import React, { useState, useEffect, useCallback } from 'react';
import { Plus } from 'lucide-react';
import Sidebar from './components/Sidebar';
import Dashboard from './components/Dashboard';
import TranscriptEditor from './components/TranscriptEditor';
import TaskTable from './components/TaskTable';
import TaskDetailModal from './components/TaskDetailModal';
import EditTaskModal from './components/EditTaskModal';
import ConfirmModal from './components/ConfirmModal';
import BenchmarkView from './components/BenchmarkView';
import Toast from './components/Toast';
import { api } from './services/api';
import './styles/App.css';

export default function App() {
  const [currentView, setCurrentView] = useState('dashboard'); // 'dashboard', 'extract', 'review', 'meetings'
  const [meetings, setMeetings] = useState([]);
  const [activeMeeting, setActiveMeeting] = useState(null);
  const [backendStatus, setBackendStatus] = useState({ online: false, model: '', ml_model_loaded: false });
  const [selectedTaskDetails, setSelectedTaskDetails] = useState(null);
  const [editingTask, setEditingTask] = useState(null);
  const [toasts, setToasts] = useState([]);
  const [deleteConfirmState, setDeleteConfirmState] = useState({
    isOpen: false,
    type: null, // 'meeting' | 'task'
    item: null,
    title: '',
    message: '',
  });

  const showToast = useCallback((message, type = 'info') => {
    const id = Date.now().toString();
    setToasts((prev) => [...prev, { id, message, type }]);
    setTimeout(() => {
      setToasts((prev) => prev.filter((t) => t.id !== id));
    }, 4000);
  }, []);

  const dismissToast = (id) => {
    setToasts((prev) => prev.filter((t) => t.id !== id));
  };

  const refreshMeetings = useCallback(async () => {
    try {
      const data = await api.getMeetings();
      setMeetings(data.meetings || []);
    } catch (err) {
      console.warn('Could not load meetings list:', err.message);
    }
  }, []);

  useEffect(() => {
    async function checkSystem() {
      try {
        const health = await api.checkHealth();
        setBackendStatus({
          online: true,
          model: health.model,
          ml_model_loaded: health.ml_model_loaded,
        });
      } catch {
        setBackendStatus({ online: false, model: '', ml_model_loaded: false });
      }
    }

    checkSystem();
    refreshMeetings();
    const interval = setInterval(checkSystem, 15000);
    return () => clearInterval(interval);
  }, [refreshMeetings]);

  const handleExtractSuccess = ({ title, meetingDate, transcript, extractionMode, tasks }) => {
    setActiveMeeting({
      id: null,
      title,
      meetingDate,
      transcript,
      extractionMode,
      tasks,
    });
    setCurrentView('review');
  };

  const handleOpenMeeting = async (meetingId) => {
    try {
      const data = await api.getMeeting(meetingId);
      setActiveMeeting({
        id: data.id,
        title: data.title,
        meetingDate: data.meeting_date,
        transcript: data.transcript,
        extractionMode: data.extraction_mode,
        tasks: data.tasks || [],
      });
      setCurrentView('review');
    } catch (err) {
      showToast(err.message, 'error');
    }
  };

  const handleSaveActiveMeeting = async () => {
    if (!activeMeeting) return;

    try {
      if (activeMeeting.id) {
        for (const t of activeMeeting.tasks) {
          await api.updateTask(t.id, {
            description: t.description,
            assignee: t.assignee,
            deadline: t.deadline,
            deadline_text: t.deadline_text,
            status: t.status,
            needs_review: t.needs_review,
          });
        }
        showToast('Meeting and tasks updated successfully', 'success');
      } else {
        const saved = await api.saveMeeting({
          title: activeMeeting.title,
          meeting_date: activeMeeting.meetingDate,
          transcript: activeMeeting.transcript,
          extraction_mode: activeMeeting.extractionMode,
          tasks: activeMeeting.tasks,
        });
        setActiveMeeting({
          ...activeMeeting,
          id: saved.id,
          tasks: saved.tasks,
        });
        showToast('Meeting saved to database', 'success');
      }
      refreshMeetings();
    } catch (err) {
      showToast(err.message, 'error');
    }
  };

  const handleUpdateTask = async (taskId, updates) => {
    if (!activeMeeting) return;

    const updatedTasks = activeMeeting.tasks.map((t) =>
      t.id === taskId ? { ...t, ...updates } : t
    );
    setActiveMeeting({ ...activeMeeting, tasks: updatedTasks });

    if (activeMeeting.id) {
      try {
        await api.updateTask(taskId, updates);
        refreshMeetings();
      } catch (err) {
        showToast(err.message, 'error');
      }
    }
  };

  // Triggers the in-app confirmation modal for deleting a task
  const handleDeleteTask = (taskOrId) => {
    const taskId = typeof taskOrId === 'object' ? taskOrId.id : taskOrId;
    const taskDesc = typeof taskOrId === 'object' ? taskOrId.description : 'this task';

    setDeleteConfirmState({
      isOpen: true,
      type: 'task',
      item: taskId,
      title: 'Delete Task',
      message: `Are you sure you want to delete "${taskDesc}"?`,
    });
  };

  // Triggers the in-app confirmation modal for deleting a meeting
  const handleDeleteMeeting = (meetingOrId) => {
    const meetingId = typeof meetingOrId === 'object' ? meetingOrId.id : meetingOrId;
    const meetingTitle = typeof meetingOrId === 'object' ? meetingOrId.title : 'this meeting';

    setDeleteConfirmState({
      isOpen: true,
      type: 'meeting',
      item: meetingId,
      title: 'Delete Meeting',
      message: `Are you sure you want to permanently delete "${meetingTitle}" and all its extracted tasks? This action cannot be undone.`,
    });
  };

  // Executes the confirmed deletion
  const handleExecuteDelete = async () => {
    const { type, item } = deleteConfirmState;
    setDeleteConfirmState((prev) => ({ ...prev, isOpen: false }));

    if (type === 'meeting') {
      try {
        await api.deleteMeeting(item);
        showToast('Meeting deleted successfully', 'success');
        if (activeMeeting && activeMeeting.id === item) {
          setActiveMeeting(null);
          setCurrentView('dashboard');
        }
        refreshMeetings();
      } catch (err) {
        showToast(`Failed to delete meeting: ${err.message}`, 'error');
      }
    } else if (type === 'task') {
      if (!activeMeeting) return;

      const updatedTasks = activeMeeting.tasks.filter((t) => t.id !== item);
      setActiveMeeting({ ...activeMeeting, tasks: updatedTasks });

      if (activeMeeting.id) {
        try {
          await api.deleteTask(item);
          showToast('Task deleted successfully', 'info');
          refreshMeetings();
        } catch (err) {
          showToast(`Failed to delete task: ${err.message}`, 'error');
        }
      } else {
        showToast('Task removed from workspace', 'info');
      }
    }
  };

  return (
    <div className="app-shell">
      {/* Restrained Left Sidebar */}
      <Sidebar
        currentView={currentView}
        setCurrentView={setCurrentView}
        backendStatus={backendStatus}
        onNewExtraction={() => {
          setActiveMeeting(null);
          setCurrentView('extract');
        }}
      />

      {/* Main Content Area */}
      <div className="main-wrapper">
        <header className="top-bar">
          <div className="breadcrumb-label">
            <span>Meeting Task Extractor</span>
            <span>/</span>
            <strong>
              {currentView === 'dashboard' && 'Workspace Overview'}
              {currentView === 'extract' && 'New Meeting Extraction'}
              {currentView === 'review' && (activeMeeting?.title || 'Review Tasks')}
              {currentView === 'meetings' && 'Saved Meetings'}
              {currentView === 'benchmark' && 'Model Evaluation & Benchmark'}
            </strong>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            {currentView !== 'extract' && (
              <button
                className="btn-primary"
                style={{ fontSize: '0.8rem', padding: '0.38rem 0.75rem' }}
                onClick={() => {
                  setActiveMeeting(null);
                  setCurrentView('extract');
                }}
              >
                <Plus size={14} />
                New Extraction
              </button>
            )}
          </div>
        </header>

        <main className="page-container">
          {currentView === 'dashboard' && (
            <Dashboard
              meetings={meetings}
              onOpenMeeting={handleOpenMeeting}
              onNewMeeting={() => setCurrentView('extract')}
              onDeleteMeeting={handleDeleteMeeting}
              showToast={showToast}
            />
          )}

          {currentView === 'extract' && (
            <TranscriptEditor
              onExtractSuccess={handleExtractSuccess}
              showToast={showToast}
            />
          )}

          {currentView === 'review' && activeMeeting && (
            <TaskTable
              meetingData={activeMeeting}
              onSaveMeeting={handleSaveActiveMeeting}
              onUpdateTask={handleUpdateTask}
              onDeleteTask={handleDeleteTask}
              onOpenDetails={(task) => setSelectedTaskDetails(task)}
              onOpenEdit={(task) => setEditingTask(task)}
              showToast={showToast}
            />
          )}

          {currentView === 'meetings' && (
            <Dashboard
              meetings={meetings}
              onOpenMeeting={handleOpenMeeting}
              onNewMeeting={() => setCurrentView('extract')}
              onDeleteMeeting={handleDeleteMeeting}
              showToast={showToast}
            />
          )}

          {currentView === 'benchmark' && (
            <BenchmarkView showToast={showToast} />
          )}
        </main>
      </div>

      {/* Task Details Modal */}
      {selectedTaskDetails && (
        <TaskDetailModal
          task={selectedTaskDetails}
          onClose={() => setSelectedTaskDetails(null)}
        />
      )}

      {/* Edit Task Modal */}
      {editingTask && (
        <EditTaskModal
          task={editingTask}
          onSave={handleUpdateTask}
          onDelete={handleDeleteTask}
          onClose={() => setEditingTask(null)}
        />
      )}

      {/* In-App Delete Confirmation Modal */}
      <ConfirmModal
        isOpen={deleteConfirmState.isOpen}
        title={deleteConfirmState.title}
        message={deleteConfirmState.message}
        confirmLabel={deleteConfirmState.type === 'meeting' ? 'Delete Meeting' : 'Delete Task'}
        cancelLabel="Cancel"
        isDanger={true}
        onConfirm={handleExecuteDelete}
        onCancel={() => setDeleteConfirmState((prev) => ({ ...prev, isOpen: false }))}
      />

      {/* Toast Notifications */}
      <Toast toasts={toasts} onDismiss={dismissToast} />
    </div>
  );
}
