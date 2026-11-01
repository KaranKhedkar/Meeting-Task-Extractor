import React from 'react';
import { 
  CheckSquare, 
  LayoutDashboard, 
  PlusCircle, 
  FolderClock,
  BarChart2,
  Cpu
} from 'lucide-react';

export default function Sidebar({ currentView, setCurrentView, backendStatus, onNewExtraction }) {
  return (
    <aside className="sidebar">
      {/* Brand Header */}
      <div className="sidebar-header">
        <div 
          className="sidebar-brand" 
          onClick={() => setCurrentView('dashboard')}
          title="Meeting Task Extractor"
        >
          <div className="brand-icon">
            <CheckSquare size={16} strokeWidth={2.5} />
          </div>
          <div className="brand-info">
            <span className="brand-title">Task Extractor</span>
            <span className="brand-tag">v2.0 Neural + ASR</span>
          </div>
        </div>
      </div>

      {/* Navigation */}
      <nav className="sidebar-nav">
        <button
          className={`sidebar-item ${currentView === 'dashboard' ? 'active' : ''}`}
          onClick={() => setCurrentView('dashboard')}
        >
          <LayoutDashboard size={16} className="item-icon" />
          <span>Dashboard</span>
        </button>

        <button
          className={`sidebar-item ${currentView === 'extract' ? 'active' : ''}`}
          onClick={onNewExtraction}
        >
          <PlusCircle size={16} className="item-icon" />
          <span>New Extraction</span>
        </button>

        <button
          className={`sidebar-item ${currentView === 'meetings' ? 'active' : ''}`}
          onClick={() => setCurrentView('meetings')}
        >
          <FolderClock size={16} className="item-icon" />
          <span>Saved Meetings</span>
        </button>

        <button
          className={`sidebar-item ${currentView === 'benchmark' ? 'active' : ''}`}
          onClick={() => setCurrentView('benchmark')}
        >
          <BarChart2 size={16} className="item-icon" />
          <span>Model Benchmark</span>
        </button>
      </nav>

      {/* Footer System Status */}
      <div className="sidebar-footer">
        <div 
          className="backend-indicator"
          title={backendStatus.online ? `Model: ${backendStatus.model || 'en_core_web_sm'} | DistilBERT + Whisper loaded` : 'Backend offline'}
        >
          <span className={`status-dot ${backendStatus.online ? '' : 'offline'}`} />
          <span>{backendStatus.online ? 'NLP & ASR Active' : 'Backend Offline'}</span>
        </div>
      </div>
    </aside>
  );
}
