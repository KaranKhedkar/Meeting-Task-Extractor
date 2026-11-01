import React from 'react';
import { CheckSquare, PlusCircle, LayoutDashboard, History, Sparkles } from 'lucide-react';

export default function Navbar({ currentView, setCurrentView, backendStatus, onNewExtraction }) {
  return (
    <header className="navbar">
      <div className="nav-inner">
        <div className="nav-brand" style={{ cursor: 'pointer' }} onClick={() => setCurrentView('dashboard')}>
          <div className="brand-icon">
            <CheckSquare size={22} />
          </div>
          <div>
            <div className="brand-title">
              Meeting Task Extractor
              <span className="brand-badge">V1 Classical NLP</span>
            </div>
          </div>
        </div>

        <nav className="nav-links">
          <button
            className={`nav-btn ${currentView === 'dashboard' ? 'active' : ''}`}
            onClick={() => setCurrentView('dashboard')}
          >
            <LayoutDashboard size={16} />
            Dashboard
          </button>

          <button
            className={`nav-btn ${currentView === 'extract' ? 'active' : ''}`}
            onClick={onNewExtraction}
          >
            <PlusCircle size={16} />
            New Meeting
          </button>

          <button
            className={`nav-btn ${currentView === 'meetings' ? 'active' : ''}`}
            onClick={() => setCurrentView('meetings')}
          >
            <History size={16} />
            Saved Meetings
          </button>

          <div
            className={`backend-badge ${backendStatus.online ? '' : 'offline'}`}
            title={backendStatus.online ? `Connected: ${backendStatus.model} | ML loaded: ${backendStatus.ml_model_loaded}` : 'Backend server offline'}
          >
            <span className="status-dot"></span>
            {backendStatus.online ? 'NLP Ready' : 'Backend Offline'}
          </div>
        </nav>
      </div>
    </header>
  );
}
