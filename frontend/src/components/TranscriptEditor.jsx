import React, { useState, useRef } from 'react';
import { 
  FileText, 
  Upload, 
  RotateCcw, 
  Play, 
  FileUp,
  Sparkles,
  Mic,
  Volume2,
  CheckCircle2,
  Loader2,
  Clock,
  Layers,
  Cpu
} from 'lucide-react';
import { api } from '../services/api';

const SAMPLE_TRANSCRIPTS = {
  sprint: {
    title: "Sprint 42 Planning & API Deliverables",
    date: new Date().toISOString().split('T')[0],
    text: `Rahul: Good morning everyone, let's start the sprint planning session.
Rahul: I'll finish the backend API by Wednesday.
Priya: Amit, please review the interface before Friday.
Amit: The login page was deployed yesterday afternoon.
Priya: Maybe we could redesign the navigation bar next sprint.
Sneha: We need to submit the architecture proposal next week.
Rahul: I already fixed the database migration issue this morning.
Amit: We discussed the possibility of using PostgreSQL instead.
Sneha: Could Amit check the latency logs before tomorrow?
Rahul: We don't need to update the legacy billing code anymore.`
  },
  design: {
    title: "UI/UX Mobile App Design Sync",
    date: new Date().toISOString().split('T')[0],
    text: `Priya: The color contrast on the dashboard looks much better now.
Priya: I will prepare the revised Figma mockups by tomorrow.
Vikram: What if we try a dark theme for the analytics view?
Priya: Vikram, please update the icon set by Thursday.
Vikram: I completed the mobile typography audit last Friday.
Neha: Someone should audit the accessibility contrast ratios.
Priya: I'll export the asset bundle for the frontend developers tonight.
Neha: I won't change the hero banner until marketing approves it.`
  },
  incident: {
    title: "Production Incident Retrospective",
    date: new Date().toISOString().split('T')[0],
    text: `Karan: Thanks for joining the post-incident review for yesterday's outage.
Karan: We rolled back the bad build yesterday within twenty minutes.
Rohan: I will increase the Redis cluster replica count by Monday.
Karan: Rohan, please configure automated memory alerts before Tuesday.
Deepa: How about we set up synthetic traffic monitoring?
Deepa: I'll draft the customer incident postmortem by end of day.
Deepa: We must update our runbook documentation within three days.`
  },
  vtt: {
    title: "Client Deliverables Sync (WebVTT)",
    date: new Date().toISOString().split('T')[0],
    text: `WEBVTT

1
00:00:01.000 --> 00:00:04.000
<v Dev>The client presentation is scheduled for next Friday.</v>

2
00:00:04.500 --> 00:00:08.000
<v Dev>Rahul, please prepare the presentation by Friday.</v>

3
00:00:08.500 --> 00:00:12.000
<v Rahul>I will polish the demonstration slides by Wednesday.</v>

4
00:00:12.500 --> 00:00:16.000
<v Dev>The sales team finalized the contract terms yesterday.</v>

5
00:00:16.500 --> 00:00:20.000
<v Shreya>I'll populate the staging database with dummy accounts tomorrow.</v>`
  }
};

export default function TranscriptEditor({ 
  onExtractSuccess, 
  showToast 
}) {
  const [title, setTitle] = useState("Weekly Team Engineering Sync");
  const [meetingDate, setMeetingDate] = useState(new Date().toISOString().split('T')[0]);
  const [extractionMode, setExtractionMode] = useState("v2_transformer"); // Default to V2 Transformer
  const [transcript, setTranscript] = useState(SAMPLE_TRANSCRIPTS.sprint.text);
  const [isExtracting, setIsExtracting] = useState(false);
  const [isUploading, setIsUploading] = useState(false);
  const [isTranscribing, setIsTranscribing] = useState(false);
  const [inputTab, setInputTab] = useState("text"); // "text" | "audio"
  const [audioFileInfo, setAudioFileInfo] = useState(null);
  const [audioTranscriptionMeta, setAudioTranscriptionMeta] = useState(null);

  const fileInputRef = useRef(null);
  const audioInputRef = useRef(null);

  const charCount = transcript.length;
  const wordCount = transcript.trim() ? transcript.trim().split(/\s+/).length : 0;
  const lineCount = transcript ? transcript.split('\n').filter(l => l.trim()).length : 0;

  const loadSample = (key) => {
    const s = SAMPLE_TRANSCRIPTS[key];
    if (s) {
      setTitle(s.title);
      setMeetingDate(s.date);
      setTranscript(s.text);
      setAudioTranscriptionMeta(null);
      showToast(`Loaded sample: ${s.title}`, 'info');
    }
  };

  const handleFileUpload = async (e) => {
    const file = e.target.files?.[0];
    if (!file) return;

    setIsUploading(true);
    try {
      const res = await api.uploadTranscript(file);
      setTranscript(res.content);
      const cleanName = file.name.replace(/\.[^/.]+$/, '').replace(/[-_]/g, ' ');
      setTitle(cleanName.charAt(0).toUpperCase() + cleanName.slice(1));
      setAudioTranscriptionMeta(null);
      showToast(`Uploaded ${file.name} (${res.character_count} characters)`, 'success');
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setIsUploading(false);
      if (fileInputRef.current) fileInputRef.current.value = '';
    }
  };

  const handleAudioSelected = (e) => {
    const file = e.target.files?.[0];
    if (!file) return;
    setAudioFileInfo({
      file,
      name: file.name,
      sizeMb: (file.size / (1024 * 1024)).toFixed(2),
    });
  };

  const handleTranscribeAudio = async () => {
    if (!audioFileInfo?.file) {
      showToast('Please select an audio file to transcribe.', 'error');
      return;
    }

    setIsTranscribing(true);
    try {
      const res = await api.transcribeAudio(audioFileInfo.file);
      if (!res.formatted_transcript && (!res.segments || res.segments.length === 0)) {
        showToast('Transcription finished but no spoken dialogue was detected.', 'warning');
      } else {
        setTranscript(res.formatted_transcript);
        setAudioTranscriptionMeta({
          filename: res.filename,
          duration: res.duration_seconds,
          segments: res.segment_count,
        });
        const cleanName = audioFileInfo.name.replace(/\.[^/.]+$/, '').replace(/[-_]/g, ' ');
        setTitle(cleanName.charAt(0).toUpperCase() + cleanName.slice(1) + ' (Audio)');
        showToast(`Transcribed ${res.segment_count} dialogue segments (${res.duration_seconds.toFixed(1)}s)`, 'success');
        setInputTab('text'); // Switch to editor view to inspect transcript
      }
    } catch (err) {
      showToast(err.message || 'Audio transcription failed', 'error');
    } finally {
      setIsTranscribing(false);
      if (audioInputRef.current) audioInputRef.current.value = '';
    }
  };

  const handleExtract = async () => {
    if (!transcript.trim()) {
      showToast('Please enter a meeting transcript or upload a file.', 'error');
      return;
    }
    if (!meetingDate) {
      showToast('Please select a reference meeting date to resolve relative deadlines.', 'error');
      return;
    }

    setIsExtracting(true);
    try {
      const result = await api.extractTasks({
        transcript,
        meeting_date: meetingDate,
        extraction_mode: extractionMode,
      });

      const modeLabel = extractionMode === 'v2_transformer' ? 'V2 DistilBERT Transformer' : 'V1 Balanced Logistic Regression';
      showToast(`Extracted ${result.tasks.length} actionable tasks using ${modeLabel}`, 'success');
      onExtractSuccess({
        title,
        meetingDate,
        transcript,
        extractionMode: result.extraction_mode,
        tasks: result.tasks,
        stats: {
          utterances: result.utterance_count,
          actionItems: result.action_item_count,
          reviewNeeded: result.review_needed_count,
        }
      });
    } catch (err) {
      showToast(err.message, 'error');
    } finally {
      setIsExtracting(false);
    }
  };

  return (
    <div className="card-panel animate-fade-in">
      <div className="card-header">
        <div>
          <div className="card-header-title">
            <FileText size={16} />
            <span>Transcript Processing Workspace</span>
          </div>
          <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.15rem' }}>
            Configure meeting parameters and provide dialogue via text, subtitles, or offline audio transcription.
          </p>
        </div>

        <div className="sample-bar" style={{ margin: 0 }}>
          <span className="sample-label">Demonstration Samples:</span>
          <button type="button" className="sample-chip" onClick={() => loadSample('sprint')}>
            Sprint Planning
          </button>
          <button type="button" className="sample-chip" onClick={() => loadSample('design')}>
            Design Sync
          </button>
          <button type="button" className="sample-chip" onClick={() => loadSample('incident')}>
            Incident Review
          </button>
          <button type="button" className="sample-chip" onClick={() => loadSample('vtt')}>
            WebVTT File
          </button>
        </div>
      </div>

      <div className="card-body">
        {/* Form Configuration Inputs */}
        <div className="form-grid-3">
          <div className="form-group">
            <label className="form-label">Meeting Title</label>
            <input
              type="text"
              className="form-control"
              value={title}
              onChange={(e) => setTitle(e.target.value)}
              placeholder="e.g. Q4 Sprint Planning"
            />
          </div>

          <div className="form-group">
            <label className="form-label">Reference Meeting Date</label>
            <input
              type="date"
              className="form-control"
              value={meetingDate}
              onChange={(e) => setMeetingDate(e.target.value)}
            />
          </div>

          <div className="form-group">
            <label className="form-label" style={{ display: 'flex', justifyContent: 'space-between' }}>
              <span>NLP Classification Engine</span>
              <span style={{ fontSize: '0.72rem', color: 'var(--primary-600)', fontWeight: 600 }}>
                {extractionMode === 'v2_transformer' ? '75.1% Action Recall' : '0.015ms Latency'}
              </span>
            </label>
            <select
              className="form-control"
              value={extractionMode}
              onChange={(e) => setExtractionMode(e.target.value)}
              style={{ fontWeight: 500 }}
            >
              <option value="v2_transformer">
                V2: Fine-Tuned DistilBERT Transformer (Recommended)
              </option>
              <option value="hybrid">
                V1: Balanced Logistic Regression + Rules (Fast)
              </option>
            </select>
          </div>
        </div>

        {/* Mode Selector Tabs: Text/VTT vs Audio ASR */}
        <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', marginBottom: '0.75rem', borderBottom: '1px solid var(--border-subtle)', paddingBottom: '0.5rem' }}>
          <div style={{ display: 'flex', gap: '0.5rem' }}>
            <button
              type="button"
              className={`btn-secondary ${inputTab === 'text' ? 'active-filter' : ''}`}
              style={{ padding: '0.3rem 0.75rem', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}
              onClick={() => setInputTab('text')}
            >
              <FileText size={14} />
              <span>Transcript Text / VTT</span>
            </button>
            <button
              type="button"
              className={`btn-secondary ${inputTab === 'audio' ? 'active-filter' : ''}`}
              style={{ padding: '0.3rem 0.75rem', fontSize: '0.8rem', display: 'flex', alignItems: 'center', gap: '0.35rem' }}
              onClick={() => setInputTab('audio')}
            >
              <Mic size={14} />
              <span>Local Audio Transcription (Whisper)</span>
            </button>
          </div>

          {audioTranscriptionMeta && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '0.4rem', fontSize: '0.75rem', color: 'var(--success-700)', backgroundColor: 'var(--success-50)', padding: '0.2rem 0.5rem', borderRadius: 'var(--radius-sm)', border: '1px solid var(--success-200)' }}>
              <CheckCircle2 size={12} />
              <span>Transcribed {audioTranscriptionMeta.filename}: {audioTranscriptionMeta.duration.toFixed(1)}s ({audioTranscriptionMeta.segments} segments)</span>
            </div>
          )}
        </div>

        {/* Tab 1: Text Document Upload Dropzone */}
        {inputTab === 'text' && (
          <div className="upload-dropzone" onClick={() => fileInputRef.current?.click()} style={{ cursor: 'pointer' }}>
            <div className="upload-meta">
              <FileUp size={16} color="var(--primary-500)" />
              <div>
                <div className="upload-meta-text">
                  Upload transcript file (.txt, .vtt)
                </div>
                <div className="upload-meta-sub">
                  Supports speaker tags and standard WebVTT subtitle formatting
                </div>
              </div>
            </div>

            <button 
              type="button" 
              className="btn-secondary" 
              style={{ padding: '0.35rem 0.65rem', fontSize: '0.78rem' }}
              disabled={isUploading}
              onClick={(e) => {
                e.stopPropagation();
                fileInputRef.current?.click();
              }}
            >
              {isUploading ? 'Uploading...' : 'Browse File'}
            </button>

            <input
              type="file"
              ref={fileInputRef}
              accept=".txt,.vtt"
              className="hidden-file-input"
              onChange={handleFileUpload}
            />
          </div>
        )}

        {/* Tab 2: Audio Transcription Dropzone (Whisper ASR) */}
        {inputTab === 'audio' && (
          <div 
            style={{ 
              border: '1px dashed var(--border-subtle)', 
              borderRadius: 'var(--radius-md)', 
              padding: '1.25rem', 
              backgroundColor: 'var(--bg-subtle)',
              marginBottom: '1rem',
              display: 'flex',
              flexDirection: 'column',
              gap: '0.75rem'
            }}
          >
            <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between' }}>
              <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
                <div style={{ padding: '0.5rem', borderRadius: 'var(--radius-sm)', backgroundColor: 'rgba(79, 70, 229, 0.08)', color: 'var(--primary-600)' }}>
                  <Volume2 size={20} />
                </div>
                <div>
                  <div style={{ fontWeight: 600, fontSize: '0.85rem', color: 'var(--text-primary)' }}>
                    Faster-Whisper Offline Speech-to-Text Pipeline
                  </div>
                  <div style={{ fontSize: '0.78rem', color: 'var(--text-muted)' }}>
                    Upload recorded meeting audio (.wav, .mp3, .m4a, .ogg) to generate formatted speaker turn transcripts locally.
                  </div>
                </div>
              </div>

              <span className="badge badge-info" style={{ fontSize: '0.7rem' }}>
                Offline ASR (CTranslate2)
              </span>
            </div>

            <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.35rem' }}>
              <input
                type="file"
                ref={audioInputRef}
                accept=".wav,.mp3,.m4a,.ogg,.flac"
                className="hidden-file-input"
                onChange={handleAudioSelected}
              />
              <button
                type="button"
                className="btn-secondary"
                onClick={() => audioInputRef.current?.click()}
                disabled={isTranscribing}
                style={{ fontSize: '0.8rem' }}
              >
                <Mic size={14} />
                {audioFileInfo ? 'Choose Different Audio' : 'Select Audio File'}
              </button>

              {audioFileInfo && (
                <div style={{ fontSize: '0.82rem', color: 'var(--text-primary)', fontWeight: 500 }}>
                  {audioFileInfo.name} <span style={{ color: 'var(--text-muted)' }}>({audioFileInfo.sizeMb} MB)</span>
                </div>
              )}

              {audioFileInfo && (
                <button
                  type="button"
                  className="btn-primary"
                  onClick={handleTranscribeAudio}
                  disabled={isTranscribing}
                  style={{ marginLeft: 'auto', fontSize: '0.8rem' }}
                >
                  {isTranscribing ? (
                    <>
                      <Loader2 size={14} className="animate-spin" />
                      Transcribing with Whisper...
                    </>
                  ) : (
                    <>
                      <Play size={14} />
                      Transcribe Audio
                    </>
                  )}
                </button>
              )}
            </div>

            {isTranscribing && (
              <div style={{ marginTop: '0.5rem', padding: '0.5rem 0.75rem', backgroundColor: 'var(--surface)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)', fontSize: '0.78rem', color: 'var(--primary-700)', display: 'flex', alignItems: 'center', gap: '0.5rem' }}>
                <Loader2 size={13} className="animate-spin" />
                <span>Running CTranslate2 Whisper model locally on CPU. Audio will be segmented by acoustic pauses into dialogue turns...</span>
              </div>
            )}
          </div>
        )}

        {/* Transcript Textarea */}
        <div className="transcript-wrapper">
          <textarea
            className="transcript-input"
            value={transcript}
            onChange={(e) => setTranscript(e.target.value)}
            placeholder={`Enter meeting transcript...\n\nExample:\nRahul: I'll finish the backend API by Wednesday.\nPriya: Amit, please review the interface before Friday.`}
          />
        </div>

        {/* Bottom Bar: Stats & Extract CTA */}
        <div className="editor-bottom-bar">
          <div className="text-stats">
            <span>{charCount} characters</span>
            <span>•</span>
            <span>{wordCount} words</span>
            <span>•</span>
            <span>{lineCount} utterances</span>
            <span>•</span>
            <span style={{ color: 'var(--primary-600)', fontWeight: 500 }}>
              Mode: {extractionMode === 'v2_transformer' ? 'V2 DistilBERT' : 'V1 Balanced LR'}
            </span>
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '0.65rem' }}>
            <button
              type="button"
              className="btn-secondary"
              onClick={() => {
                setTranscript('');
                setAudioTranscriptionMeta(null);
              }}
              title="Clear transcript text"
            >
              <RotateCcw size={14} />
              Clear
            </button>

            <button
              type="button"
              className="btn-primary"
              onClick={handleExtract}
              disabled={isExtracting}
            >
              <Play size={14} className={isExtracting ? 'animate-spin' : ''} />
              {isExtracting ? 'Extracting Action Items...' : 'Extract Tasks'}
            </button>
          </div>
        </div>
      </div>
    </div>
  );
}
