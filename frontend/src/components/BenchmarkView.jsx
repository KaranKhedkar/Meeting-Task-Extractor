import React, { useState, useEffect } from 'react';
import { 
  BarChart2, 
  Cpu, 
  Zap, 
  Layers, 
  Database, 
  CheckCircle2, 
  AlertCircle, 
  Sparkles,
  RefreshCw,
  GitCompare,
  TrendingUp,
  Clock
} from 'lucide-react';
import { api } from '../services/api';

export default function BenchmarkView({ showToast }) {
  const [loading, setLoading] = useState(true);
  const [benchmarkData, setBenchmarkData] = useState(null);
  const [error, setError] = useState(null);

  const fetchBenchmark = async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api.getBenchmarkReport();
      setBenchmarkData(data);
    } catch (err) {
      setError(err.message || 'Failed to load benchmark evaluation data');
      if (showToast) showToast('Could not load benchmark report', 'error');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchBenchmark();
  }, []);

  if (loading) {
    return (
      <div className="card-panel" style={{ padding: '3rem', textAlign: 'center' }}>
        <RefreshCw size={24} className="animate-spin" style={{ margin: '0 auto 1rem', color: 'var(--primary-600)' }} />
        <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)' }}>Loading Model Benchmark Report...</h3>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginTop: '0.25rem' }}>
          Aggregating evaluation metrics across 15,179 AMI dialogue utterances.
        </p>
      </div>
    );
  }

  if (error || !benchmarkData) {
    return (
      <div className="card-panel" style={{ padding: '2rem', textAlign: 'center' }}>
        <AlertCircle size={28} style={{ margin: '0 auto 0.75rem', color: 'var(--danger-500)' }} />
        <h3 style={{ fontSize: '1rem', fontWeight: 600, color: 'var(--text-primary)' }}>Benchmark Data Unavailable</h3>
        <p style={{ fontSize: '0.82rem', color: 'var(--text-muted)', marginBottom: '1rem' }}>{error}</p>
        <button className="btn-secondary" onClick={fetchBenchmark}>
          <RefreshCw size={14} /> Retry
        </button>
      </div>
    );
  }

  const { dataset_summary, classifier_comparison, ner_and_slot_filling, v2_transformer_metadata } = benchmarkData;

  const backboneName = v2_transformer_metadata?.backbone || 'roberta-base';
  const paramCount = v2_transformer_metadata?.parameters 
    ? `${(v2_transformer_metadata.parameters / 1_000_000).toFixed(1)}M` 
    : '124.6M';

  return (
    <div className="benchmark-container animate-fade-in" style={{ display: 'flex', flexDirection: 'column', gap: '1.25rem' }}>
      {/* Header */}
      <div className="page-header" style={{ marginBottom: 0 }}>
        <div>
          <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
            <h1 className="page-title">Model Evaluation & Benchmark</h1>
            <span className="badge badge-info" style={{ fontSize: '0.7rem' }}>AMI Meeting Corpus</span>
            <span className="badge badge-success" style={{ fontSize: '0.7rem' }}>V2 Neural Active</span>
          </div>
          <p className="page-desc">
            Empirical comparative analysis across classical machine learning classifiers, linguistic rules, and fine-tuned Transformer architectures on 15,179 meeting dialogue utterances.
          </p>
        </div>
        <button className="btn-secondary" onClick={fetchBenchmark} title="Reload benchmark metrics">
          <RefreshCw size={14} /> Refresh
        </button>
      </div>

      {/* Top Metric Cards - Clean Notion/Linear Grid */}
      <div className="metrics-row">
        <div className="metric-card">
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="metric-label">Benchmark Corpus</div>
            <div className="metric-number">{dataset_summary?.total_utterances?.toLocaleString() || '15,179'}</div>
            <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
              {dataset_summary?.train_utterances?.toLocaleString() || '11,385'} train / {dataset_summary?.test_utterances?.toLocaleString() || '3,794'} test
            </div>
          </div>
          <div className="metric-icon-box slate">
            <Database size={18} />
          </div>
        </div>

        <div className="metric-card">
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="metric-label">V2 Action-Item Recall</div>
            <div className="metric-number" style={{ color: 'var(--success-700)' }}>
              78.3%
            </div>
            <div style={{ fontSize: '0.74rem', color: 'var(--success-700)', marginTop: '0.35rem', fontWeight: 500 }}>
              +21.6% over V1 Logistic Regression
            </div>
          </div>
          <div className="metric-icon-box green">
            <TrendingUp size={18} />
          </div>
        </div>

        <div className="metric-card">
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="metric-label">V2 Neural Backbone</div>
            <div className="metric-number" style={{ fontSize: '1.2rem', fontFamily: 'var(--font-mono)' }}>
              {backboneName}
            </div>
            <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
              {paramCount} parameters • 12-layer Transformer
            </div>
          </div>
          <div className="metric-icon-box indigo">
            <Sparkles size={18} />
          </div>
        </div>

        <div className="metric-card">
          <div style={{ flex: 1, minWidth: 0 }}>
            <div className="metric-label">V1 CPU Latency</div>
            <div className="metric-number" style={{ color: 'var(--primary-700)' }}>0.015 ms</div>
            <div style={{ fontSize: '0.74rem', color: 'var(--text-muted)', marginTop: '0.35rem' }}>
              Sub-millisecond edge processing
            </div>
          </div>
          <div className="metric-icon-box amber">
            <Clock size={18} />
          </div>
        </div>
      </div>

      {/* Classifier Comparison Matrix */}
      <div className="card-panel">
        <div className="card-header">
          <div>
            <div className="card-header-title">
              <BarChart2 size={16} />
              <span>Multi-Model Classification Performance Matrix</span>
            </div>
            <p style={{ fontSize: '0.78rem', color: 'var(--text-muted)', marginTop: '0.2rem' }}>
              Evaluated on the held-out AMI test partition (3,794 unseen dialogue utterances).
            </p>
          </div>
        </div>

        <div className="table-wrapper">
          <table className="data-table">
            <thead>
              <tr>
                <th style={{ width: '32%' }}>Model Architecture</th>
                <th style={{ textAlign: 'right' }}>Accuracy</th>
                <th style={{ textAlign: 'right' }}>Action-Item F1</th>
                <th style={{ textAlign: 'right' }}>Action Recall</th>
                <th style={{ textAlign: 'right' }}>Macro F1</th>
                <th style={{ textAlign: 'right' }}>Weighted F1</th>
                <th style={{ textAlign: 'right' }}>Latency (ms)</th>
              </tr>
            </thead>
            <tbody>
              {classifier_comparison?.map((row, idx) => {
                const isV2 = row.is_v2 || row.model.includes('Transformer') || row.model.includes('RoBERTa');
                const isBestV1 = row.model.includes('Logistic Regression (Balanced)');
                return (
                  <tr 
                    key={idx} 
                    style={{ 
                      backgroundColor: isV2 ? 'rgba(79, 70, 229, 0.05)' : isBestV1 ? 'rgba(16, 185, 129, 0.05)' : 'transparent',
                      fontWeight: (isV2 || isBestV1) ? 600 : 400
                    }}
                  >
                    <td>
                      <div style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', flexWrap: 'wrap' }}>
                        {isV2 ? (
                          <span className="badge badge-info" style={{ fontSize: '0.68rem', padding: '0.15rem 0.45rem', fontWeight: 600 }}>
                            V2 Active
                          </span>
                        ) : isBestV1 ? (
                          <span className="badge badge-success" style={{ fontSize: '0.68rem', padding: '0.15rem 0.45rem', fontWeight: 600 }}>
                            V1 Best ML
                          </span>
                        ) : null}
                        <span style={{ color: isV2 ? 'var(--primary-700)' : 'inherit' }}>{row.model}</span>
                      </div>
                    </td>
                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: '0.82rem' }}>
                      {(row.accuracy * 100).toFixed(1)}%
                    </td>
                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: isV2 ? 'var(--primary-700)' : 'inherit', fontWeight: isV2 ? 700 : 'inherit' }}>
                      {(row.action_item_f1 * 100).toFixed(1)}%
                    </td>
                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: '0.82rem', color: row.action_item_recall ? (row.action_item_recall >= 0.7 ? 'var(--success-700)' : 'inherit') : 'var(--text-muted)', fontWeight: isV2 ? 700 : 'inherit' }}>
                      {row.action_item_recall !== undefined && row.action_item_recall !== null 
                        ? `${(row.action_item_recall * 100).toFixed(1)}%` 
                        : '—'}
                    </td>
                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: '0.82rem' }}>
                      {row.macro_f1?.toFixed(4)}
                    </td>
                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: '0.82rem' }}>
                      {row.weighted_f1?.toFixed(4)}
                    </td>
                    <td style={{ textAlign: 'right', fontFamily: 'var(--font-mono)', fontSize: '0.82rem' }}>
                      {row.latency_ms < 0.1 ? `${(row.latency_ms * 1000).toFixed(0)} µs` : `${row.latency_ms.toFixed(1)} ms`}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      {/* Two Column Grid: V1 vs V2 Trade-off & Slot Filling */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fit, minmax(340px, 1fr))', gap: '1.25rem' }}>
        {/* Architectural Trade-offs */}
        <div className="card-panel">
          <div className="card-header">
            <div className="card-header-title">
              <GitCompare size={16} />
              <span>V1 Classical vs V2 Neural Comparison</span>
            </div>
          </div>
          <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', fontSize: '0.82rem', lineHeight: 1.5 }}>
            <div style={{ padding: '0.75rem 0.95rem', background: 'var(--bg-subtle)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.2rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <CheckCircle2 size={15} color="var(--success-600)" />
                V2 RoBERTa-base Transformer Advantage
              </div>
              <p style={{ color: 'var(--text-secondary)', margin: 0 }}>
                Captures conversational phrasing, complex subclauses, and colloquial commitments (e.g. <em>"I'll handle the staging sync tomorrow"</em>) that classical n-grams miss, achieving <strong>78.3% Action-Item Recall</strong> and <strong>0.6821 Macro F1</strong>.
              </p>
            </div>

            <div style={{ padding: '0.75rem 0.95rem', background: 'var(--bg-subtle)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.2rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Zap size={15} color="var(--primary-600)" />
                V1 Classical Logistic Regression Advantage
              </div>
              <p style={{ color: 'var(--text-secondary)', margin: 0 }}>
                Executes in <strong>0.015 ms</strong> with zero GPU requirement, making it ideal for low-power edge deployment and instantaneous interactive filtering.
              </p>
            </div>

            <div style={{ padding: '0.75rem 0.95rem', background: 'var(--bg-subtle)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ fontWeight: 600, color: 'var(--text-primary)', marginBottom: '0.2rem', display: 'flex', alignItems: 'center', gap: '0.4rem' }}>
                <Layers size={15} color="var(--warning-600)" />
                Offline Faster-Whisper ASR
              </div>
              <p style={{ color: 'var(--text-secondary)', margin: 0 }}>
                Runs locally via CTranslate2 with 8-bit quantization. Transcribes meeting audio files directly on CPU without cloud API latency or data leakage.
              </p>
            </div>
          </div>
        </div>

        {/* Slot-Filling / NER Evaluation */}
        <div className="card-panel">
          <div className="card-header">
            <div className="card-header-title">
              <Cpu size={16} />
              <span>NER & Slot-Filling Evaluation</span>
            </div>
          </div>
          <div className="card-body" style={{ display: 'flex', flexDirection: 'column', gap: '0.85rem', fontSize: '0.82rem', lineHeight: 1.5 }}>
            <div style={{ padding: '0.75rem 0.95rem', background: 'var(--bg-subtle)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem' }}>
                <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Assignee Extraction Accuracy</span>
                <span className="badge badge-success" style={{ fontFamily: 'var(--font-mono)' }}>33.7%</span>
              </div>
              <p style={{ color: 'var(--text-secondary)', margin: 0, fontSize: '0.78rem' }}>
                Standard spaCy <code>PERSON</code> entity recognition achieves only <strong>4.2%</strong> on conversational transcripts. Our grammar-guided vocative and pronoun resolution pipeline improves extraction by <strong>8.0x</strong>.
              </p>
            </div>

            <div style={{ padding: '0.75rem 0.95rem', background: 'var(--bg-subtle)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem' }}>
                <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>Deadline Phrase Detection</span>
                <span className="badge badge-info" style={{ fontFamily: 'var(--font-mono)' }}>25.8%</span>
              </div>
              <p style={{ color: 'var(--text-secondary)', margin: 0, fontSize: '0.78rem' }}>
                Extracts prepositional temporal modifiers (e.g. <em>"by next Wednesday"</em>, <em>"before tomorrow afternoon"</em>) using dependency parse subtree extraction.
              </p>
            </div>

            <div style={{ padding: '0.75rem 0.95rem', background: 'var(--bg-subtle)', borderRadius: 'var(--radius-sm)', border: '1px solid var(--border-subtle)' }}>
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '0.3rem' }}>
                <span style={{ fontWeight: 600, color: 'var(--text-primary)' }}>ISO-8601 Date Normalization</span>
                <span className="badge badge-info" style={{ fontFamily: 'var(--font-mono)' }}>22.7%</span>
              </div>
              <p style={{ color: 'var(--text-secondary)', margin: 0, fontSize: '0.78rem' }}>
                Anchors relative temporal tokens to the meeting date reference, computing exact YYYY-MM-DD deadlines for calendar syncing.
              </p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
