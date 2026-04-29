'use client';
import { useEffect, useState, useRef } from 'react';
import styles from './page.module.css';

const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL || 'http://localhost:8000';

const FALLBACK_MODELS = {
  speechbrain: {
    id: 'speechbrain',
    available: true,
    label: 'SpeechBrain SepFormer (Baseline)',
    architecture: 'SepFormer (energy-based selection)',
    dataset: 'WSJ0-2Mix',
    benchmark: '22.3 dB SI-SDRi',
    supports_reference: false,
  },
  ecw_tse: {
    id: 'ecw_tse',
    available: true,
    label: 'ECW-TSE (Novel Pipeline)',
    architecture: 'SepFormer + ECAPA-TDNN + Embedding-Conditioned Wiener mask',
    dataset: 'WSJ0-2Mix (sep) + VoxCeleb (embed)',
    benchmark: 'Reference-aware target speaker extraction',
    supports_reference: true,
  },
  math_model: {
    id: 'math_model',
    available: true,
    label: 'My Custom TF-GridNet',
    architecture: 'TF-GridNet + ratio-mask refinement',
    dataset: 'Libri2Mix (project)',
    benchmark: 'Project-trained',
    supports_reference: false,
  },
};

export default function Home() {
  const [file, setFile] = useState(null);
  const [referenceFile, setReferenceFile] = useState(null);
  const [selectedModel, setSelectedModel] = useState('ecw_tse');
  const [modelInfo, setModelInfo] = useState(FALLBACK_MODELS);
  const [backendModelUsed, setBackendModelUsed] = useState(null);
  const [selectedSourceIndex, setSelectedSourceIndex] = useState(null);
  const [similarities, setSimilarities] = useState(null);
  const [confidenceMargin, setConfidenceMargin] = useState(null);

  const [isExtracting, setIsExtracting] = useState(false);
  const [error, setError] = useState(null);
  const [audioUrl, setAudioUrl] = useState(null);
  const [elapsed, setElapsed] = useState(null);
  const [metrics, setMetrics] = useState(null);
  const [viz, setViz] = useState(null);
  const fileInputRef = useRef(null);
  const referenceInputRef = useRef(null);
  const mixtureCanvasRef = useRef(null);
  const targetCanvasRef = useRef(null);

  const activeModel = modelInfo[selectedModel] || FALLBACK_MODELS[selectedModel];
  const isSelectedModelAvailable = Boolean(activeModel?.available);
  const supportsReference = Boolean(activeModel?.supports_reference);

  useEffect(() => {
    let isCancelled = false;

    const loadModels = async () => {
      try {
        const response = await fetch(`${API_BASE_URL}/models`);
        if (!response.ok) {
          return;
        }

        const payload = await response.json();
        if (isCancelled) {
          return;
        }

        const merged = { ...FALLBACK_MODELS };
        if (Array.isArray(payload?.models)) {
          for (const model of payload.models) {
            if (!model?.id || !merged[model.id]) {
              continue;
            }
            merged[model.id] = {
              ...merged[model.id],
              ...model,
              available: Boolean(model.available),
            };
          }
        }

        setModelInfo(merged);

        setSelectedModel((previous) => {
          if (payload?.default && merged[payload.default]?.available) {
            return payload.default;
          }
          if (merged[previous]?.available) {
            return previous;
          }
          const firstAvailable = Object.keys(merged).find((id) => merged[id].available);
          return firstAvailable || previous;
        });
      } catch {
        // Keep fallback models if backend metadata cannot be fetched.
      }
    };

    loadModels();

    return () => {
      isCancelled = true;
    };
  }, []);

  const handleUploadTrigger = () => {
    fileInputRef.current.click();
  };

  const handleReferenceTrigger = () => {
    referenceInputRef.current?.click();
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setAudioUrl(null);
      setError(null);
      setElapsed(null);
      setBackendModelUsed(null);
      setSelectedSourceIndex(null);
      setSimilarities(null);
      setConfidenceMargin(null);
    }
  };

  const handleReferenceChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setReferenceFile(e.target.files[0]);
    } else {
      setReferenceFile(null);
    }
  };

  const clearReference = () => {
    setReferenceFile(null);
    if (referenceInputRef.current) {
      referenceInputRef.current.value = '';
    }
  };

  const extractAudio = async () => {
    if (!file) {
      setError('Please upload an audio file first.');
      return;
    }

    if (!isSelectedModelAvailable) {
      setError('Selected model is unavailable on the backend.');
      return;
    }

    setIsExtracting(true);
    setError(null);
    setAudioUrl(null);
    setElapsed(null);
    setBackendModelUsed(null);
    setSelectedSourceIndex(null);
    setSimilarities(null);
    setConfidenceMargin(null);
    setMetrics(null);
    setViz(null);

    const startTime = performance.now();
    const formData = new FormData();
    formData.append('mixture_file', file);
    formData.append('model_name', selectedModel);
    if (supportsReference && referenceFile) {
      formData.append('reference_file', referenceFile);
    }

    try {
      const response = await fetch(`${API_BASE_URL}/extract_voice`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        let detail = 'Separation failed. Is the backend running?';
        try {
          const errorPayload = await response.json();
          if (errorPayload?.detail) {
            detail = errorPayload.detail;
          }
        } catch {
          // Keep default fallback message when no JSON body is returned.
        }
        throw new Error(detail);
      }

      const payload = await response.json();
      setBackendModelUsed(payload.model_used || selectedModel);
      setSelectedSourceIndex(payload.selected_source_index ?? null);
      setSimilarities(payload.similarities ?? null);
      setConfidenceMargin(payload.confidence_margin ?? null);
      setMetrics(payload.metrics ?? null);
      setViz(payload.viz ?? null);

      // Decode base64 WAV → blob URL for the <audio> element
      const audioFmt = payload.audio_format || 'wav';
      const binStr = atob(payload.audio_b64);
      const bytes = new Uint8Array(binStr.length);
      for (let i = 0; i < binStr.length; i++) bytes[i] = binStr.charCodeAt(i);
      const blob = new Blob([bytes], { type: `audio/${audioFmt}` });
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
      setElapsed(payload.elapsed_seconds?.toFixed?.(2) ?? ((performance.now() - startTime) / 1000).toFixed(2));
    } catch (err) {
      setError(err.message);
    } finally {
      setIsExtracting(false);
    }
  };

  useEffect(() => {
    return () => {
      if (audioUrl) {
        URL.revokeObjectURL(audioUrl);
      }
    };
  }, [audioUrl]);

  const drawWaveform = (canvas, samples, color) => {
    if (!canvas || !samples?.length) return;
    const ctx = canvas.getContext('2d');
    const w = canvas.width;
    const h = canvas.height;
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = '#0a0e1a';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = '#1f2937';
    ctx.beginPath();
    ctx.moveTo(0, h / 2);
    ctx.lineTo(w, h / 2);
    ctx.stroke();

    const N = samples.length;
    let peak = 0;
    for (let i = 0; i < N; i++) {
      const a = Math.abs(samples[i]);
      if (a > peak) peak = a;
    }
    const norm = peak > 0 ? 1 / peak : 1;

    ctx.strokeStyle = color;
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const x = (i / (N - 1)) * w;
      const y = h / 2 - samples[i] * norm * (h / 2 - 4);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    }
    ctx.stroke();
  };

  useEffect(() => {
    if (!viz) return;
    drawWaveform(mixtureCanvasRef.current, viz.mixture_waveform, '#888');
    drawWaveform(targetCanvasRef.current, viz.target_waveform, '#22c55e');
  }, [viz]);

  const waveBars = Array.from({ length: 40 }, (_, i) => {
    const h = 4 + ((Math.sin(i * 3.7 + 1.3) + 1) / 2) * 8;
    return { id: i, height: h };
  });

  return (
    <main className={styles.main}>
      <header className={styles.header}>
        <h1 className={styles.title}>Neural Voice Separation</h1>
        <p className={styles.subtitle}>
          Compare a SOTA baseline, a novel embedding-conditioned pipeline, and a custom architecture for isolating one voice from a mixture.
        </p>
        <div className={styles.headerBadge}>
          <span className="status-badge live">
            {activeModel?.label || 'Model Selected'}
          </span>
        </div>
      </header>

      <div className={styles.workspace}>
        <div className={`glass-panel ${styles.panel}`}>
          <div className={styles.panelHeader}>
            <h2 className={styles.panelTitle}>
              <span className={styles.panelIcon}>🎙️</span> Input Mixture
            </h2>
          </div>

          <input
            type="file"
            ref={fileInputRef}
            onChange={handleFileChange}
            accept="audio/*"
            style={{ display: 'none' }}
            id="audio-upload-input"
          />

          <div className={styles.uploadZone} onClick={handleUploadTrigger} id="upload-zone">
            <span className={styles.uploadIcon}>🎧</span>
            {file ? (
              <p className={styles.fileName}>📎 {file.name}</p>
            ) : (
              <p>Click to Upload Crowded Room Audio</p>
            )}
            <p className={styles.uploadHint}>WAV, FLAC, OGG — up to 25MB</p>
          </div>

          <div className={styles.controlGroup}>
            <label className={styles.controlLabel} htmlFor="model-select">Model Engine</label>
            <select
              id="model-select"
              className={styles.selectControl}
              value={selectedModel}
              onChange={(e) => setSelectedModel(e.target.value)}
            >
              <option value="ecw_tse" disabled={!modelInfo.ecw_tse?.available}>
                ECW-TSE (Novel Pipeline)
              </option>
              <option value="speechbrain" disabled={!modelInfo.speechbrain.available}>
                SpeechBrain SepFormer (Baseline)
              </option>
              <option value="math_model" disabled={!modelInfo.math_model.available}>
                Custom TF-GridNet
              </option>
            </select>
            <p className={styles.controlHint}>
              {activeModel?.available
                ? activeModel.label
                : 'Selected model is unavailable on backend.'}
            </p>
          </div>

          {supportsReference && (
            <div className={styles.controlGroup}>
              <label className={styles.controlLabel}>
                Reference Speaker (optional)
              </label>
              <input
                type="file"
                ref={referenceInputRef}
                onChange={handleReferenceChange}
                accept="audio/*"
                style={{ display: 'none' }}
                id="reference-upload-input"
              />
              <div className={styles.uploadZone} onClick={handleReferenceTrigger}>
                <span className={styles.uploadIcon}>🎯</span>
                {referenceFile ? (
                  <p className={styles.fileName}>📎 {referenceFile.name}</p>
                ) : (
                  <p>Click to upload a clean clip of the target speaker</p>
                )}
                <p className={styles.uploadHint}>
                  Used to compute ECAPA-TDNN embedding for source selection
                </p>
              </div>
              {referenceFile && (
                <button
                  type="button"
                  className={styles.controlHint}
                  onClick={clearReference}
                  style={{ background: 'none', border: 'none', cursor: 'pointer', textDecoration: 'underline' }}
                >
                  Clear reference
                </button>
              )}
            </div>
          )}

          {file && (
            <div className={styles.metricsGrid}>
              <div className={styles.metricCard}>
                <div className={styles.metricLabel}>File Name</div>
                <div className={styles.metricValue}>
                  {file.name.length > 20 ? file.name.slice(0, 20) + '…' : file.name}
                </div>
              </div>
              <div className={styles.metricCard}>
                <div className={styles.metricLabel}>File Size</div>
                <div className={`${styles.metricValue} accent`}>
                  {(file.size / 1024 / 1024).toFixed(2)} MB
                </div>
              </div>
            </div>
          )}

          {error && <p className={styles.error}>⚠️ {error}</p>}
        </div>

        <div className={`glass-panel ${styles.panel}`}>
          <div className={styles.panelHeader}>
            <h2 className={styles.panelTitle}>
              <span className={styles.panelIcon}>🧠</span> Extracted Signal
            </h2>
            <button
              className="btn-primary"
              onClick={extractAudio}
              disabled={isExtracting || !file || !isSelectedModelAvailable}
              id="extract-button"
            >
              {isExtracting ? 'Separating...' : 'Extract'}
            </button>
          </div>

          <div className={styles.visualizerPlaceholder}>
            <div className={styles.waveformBars}>
              {waveBars.map((bar) => (
                <div
                  key={bar.id}
                  className={`${styles.waveBar} ${isExtracting ? styles.active : ''}`}
                  style={{
                    height: isExtracting ? undefined : `${bar.height}px`,
                    animationDelay: isExtracting ? `${bar.id * 0.05}s` : undefined,
                  }}
                />
              ))}
            </div>
            <div className={styles.waveLine} style={isExtracting ? { animation: 'pulseGlow 1.5s infinite' } : {}} />
            <span className={styles.statusText}>
              {isExtracting ? 'PROCESSING NEURAL GRAPH...' : audioUrl ? 'SEPARATION COMPLETE' : 'AWAITING SIGNAL'}
            </span>
          </div>

          {audioUrl && (
            <div className={styles.audioResult}>
              <span className={styles.audioLabel}>✅ Speaker Isolated Successfully</span>
              <audio controls src={audioUrl} className={styles.audioPlayer} id="audio-player" />
              <div className={styles.outputMeta}>
                <span>Model used: {backendModelUsed || selectedModel}</span>
                <span>Source index: {selectedSourceIndex ?? 'auto'}</span>
              </div>
              {similarities && (
                <div className={styles.outputMeta}>
                  <span>
                    Speaker similarities: [
                    {similarities.map((s, i) => (
                      <span key={i}>
                        {i > 0 ? ', ' : ''}
                        <span style={{ color: i === Number(selectedSourceIndex) ? '#22c55e' : '#888' }}>
                          {s.toFixed(3)}
                        </span>
                      </span>
                    ))}
                    ]
                  </span>
                  {confidenceMargin !== null && (
                    <span>Confidence margin: {confidenceMargin.toFixed(3)}</span>
                  )}
                </div>
              )}
            </div>
          )}

          {viz && (
            <div className={styles.audioResult}>
              <div style={{ marginBottom: 12, fontSize: 12, color: '#888', textTransform: 'uppercase', letterSpacing: 1.2 }}>
                Waveforms
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                <div>
                  <div style={{ fontSize: 11, color: '#888' }}>Mixture (input)</div>
                  <canvas ref={mixtureCanvasRef} width={560} height={70} style={{ width: '100%', borderRadius: 6, background: '#0a0e1a' }} />
                </div>
                <div>
                  <div style={{ fontSize: 11, color: '#22c55e' }}>Extracted target</div>
                  <canvas ref={targetCanvasRef} width={560} height={70} style={{ width: '100%', borderRadius: 6, background: '#0a0e1a' }} />
                </div>
              </div>

              <div style={{ marginTop: 16, fontSize: 12, color: '#888', textTransform: 'uppercase', letterSpacing: 1.2 }}>
                Spectrograms
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, marginTop: 6 }}>
                {viz.mixture_spectrogram_png && (
                  <img alt="Mixture spectrogram"
                       src={`data:image/png;base64,${viz.mixture_spectrogram_png}`}
                       style={{ width: '100%', borderRadius: 6 }} />
                )}
                {viz.target_spectrogram_png && (
                  <img alt="Target spectrogram"
                       src={`data:image/png;base64,${viz.target_spectrogram_png}`}
                       style={{ width: '100%', borderRadius: 6 }} />
                )}
              </div>

              {viz.mask_png && (
                <>
                  <div style={{ marginTop: 16, fontSize: 12, color: '#888', textTransform: 'uppercase', letterSpacing: 1.2 }}>
                    Fused MR-ECWM Mask
                  </div>
                  <img alt="ECWM mask"
                       src={`data:image/png;base64,${viz.mask_png}`}
                       style={{ width: '100%', borderRadius: 6, marginTop: 6 }} />
                </>
              )}
            </div>
          )}

          {metrics && (
            <div className={styles.audioResult}>
              <div style={{ marginBottom: 10, fontSize: 12, color: '#888', textTransform: 'uppercase', letterSpacing: 1.2 }}>
                Quality Metrics
              </div>
              <div className={styles.metricsGrid}>
                <div className={styles.metricCard}>
                  <div className={styles.metricLabel}>SI-SDRi (est)</div>
                  <div className={`${styles.metricValue} accent`}>
                    {metrics.si_sdr_improvement_db?.toFixed?.(2) ?? '—'} dB
                  </div>
                </div>
                <div className={styles.metricCard}>
                  <div className={styles.metricLabel}>Target similarity</div>
                  <div className={`${styles.metricValue} success`}>
                    {metrics.target_similarity?.toFixed?.(3) ?? '—'}
                  </div>
                </div>
                <div className={styles.metricCard}>
                  <div className={styles.metricLabel}>Confidence margin</div>
                  <div className={styles.metricValue}>
                    {metrics.confidence_margin?.toFixed?.(3) ?? '—'}
                  </div>
                </div>
                <div className={styles.metricCard}>
                  <div className={styles.metricLabel}>Voice activity</div>
                  <div className={`${styles.metricValue} secondary`}>
                    {metrics.voice_activity_ratio
                      ? (metrics.voice_activity_ratio * 100).toFixed(0) + '%'
                      : '—'}
                  </div>
                </div>
                <div className={styles.metricCard}>
                  <div className={styles.metricLabel}>ICR iterations</div>
                  <div className={styles.metricValue}>
                    {metrics.icr_iterations ?? 0}
                  </div>
                </div>
                <div className={styles.metricCard}>
                  <div className={styles.metricLabel}>Energy ratio (target)</div>
                  <div className={`${styles.metricValue} accent`}>
                    {metrics.energy_ratio_target
                      ? (metrics.energy_ratio_target * 100).toFixed(1) + '%'
                      : '—'}
                  </div>
                </div>
              </div>

              {metrics.icr_trace?.length > 1 && (
                <div style={{ marginTop: 10 }}>
                  <div style={{ fontSize: 11, color: '#888', textTransform: 'uppercase', letterSpacing: 1.2, marginBottom: 4 }}>
                    ICR convergence (α_target across iterations)
                  </div>
                  <div style={{ display: 'flex', gap: 4, alignItems: 'flex-end', height: 50 }}>
                    {metrics.icr_trace.map((step, i) => {
                      const alpha = step.alphas?.[0] ?? 0;
                      const heightPct = Math.max(2, Math.min(100, ((alpha + 1) / 2) * 100));
                      return (
                        <div key={i} style={{ flex: 1, textAlign: 'center' }}>
                          <div style={{
                            height: `${heightPct}%`,
                            background: 'linear-gradient(to top, #22c55e, #a855f7)',
                            borderRadius: 4,
                            marginBottom: 4,
                          }} />
                          <div style={{ fontSize: 10, color: '#888' }}>
                            {alpha.toFixed(2)}
                          </div>
                        </div>
                      );
                    })}
                  </div>
                </div>
              )}
            </div>
          )}

          <div className={styles.metricsGrid}>
            <div className={styles.metricCard}>
              <div className={styles.metricLabel}>Architecture</div>
              <div className={`${styles.metricValue} accent`}>{activeModel?.architecture || '—'}</div>
            </div>
            <div className={styles.metricCard}>
              <div className={styles.metricLabel}>Dataset</div>
              <div className={`${styles.metricValue} secondary`}>{activeModel?.dataset || '—'}</div>
            </div>
            <div className={styles.metricCard}>
              <div className={styles.metricLabel}>Benchmark</div>
              <div className={`${styles.metricValue} success`}>{activeModel?.benchmark || '—'}</div>
            </div>
            <div className={styles.metricCard}>
              <div className={styles.metricLabel}>Latency</div>
              <div className={styles.metricValue}>{elapsed ? `${elapsed}s` : '—'}</div>
            </div>
          </div>
        </div>
      </div>
    </main>
  );
}