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
  const fileInputRef = useRef(null);
  const referenceInputRef = useRef(null);

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

      const modelHeader = response.headers.get('X-Model-Name');
      const sourceHeader = response.headers.get('X-Selected-Source-Index');
      const simsHeader = response.headers.get('X-Speaker-Similarities');
      const marginHeader = response.headers.get('X-Speaker-Confidence-Margin');

      setBackendModelUsed(modelHeader || selectedModel);
      setSelectedSourceIndex(sourceHeader);
      if (simsHeader) {
        setSimilarities(simsHeader.split(',').map((s) => parseFloat(s)));
      }
      if (marginHeader) {
        setConfidenceMargin(parseFloat(marginHeader));
      }

      const blob = await response.blob();
      const url = URL.createObjectURL(blob);
      setAudioUrl(url);
      setElapsed(((performance.now() - startTime) / 1000).toFixed(2));
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