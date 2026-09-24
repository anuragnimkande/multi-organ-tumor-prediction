import { useEffect, useState } from 'react'
import { useNavigate, Link } from 'react-router-dom'
import axios from 'axios'
import toast from 'react-hot-toast'

import ResultCard from '../components/ResultCard'

const API_BASE = import.meta.env.VITE_API_URL || ''

export default function ResultPage() {
  const navigate   = useNavigate()
  const [result,   setResult]   = useState(null)
  const [preview,  setPreview]  = useState(null)
  const [compare,  setCompare]  = useState(null)
  const [compLoading, setCompLoading] = useState(false)

  // Load result from sessionStorage (set by UploadPage after inference)
  useEffect(() => {
    const storedResult  = sessionStorage.getItem('quantum_result')
    const storedPreview = sessionStorage.getItem('quantum_preview')

    if (!storedResult) {
      toast.error('No result found. Please upload an MRI scan first.')
      navigate('/upload')
      return
    }

    setResult(JSON.parse(storedResult))
    setPreview(storedPreview)
  }, [navigate])

  // Handle "analyze another" — clear state and go back
  const handleReset = () => {
    sessionStorage.removeItem('quantum_result')
    sessionStorage.removeItem('quantum_preview')
    navigate('/upload')
  }

  // Fetch classical vs hybrid comparison
  const fetchComparison = async () => {
    const storedPreview = sessionStorage.getItem('quantum_preview')
    if (!storedPreview) return

    setCompLoading(true)
    try {
      // Convert base64 data URI → Blob
      const response = await fetch(storedPreview)
      const blob = await response.blob()
      const file = new File([blob], 'mri.jpg', { type: blob.type })

      const formData = new FormData()
      formData.append('file', file)

      const res = await axios.post(`${API_BASE}/compare`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120_000,
      })
      setCompare(res.data)
    } catch (err) {
      toast.error('Comparison failed. Ensure both model checkpoints exist.')
    } finally {
      setCompLoading(false)
    }
  }

  if (!result) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-slate-500 text-sm">Loading result...</div>
      </div>
    )
  }

  const isTumor = result.prediction === 'Tumor'

  return (
    <div className="min-h-screen flex flex-col items-center justify-start pt-10 pb-20 px-4">
      <div className="w-full max-w-2xl space-y-6">

        {/* Breadcrumb */}
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <Link to="/" id="result-breadcrumb-home" className="hover:text-slate-300 transition-colors">Home</Link>
          <span>›</span>
          <Link to="/upload" id="result-breadcrumb-upload" className="hover:text-slate-300 transition-colors">Upload</Link>
          <span>›</span>
          <span className="text-slate-300">Result</span>
        </div>

        {/* Page title */}
        <div>
          <h1 className="font-display font-bold text-3xl text-white">
            Analysis Complete
          </h1>
          <p className="text-slate-400 text-sm mt-1">
            Hybrid Quantum-Classical inference result
          </p>
        </div>

        {/* Main result card */}
        <ResultCard result={result} previewSrc={preview} onReset={handleReset} />

        {/* ── Classical vs Hybrid Comparison ── */}
        <div id="comparison-section" className="glass-card p-6 space-y-4">
          <div className="flex items-center justify-between">
            <h2 className="font-display font-semibold text-lg text-white">
              Model Comparison
            </h2>
            {!compare && (
              <button
                id="compare-button"
                onClick={fetchComparison}
                disabled={compLoading}
                className={`btn-outline text-sm py-2 px-4 ${compLoading ? 'opacity-50 cursor-not-allowed' : ''}`}
              >
                {compLoading ? '⚛ Running...' : 'Compare Models →'}
              </button>
            )}
          </div>

          {!compare && !compLoading && (
            <p className="text-slate-500 text-sm">
              Compare predictions from the classical CNN baseline vs the hybrid
              quantum model. Requires both checkpoints to be trained.
            </p>
          )}

          {compLoading && (
            <div className="flex items-center gap-3 py-4 text-slate-400 text-sm">
              <div className="w-4 h-4 rounded-full border-2 border-quantum-500 border-t-transparent animate-spin" />
              Running classical + quantum inference...
            </div>
          )}

          {compare && (
            <div id="comparison-table" className="space-y-3 animate-slide-up">
              {[
                {
                  id:    'classical',
                  label: '🔷 Classical CNN (ResNet18)',
                  data:  compare.classical,
                },
                {
                  id:    'hybrid',
                  label: '⚛️ Hybrid Quantum-Classical',
                  data:  compare.hybrid,
                },
              ].map(({ id, label, data }) => {
                const isT = data.prediction === 'Tumor'
                return (
                  <div key={id} className={`rounded-xl p-4 border
                    ${isT
                      ? 'bg-tumor/5 border-tumor/20'
                      : 'bg-healthy/5 border-healthy/20'
                    }`}>
                    <div className="flex items-center justify-between flex-wrap gap-2">
                      <span className="text-sm font-medium text-slate-300">{label}</span>
                      <span className={`text-sm font-bold ${isT ? 'text-tumor' : 'text-healthy'}`}>
                        {data.prediction}
                      </span>
                    </div>
                    <div className="mt-2 flex gap-6 text-xs text-slate-500">
                      <span>Confidence: <strong className="text-slate-300">
                        {(data.confidence * 100).toFixed(1)}%
                      </strong></span>
                      <span>Raw prob: <strong className="text-slate-300 font-mono">
                        {data.raw_prob.toFixed(4)}
                      </strong></span>
                    </div>
                  </div>
                )
              })}
            </div>
          )}
        </div>

        {/* ── Technical Details ── */}
        <div id="technical-details" className="glass-card p-6 space-y-3">
          <h2 className="font-display font-semibold text-lg text-white mb-4">
            Technical Pipeline
          </h2>
          <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-sm">
            {[
              { label: 'Input Shape',      value: '(1, 224, 224)' },
              { label: 'CNN Backbone',     value: 'ResNet18 (ImageNet)' },
              { label: 'Feature Dim',      value: '128-dimensional' },
              { label: 'Quantum Device',   value: 'default.qubit' },
              { label: 'Qubits',           value: '4' },
              { label: 'VQC Layers',       value: '2' },
              { label: 'Encoding',         value: 'AngleEmbedding (RY)' },
              { label: 'Measurement',      value: 'PauliZ ⟨Z⟩' },
            ].map(({ label, value }) => (
              <div key={label} className="flex justify-between rounded-lg
                                          bg-void-800/60 border border-white/5 px-3 py-2">
                <span className="text-slate-500">{label}</span>
                <span className="text-slate-300 font-mono text-xs">{value}</span>
              </div>
            ))}
          </div>
        </div>

        {/* Disclaimer */}
        <p className="text-xs text-slate-600 text-center leading-relaxed px-4">
          ⚠️ This AI system is for research purposes only. Not intended as a substitute
          for clinical diagnosis. Consult a qualified medical professional.
        </p>
      </div>
    </div>
  )
}
