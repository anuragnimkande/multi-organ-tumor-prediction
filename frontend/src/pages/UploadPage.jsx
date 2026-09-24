import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import axios from 'axios'
import toast from 'react-hot-toast'

import UploadZone       from '../components/UploadZone'
import LoadingAnimation from '../components/LoadingAnimation'

const API_BASE = import.meta.env.VITE_API_URL || ''

export default function UploadPage() {
  const navigate    = useNavigate()
  const [file,    setFile]    = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)

  // Handle file selection from UploadZone
  const handleFile = useCallback((f) => {
    setFile(f)
    const reader = new FileReader()
    reader.onload = (e) => setPreview(e.target.result)
    reader.readAsDataURL(f)
  }, [])

  // Clear selection
  const handleClear = () => {
    setFile(null)
    setPreview(null)
  }

  // Submit to Flask /predict
  const handleAnalyze = async () => {
    if (!file) {
      toast.error('Please upload an MRI image first.')
      return
    }

    setLoading(true)

    try {
      const formData = new FormData()
      formData.append('file', file)

      const response = await axios.post(`${API_BASE}/predict`, formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120_000,   // 2 min — quantum inference can be slow on CPU
      })

      // Store result in sessionStorage and navigate to result page
      sessionStorage.setItem('quantum_result',   JSON.stringify(response.data))
      sessionStorage.setItem('quantum_preview',  preview)
      navigate('/result')

    } catch (err) {
      setLoading(false)

      if (err.code === 'ECONNABORTED') {
        toast.error('Request timed out. The server may be starting up.')
      } else if (err.response) {
        const msg = err.response.data?.error || 'Server returned an error.'
        toast.error(`Error ${err.response.status}: ${msg}`)
      } else {
        toast.error('Cannot connect to backend. Ensure Flask is running on port 5000.')
      }
    }
  }

  return (
    <div className="min-h-screen flex flex-col items-center justify-start pt-12 pb-20 px-4">
      {loading && <LoadingAnimation />}

      <div className="w-full max-w-2xl">

        {/* Header */}
        <div className="text-center mb-10">
          <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full
                          border border-quantum-500/30 bg-quantum-500/5 text-quantum-500
                          text-sm font-medium mb-4">
            <span className="w-2 h-2 rounded-full bg-quantum-500 animate-pulse" />
            MRI Analysis
          </div>
          <h1 className="font-display font-bold text-4xl text-white mb-3">
            Upload MRI Scan
          </h1>
          <p className="text-slate-400 text-base max-w-md mx-auto">
            Upload a grayscale MRI brain scan image. The hybrid quantum model
            will analyze it for tumor indicators.
          </p>
        </div>

        {/* Upload zone */}
        <UploadZone onFile={handleFile} disabled={loading} />

        {/* Image preview */}
        {preview && !loading && (
          <div id="image-preview-card" className="mt-6 glass-card p-4 animate-slide-up">
            <div className="flex items-center justify-between mb-3">
              <span className="text-sm font-medium text-slate-300 font-display">
                Preview
              </span>
              <button
                id="clear-image-button"
                onClick={handleClear}
                className="text-xs text-slate-500 hover:text-slate-300 transition-colors px-2 py-1
                           rounded border border-slate-700 hover:border-slate-500"
              >
                ✕ Clear
              </button>
            </div>

            <div className="flex gap-4 items-start">
              {/* Image */}
              <div className="w-36 h-36 rounded-xl overflow-hidden border border-quantum-500/20
                              bg-void-900 flex-shrink-0 flex items-center justify-center">
                <img
                  id="preview-image"
                  src={preview}
                  alt="MRI preview"
                  className="w-full h-full object-contain filter grayscale"
                />
              </div>

              {/* File info */}
              <div className="flex-1 space-y-2 pt-1">
                <div>
                  <span className="text-xs text-slate-500 uppercase tracking-wide">Filename</span>
                  <p className="text-sm text-slate-200 font-mono truncate">{file?.name}</p>
                </div>
                <div>
                  <span className="text-xs text-slate-500 uppercase tracking-wide">Size</span>
                  <p className="text-sm text-slate-300">
                    {(file?.size / 1024).toFixed(1)} KB
                  </p>
                </div>
                <div>
                  <span className="text-xs text-slate-500 uppercase tracking-wide">Type</span>
                  <p className="text-sm text-slate-300">{file?.type || 'image'}</p>
                </div>

                {/* Ready badge */}
                <div className="inline-flex items-center gap-1.5 text-healthy text-xs
                                bg-healthy/10 border border-healthy/30 px-2.5 py-1 rounded-full">
                  <div className="w-1.5 h-1.5 rounded-full bg-healthy animate-pulse" />
                  Ready to analyze
                </div>
              </div>
            </div>
          </div>
        )}

        {/* Tips */}
        {!preview && (
          <div className="mt-6 grid grid-cols-1 sm:grid-cols-3 gap-3">
            {[
              { icon: '🧲', tip: 'Axial view MRI works best' },
              { icon: '⚫', tip: 'Grayscale images preferred' },
              { icon: '📐', tip: 'Any resolution accepted' },
            ].map((item, i) => (
              <div key={i} className="glass-card px-4 py-3 flex items-center gap-2 text-sm text-slate-500">
                <span>{item.icon}</span>
                <span>{item.tip}</span>
              </div>
            ))}
          </div>
        )}

        {/* Analyze button */}
        <div className="mt-8 flex flex-col gap-3">
          <button
            id="analyze-button"
            onClick={handleAnalyze}
            disabled={!file || loading}
            className={`btn-quantum w-full text-base py-4 justify-center
              ${(!file || loading) ? 'opacity-50 cursor-not-allowed hover:scale-100' : ''}`}
          >
            {loading ? '⚛ Analyzing...' : '⚛️ Run Quantum Analysis →'}
          </button>

          <p className="text-center text-xs text-slate-600">
            ⏱ Inference may take 15–30 seconds on CPU (quantum simulation)
          </p>
        </div>
      </div>
    </div>
  )
}
