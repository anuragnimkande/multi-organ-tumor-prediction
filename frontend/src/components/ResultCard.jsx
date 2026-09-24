import ConfidenceMeter from './ConfidenceMeter'
import HeatmapView from './HeatmapView'
import axios from 'axios'
import toast from 'react-hot-toast'

/**
 * ResultCard
 * Full prediction result display card.
 *
 * Props:
 *   result: {
 *     prediction:  "Tumor" | "No Tumor"
 *     confidence:  number  0–1
 *     raw_prob:    number
 *     gradcam:     string | null  (base64 PNG)
 *     model:       string
 *   }
 *   previewSrc:  string  — data URI of original uploaded image
 *   onReset:     () => void
 */
export default function ResultCard({ result, previewSrc, onReset, organColor = '#10b981' }) {
  const isAnomaly = result.prediction.toLowerCase().includes('tumor') || 
                    result.prediction.toLowerCase().includes('melanoma') ||
                    result.prediction.toLowerCase().includes('malignant');

  return (
    <div id="result-card" className="glass-card p-6 space-y-6 animate-slide-up" style={{ borderColor: `${organColor}30` }}>

      {/* Header badge */}
      <div className="flex flex-col items-center gap-3 py-2">
        <div className={`text-5xl mb-1 animate-float`}>
          {isAnomaly ? '⚠️' : '✅'}
        </div>

        <span className={isAnomaly ? 'badge-tumor text-lg px-6 py-2' : 'badge-healthy text-lg px-6 py-2'}>
          {isAnomaly ? '🔴 Anomaly Detected' : '🟢 No Anomaly Detected'}
        </span>

        <p className="text-slate-400 text-sm text-center max-w-sm">
          <strong className="text-white block mb-1">{result.prediction}</strong>
          {isAnomaly
            ? 'The model detected abnormal tissue patterns. Please consult a medical professional for clinical assessment.'
            : 'No significant abnormal indicators were found in this scan.'}
        </p>
      </div>

      <div className="border-t border-white/5" />

      {/* Confidence meter */}
      <ConfidenceMeter confidence={result.confidence} isTumor={isAnomaly} />

      <div className="border-t border-white/5" />

      {/* Grad-CAM + original side by side */}
      <HeatmapView
        originalSrc={previewSrc}
        heatmapSrc={result.gradcam}
      />

      <div className="border-t border-white/5" />

      {/* Technical details */}
      <div className="grid grid-cols-2 gap-3">
        <div className="metric-card">
          <span className="text-xs text-slate-500 uppercase tracking-wide">Anomaly Probability</span>
          <span className={`text-lg font-bold font-mono ${isAnomaly ? 'text-tumor' : 'text-healthy'}`}>
            {(result.raw_prob * 100).toFixed(2)}%
          </span>
        </div>
        <div className="metric-card">
          <span className="text-xs text-slate-500 uppercase tracking-wide">Decision Threshold</span>
          <span className="text-lg font-bold font-mono text-slate-300">
            50.00%
          </span>
        </div>
      </div>

      {/* Model info */}
      <div className="rounded-xl bg-void-800/60 border border-white/5 px-4 py-3 text-xs text-slate-500 font-mono">
        <span className="text-quantum-500">MODEL:</span> {result.model}
      </div>

      {/* Actions */}
      <div className="flex gap-3 pt-1">
        <button
          id="result-reset-button"
          onClick={onReset}
          className="flex-1 btn-outline"
        >
          ← Analyze Another
        </button>
        <button
          id="result-download-button"
          onClick={async () => {
            try {
              toast.loading('Generating PDF report...', { id: 'pdf-toast' })
              const response = await fetch(previewSrc)
              const blob = await response.blob()
              const file = new File([blob], 'mri.jpg', { type: blob.type })

              const formData = new FormData()
              formData.append('file', file)

              const res = await axios.post(`${import.meta.env.VITE_API_URL || ''}/report`, formData, {
                responseType: 'blob',
                headers: { 'Content-Type': 'multipart/form-data' },
              })

              const url = URL.createObjectURL(new Blob([res.data], { type: 'application/pdf' }))
              const a = document.createElement('a')
              a.href = url
              a.download = 'brain_tumor_report.pdf'
              a.click()
              URL.revokeObjectURL(url)

              toast.success('Report downloaded successfully!', { id: 'pdf-toast' })
            } catch (error) {
              console.error(error)
              toast.error('Failed to generate PDF report', { id: 'pdf-toast' })
            }
          }}
          className="flex-1 btn-quantum"
        >
          ↓ Download PDF Report
        </button>
      </div>
    </div>
  )
}
