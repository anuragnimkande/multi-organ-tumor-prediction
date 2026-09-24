import { useState } from 'react'

/**
 * HeatmapView
 * Shows the original MRI image and Grad-CAM overlay with a toggle.
 *
 * Props:
 *   originalSrc  string  — data URI or URL of original (preprocessed) MRI
 *   heatmapSrc   string  — base64 data URI of Grad-CAM PNG from backend
 */
export default function HeatmapView({ originalSrc, heatmapSrc }) {
  const [showHeatmap, setShowHeatmap] = useState(true)

  if (!originalSrc && !heatmapSrc) return null

  return (
    <div id="heatmap-view" className="w-full">
      {/* Toggle */}
      <div className="flex items-center justify-between mb-3">
        <span className="text-sm font-medium text-slate-300">
          Attention Heatmap (Grad-CAM)
        </span>

        {heatmapSrc && (
          <div className="flex items-center gap-1 bg-void-800 rounded-lg p-1">
            <button
              id="heatmap-toggle-original"
              onClick={() => setShowHeatmap(false)}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
                !showHeatmap
                  ? 'bg-quantum-500/20 text-quantum-400'
                  : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              Original
            </button>
            <button
              id="heatmap-toggle-gradcam"
              onClick={() => setShowHeatmap(true)}
              className={`px-3 py-1 rounded-md text-xs font-medium transition-all ${
                showHeatmap
                  ? 'bg-quantum-500/20 text-quantum-400'
                  : 'text-slate-500 hover:text-slate-300'
              }`}
            >
              Grad-CAM
            </button>
          </div>
        )}
      </div>

      {/* Image display */}
      <div className="relative rounded-xl overflow-hidden border border-quantum-500/20
                      bg-void-900 aspect-square max-h-64 flex items-center justify-center">
        {showHeatmap && heatmapSrc ? (
          <img
            id="heatmap-image"
            src={heatmapSrc}
            alt="Grad-CAM attention heatmap"
            className="w-full h-full object-contain animate-fade-in"
          />
        ) : (
          <img
            id="original-mri-image"
            src={originalSrc}
            alt="Original MRI scan"
            className="w-full h-full object-contain animate-fade-in
                       filter grayscale"
          />
        )}

        {/* Label badge */}
        <div className="absolute top-2 right-2">
          <span className={`text-xs px-2 py-1 rounded-md font-mono ${
            showHeatmap && heatmapSrc
              ? 'bg-quantum-purple/80 text-white'
              : 'bg-void-800/90 text-slate-400'
          }`}>
            {showHeatmap && heatmapSrc ? 'Grad-CAM' : 'MRI Scan'}
          </span>
        </div>
      </div>

      {heatmapSrc && showHeatmap && (
        <p className="mt-2 text-xs text-slate-600 text-center">
          🔴 Red regions = areas most influential in the prediction
        </p>
      )}
    </div>
  )
}
