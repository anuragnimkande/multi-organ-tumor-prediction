/**
 * ConfidenceMeter
 * Animated horizontal progress bar showing model confidence percentage.
 *
 * Props:
 *   confidence  number  0–1
 *   isTumor     boolean
 */
export default function ConfidenceMeter({ confidence = 0, isTumor = false }) {
  const pct = Math.round(confidence * 100)

  return (
    <div id="confidence-meter" className="w-full">
      <div className="flex justify-between items-baseline mb-2">
        <span className="text-sm text-slate-400 font-medium">Confidence</span>
        <span className={`text-2xl font-bold font-display ${
          isTumor ? 'text-tumor' : 'text-healthy'
        }`}>
          {pct}%
        </span>
      </div>

      {/* Track */}
      <div className="relative h-3 rounded-full bg-void-800 border border-white/5 overflow-hidden">
        {/* Fill */}
        <div
          className="confidence-bar-fill"
          style={{
            width: `${pct}%`,
            background: isTumor
              ? 'linear-gradient(90deg, #7c3aed, #ef4444)'
              : 'linear-gradient(90deg, #7c3aed, #10b981)',
          }}
        />

        {/* Shimmer effect */}
        <div
          className="absolute inset-0 opacity-30"
          style={{
            background: 'linear-gradient(90deg, transparent 0%, rgba(255,255,255,0.2) 50%, transparent 100%)',
            animation: 'scan 2s ease-in-out infinite',
          }}
        />
      </div>

      {/* Scale labels */}
      <div className="flex justify-between mt-1 text-xs text-slate-600">
        <span>0%</span>
        <span>50%</span>
        <span>100%</span>
      </div>
    </div>
  )
}
