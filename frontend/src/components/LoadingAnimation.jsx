/**
 * LoadingAnimation
 * Full-screen scanning animation shown while the model runs inference.
 */
export default function LoadingAnimation() {
  return (
    <div
      id="loading-overlay"
      className="fixed inset-0 z-50 flex flex-col items-center justify-center
                 bg-void-950/95 backdrop-blur-sm animate-fade-in"
    >
      {/* Brain scanner graphic */}
      <div className="relative w-56 h-56 mb-8">
        {/* Outer glow ring */}
        <div className="absolute inset-0 rounded-full border border-quantum-500/20 animate-pulse-quantum" />
        <div className="absolute inset-3 rounded-full border border-quantum-500/30 animate-pulse-quantum"
             style={{ animationDelay: '0.3s' }} />
        <div className="absolute inset-6 rounded-full border border-quantum-500/40 animate-pulse-quantum"
             style={{ animationDelay: '0.6s' }} />

        {/* Scan box */}
        <div className="absolute inset-10 rounded-xl overflow-hidden
                        border border-quantum-500/40 bg-void-900">
          {/* Simulated scan image (gradient placeholder) */}
          <div className="w-full h-full bg-gradient-to-b from-slate-800 via-slate-700 to-slate-800
                          flex items-center justify-center text-5xl opacity-60">
            🧬
          </div>

          {/* Scanner line sweeping up and down */}
          <div className="scanner-line" />
        </div>

        {/* Orbiting qubits */}
        {[0, 1, 2, 3].map(i => (
          <div
            key={i}
            className="absolute top-1/2 left-1/2 w-3 h-3 rounded-full bg-quantum-500
                       shadow-[0_0_8px_#00d4ff]"
            style={{
              animation: `orbit ${2 + i * 0.5}s linear infinite`,
              animationDelay: `${i * 0.4}s`,
              transformOrigin: '-50px 0',
              marginTop: '-6px',
              marginLeft: '-6px',
            }}
          />
        ))}
      </div>

      {/* Text */}
      <h2 className="font-display font-bold text-2xl text-quantum-500 mb-2 animate-pulse-quantum">
        Analyzing Medical Scan...
      </h2>
      <p className="text-slate-400 text-sm text-center max-w-xs px-4">
        Running quantum feature extraction through variational circuit...
      </p>

      {/* Progress dots */}
      <div className="flex gap-2 mt-6">
        {[0, 1, 2].map(i => (
          <div
            key={i}
            className="w-2 h-2 rounded-full bg-quantum-500 animate-bounce"
            style={{ animationDelay: `${i * 0.15}s` }}
          />
        ))}
      </div>

      {/* Stage labels */}
      <div className="mt-8 space-y-1 text-xs text-slate-600 font-mono text-center">
        {[
          '⬡ Preprocessing MRI image...',
          '⬡ Extracting CNN features...',
          '⚛ Running quantum circuit...',
          '⬡ Computing prediction...',
        ].map((label, i) => (
          <div
            key={i}
            className="animate-fade-in"
            style={{ animationDelay: `${i * 0.4}s`, opacity: 0,
                     animationFillMode: 'forwards' }}
          >
            {label}
          </div>
        ))}
      </div>
    </div>
  )
}
