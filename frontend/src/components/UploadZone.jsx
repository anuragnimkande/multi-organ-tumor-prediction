import { useCallback, useState } from 'react'
import { useDropzone } from 'react-dropzone'

const ACCEPT = {
  'image/jpeg': ['.jpg', '.jpeg'],
  'image/png':  ['.png'],
  'image/bmp':  ['.bmp'],
  'image/tiff': ['.tif', '.tiff'],
}

/**
 * UploadZone
 * Drag-and-drop / click-to-select image uploader.
 *
 * Props:
 *   onFile(File) — called when a valid file is selected
 *   disabled     — disable while loading
 *   title        - heading text
 *   subtitle     - subtitle text
 */
export default function UploadZone({ onFile, disabled = false, title = "Upload Scan", subtitle = "Drag & drop or click to select a medical image" }) {
  const [error, setError] = useState('')

  const onDrop = useCallback(
    (accepted, rejected) => {
      setError('')
      if (rejected.length > 0) {
        setError('Invalid file. Please upload a JPEG, PNG, BMP, or TIFF image.')
        return
      }
      if (accepted.length > 0) {
        onFile(accepted[0])
      }
    },
    [onFile]
  )

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept:    ACCEPT,
    maxFiles:  1,
    disabled,
  })

  return (
    <div className="w-full">
      <div
        id="upload-dropzone"
        {...getRootProps()}
        className={`
          relative group rounded-2xl border-2 border-dashed p-10 text-center
          transition-all duration-300 cursor-pointer select-none
          ${isDragActive
            ? 'border-quantum-500 bg-quantum-500/10 shadow-quantum-lg scale-[1.01]'
            : disabled
              ? 'border-slate-700 bg-white/[0.02] cursor-not-allowed opacity-60'
              : 'border-quantum-500/30 bg-white/[0.02] hover:border-quantum-500/60 hover:bg-quantum-500/5 hover:shadow-quantum'
          }
        `}
      >
        <input id="upload-input" {...getInputProps()} />

        {/* Animated orbit icon */}
        <div className="relative inline-flex items-center justify-center mb-5">
          <div className={`w-20 h-20 rounded-full flex items-center justify-center text-4xl
                           bg-gradient-to-br from-void-800 to-void-700 border border-quantum-500/20
                           transition-all duration-300 group-hover:shadow-glow
                           ${isDragActive ? 'animate-pulse-quantum' : ''}`}>
            📂
          </div>
          {/* Orbiting dots */}
          <div className="absolute w-20 h-20 animate-spin-slow">
            {[0, 90, 180, 270].map((deg, i) => (
              <div
                key={i}
                className="absolute w-2 h-2 rounded-full bg-quantum-500/60"
                style={{
                  top:  `${50 - 50 * Math.cos((deg * Math.PI) / 180)}%`,
                  left: `${50 + 50 * Math.sin((deg * Math.PI) / 180)}%`,
                  transform: 'translate(-50%, -50%)',
                }}
              />
            ))}
          </div>
        </div>

        {isDragActive ? (
          <div className="animate-fade-in">
            <p className="text-quantum-500 font-semibold text-lg font-display">
              Release to analyze image →
            </p>
          </div>
        ) : (
          <>
            <p className="text-slate-200 font-semibold text-lg font-display mb-1">
              {title}
            </p>
            <p className="text-slate-500 text-sm mb-4">
              {subtitle}
            </p>
            <div className="flex flex-wrap justify-center gap-2">
              {['JPEG', 'PNG', 'BMP', 'TIFF'].map(fmt => (
                <span key={fmt}
                      className="px-2.5 py-0.5 rounded-md bg-void-800 text-slate-500
                                 border border-slate-700 text-xs font-mono">
                  .{fmt.toLowerCase()}
                </span>
              ))}
            </div>
          </>
        )}
      </div>

      {error && (
        <p id="upload-error" className="mt-3 text-tumor text-sm text-center animate-fade-in">
          ⚠ {error}
        </p>
      )}
    </div>
  )
}
