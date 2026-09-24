import React from 'react';

const MedicalDisclaimer = ({ type = 'standard' }) => {
  return (
    <div className="mt-8 p-4 glass-card border-amber-500/30 bg-amber-500/5">
      <div className="flex gap-3">
        <div className="text-amber-400 text-xl">⚠️</div>
        <div className="text-sm text-slate-300 space-y-1">
          <p className="font-semibold text-amber-400/90">AI-generated assessment — Not a clinical diagnosis</p>
          <p>
            {type === 'low_confidence' 
              ? "The model has flagged this scan with low confidence. This may be due to image artifacts, out-of-distribution presentation, or complex pathology."
              : "This system is a research prototype utilizing experimental quantum-classical algorithms."}
            {' '}Always consult a qualified healthcare professional for medical advice and diagnosis.
          </p>
        </div>
      </div>
    </div>
  );
};

export default MedicalDisclaimer;
