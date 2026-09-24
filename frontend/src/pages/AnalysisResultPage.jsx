import React from 'react';
import { useLocation, useNavigate, useParams, Link } from 'react-router-dom';
import ResultCard from '../components/ResultCard';
import { ORGANS } from '../config/organs';

export default function AnalysisResultPage() {
  const { organId } = useParams();
  const location = useLocation();
  const navigate = useNavigate();
  
  const organ = ORGANS[organId] || ORGANS['brain'];
  
  // We passed the result via router state in AnalysisPage
  const { result, previewUrl } = location.state || {};

  if (!result) {
    return (
      <div className="min-h-screen flex items-center justify-center">
        <div className="text-slate-500 text-sm">No analysis result found. Please upload a scan first.</div>
        <button onClick={() => navigate('/analysis')} className="ml-4 btn-outline text-xs px-3 py-1">Go back</button>
      </div>
    );
  }

  const handleReset = () => {
    navigate(`/analysis/${organId}`);
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-start pt-10 pb-20 px-4">
      <div className="w-full max-w-2xl space-y-6">

        {/* Breadcrumb */}
        <div className="flex items-center gap-2 text-sm text-slate-500">
          <Link to="/dashboard" className="hover:text-slate-300 transition-colors">Dashboard</Link>
          <span>›</span>
          <Link to={`/analysis/${organId}`} className="hover:text-slate-300 transition-colors">{organ.name}</Link>
          <span>›</span>
          <span className="text-slate-300">Result</span>
        </div>

        {/* Page title */}
        <div className="flex items-center gap-3">
          <div 
            className="text-2xl p-2 rounded-lg bg-white/5 border border-white/10"
            style={{ boxShadow: `inset 0 0 10px ${organ.color}30` }}
          >
            {organ.icon}
          </div>
          <div>
            <h1 className="font-display font-bold text-3xl text-white">
              Analysis Complete
            </h1>
            <p className="text-slate-400 text-sm mt-1">
              Hybrid Quantum-Classical {organ.modality} inference result
            </p>
          </div>
        </div>

        {/* Main result card (reused component, but we pass organ context) */}
        <ResultCard 
          result={result} 
          previewSrc={previewUrl} 
          onReset={handleReset}
          organColor={organ.color}
        />

      </div>
    </div>
  );
}
