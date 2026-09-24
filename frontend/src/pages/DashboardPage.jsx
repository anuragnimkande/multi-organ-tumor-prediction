import React from 'react';
import { useNavigate } from 'react-router-dom';
import OrganSelector from '../components/OrganSelector';
import { ORGANS } from '../config/organs';

const DashboardPage = () => {
  const navigate = useNavigate();

  const handleOrganSelect = (organ) => {
    navigate(`/analysis/${organ.id}`);
  };

  return (
    <div className="max-w-7xl mx-auto py-12 px-4 sm:px-6 lg:px-8 space-y-16">
      
      {/* Hero Section */}
      <div className="text-center space-y-6">
        <h1 className="text-5xl font-extrabold tracking-tight">
          Hybrid Quantum-Classical <br/>
          <span className="text-transparent bg-clip-text bg-gradient-to-r from-quantum-400 to-quantum-200">
            Multi-Organ Tumor Detection
          </span>
        </h1>
        <p className="max-w-2xl mx-auto text-xl text-slate-400">
          Select an organ below to begin analysis. Our system leverages advanced classical CNNs combined with PennyLane variational quantum circuits for enhanced pattern recognition.
        </p>
      </div>

      {/* Organ Selector */}
      <div>
        <h2 className="text-2xl font-bold mb-6 flex items-center gap-2">
          <span className="w-1.5 h-6 bg-quantum-500 rounded-full"></span>
          Supported Organs
        </h2>
        <OrganSelector onSelect={handleOrganSelect} />
      </div>

      {/* System Stats (Mock) */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <div className="metric-card">
          <span className="text-sm font-medium text-slate-400">Total Analyses</span>
          <span className="text-3xl font-bold text-white">1,204</span>
        </div>
        <div className="metric-card">
          <span className="text-sm font-medium text-slate-400">Quantum Inference Avg Time</span>
          <span className="text-3xl font-bold text-quantum-400">0.82s</span>
        </div>
        <div className="metric-card">
          <span className="text-sm font-medium text-slate-400">Active Models</span>
          <span className="text-3xl font-bold text-emerald-400">1 (Brain MRI)</span>
        </div>
      </div>

    </div>
  );
};

export default DashboardPage;
