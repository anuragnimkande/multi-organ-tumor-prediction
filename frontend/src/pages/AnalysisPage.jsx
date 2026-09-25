import React, { useState, useEffect } from 'react';
import { useParams, useNavigate } from 'react-router-dom';
import axios from 'axios';
import { toast } from 'react-hot-toast';
import { ORGANS } from '../config/organs';
import UploadZone from '../components/UploadZone';
import OrganSelector from '../components/OrganSelector';

const AnalysisPage = () => {
  const { organId } = useParams();
  const navigate = useNavigate();
  const [selectedOrgan, setSelectedOrgan] = useState(null);
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    if (organId && ORGANS[organId]) {
      setSelectedOrgan(ORGANS[organId]);
    } else if (organId) {
      toast.error('Invalid organ selected');
      navigate('/analysis');
    }
  }, [organId, navigate]);

  const handleOrganSelect = (organ) => {
    // coming_soon organs are visually disabled in OrganCard (cursor-not-allowed).
    // Guard here too in case state is manipulated directly.
    if (organ.status !== 'active') {
      toast.error(`${organ.name} analysis is not yet available.`);
      return;
    }
    navigate(`/analysis/${organ.id}`);
  };

  const handleAnalyze = async (selectedFile) => {
    if (!selectedOrgan) {
      toast.error('Please select an organ first');
      return;
    }
    if (selectedOrgan.status !== 'active') {
      toast.error(`${selectedOrgan.name} analysis is not yet available.`);
      return;
    }

    setFile(selectedFile);
    setLoading(true);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('organ', selectedOrgan.id);

    try {
      const response = await axios.post('/api/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
        timeout: 120_000,
      });

      // Save to localStorage history
      const historyItem = {
        id: Date.now(),
        organ: selectedOrgan.id,
        modality: selectedOrgan.modality,
        prediction: response.data.prediction,
        confidence: response.data.confidence,
        timestamp: new Date().toISOString(),
      };

      const existingHistory = JSON.parse(localStorage.getItem('quantum_history') || '[]');
      localStorage.setItem('quantum_history', JSON.stringify([historyItem, ...existingHistory].slice(0, 50)));

      navigate(`/analysis/${selectedOrgan.id}/result`, {
        state: {
          result: response.data,
          previewUrl: URL.createObjectURL(selectedFile),
        },
      });
    } catch (err) {
      console.error(err);
      const msg = err.response?.data?.error || 'Analysis failed. Please try again.';
      toast.error(msg);
      setFile(null);
    } finally {
      setLoading(false);
    }
  };

  // ── Coming Soon panel ─────────────────────────────────────────────────────
  const ComingSoonPanel = ({ organ }) => (
    <div className="glass-card p-10 text-center space-y-5 animate-slide-up">
      <div
        className="mx-auto w-20 h-20 rounded-2xl bg-white/5 border border-white/10
                   flex items-center justify-center text-5xl"
        style={{ boxShadow: `inset 0 0 20px ${organ.color}30` }}
      >
        {organ.icon}
      </div>

      <div>
        <h2 className="text-2xl font-bold text-white mb-2">{organ.name}</h2>
        <p className="text-sm font-medium" style={{ color: organ.color }}>{organ.modality}</p>
      </div>

      <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full
                      bg-slate-500/10 border border-slate-500/30 text-slate-400 text-sm font-medium">
        <span className="w-2 h-2 rounded-full bg-slate-400" />
        Coming Soon
      </div>

      <p className="text-slate-400 text-sm max-w-md mx-auto leading-relaxed">
        {organ.description}
      </p>

      {organ.classes && (
        <div className="text-left max-w-sm mx-auto">
          <p className="text-xs text-slate-500 uppercase tracking-wider mb-2">Planned output classes</p>
          <div className="flex flex-wrap gap-2">
            {organ.classes.map((cls) => (
              <span
                key={cls}
                className="px-2 py-0.5 rounded-full text-xs border border-white/10 text-slate-400 bg-white/5"
              >
                {cls}
              </span>
            ))}
          </div>
        </div>
      )}

      <p className="text-xs text-slate-600">
        A trained model checkpoint is required to enable prediction for this organ.
        See <code className="font-mono">checkpoints/</code> directory.
      </p>

      <button
        onClick={() => navigate('/analysis')}
        className="btn-outline text-sm py-2 px-6 mx-auto"
      >
        ← Choose a Different Organ
      </button>
    </div>
  );

  return (
    <div className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8 space-y-12">

      {!organId && (
        <div className="space-y-6">
          <div className="text-center">
            <h2 className="text-3xl font-bold">Select Analysis Type</h2>
            <p className="text-slate-400 mt-2">
              Choose the organ and modality for your medical scan.{' '}
              <span className="text-emerald-400 font-medium">Active</span> organs have
              trained models available. Others are Coming Soon.
            </p>
          </div>
          <OrganSelector selectedOrgan={selectedOrgan} onSelect={handleOrganSelect} />
        </div>
      )}

      {selectedOrgan && selectedOrgan.status !== 'active' && (
        <ComingSoonPanel organ={selectedOrgan} />
      )}

      {selectedOrgan && selectedOrgan.status === 'active' && (
        <div className="space-y-8 animate-in fade-in slide-in-from-bottom-4 duration-500">

          <div className="flex items-center justify-between">
            <div className="flex items-center gap-4">
              <div
                className="text-4xl p-4 rounded-xl bg-white/5 border border-white/10"
                style={{ boxShadow: `inset 0 0 15px ${selectedOrgan.color}30` }}
              >
                {selectedOrgan.icon}
              </div>
              <div>
                <h1 className="text-3xl font-bold text-white">{selectedOrgan.name} Analysis</h1>
                <p className="text-slate-400">
                  Modality:{' '}
                  <span style={{ color: selectedOrgan.color }} className="font-semibold">
                    {selectedOrgan.modality}
                  </span>
                </p>
              </div>
            </div>

            <button
              onClick={() => navigate('/analysis')}
              className="text-sm text-slate-400 hover:text-white transition-colors"
            >
              Change Organ
            </button>
          </div>

          <div className="glass-card p-1">
            <UploadZone
              onFile={handleAnalyze}
              disabled={loading}
              accept={{ 'image/*': ['.jpeg', '.jpg', '.png'] }}
              maxSize={10 * 1024 * 1024}
              title={`Upload ${selectedOrgan.modality} Scan`}
              subtitle={`Drag & drop or click to select a ${selectedOrgan.modality} image`}
            />
          </div>

        </div>
      )}

    </div>
  );
};

export default AnalysisPage;
