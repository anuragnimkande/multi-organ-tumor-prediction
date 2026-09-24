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
    navigate(`/analysis/${organ.id}`);
  };

  const handleAnalyze = async (selectedFile) => {
    if (!selectedOrgan) {
      toast.error('Please select an organ first');
      return;
    }

    setFile(selectedFile);
    setLoading(true);

    const formData = new FormData();
    formData.append('file', selectedFile);
    formData.append('organ', selectedOrgan.id);

    try {
      // In this hybrid setup, we use /predict directly for brain to ensure 100% backward compat,
      // but we could use /api/analyze for everything. Let's use /api/analyze as designed in the backend.
      const response = await axios.post('/api/analyze', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
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

      // Pass result via state to ResultPage (same pattern as original)
      navigate(`/analysis/${selectedOrgan.id}/result`, { 
        state: { 
          result: response.data,
          previewUrl: URL.createObjectURL(selectedFile)
        } 
      });

    } catch (err) {
      console.error(err);
      toast.error(err.response?.data?.error || 'Analysis failed. Please try again.');
      setFile(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="max-w-4xl mx-auto py-12 px-4 sm:px-6 lg:px-8 space-y-12">
      
      {!organId && (
        <div className="space-y-6">
          <div className="text-center">
            <h2 className="text-3xl font-bold">Select Analysis Type</h2>
            <p className="text-slate-400 mt-2">Choose the organ and modality for your medical scan.</p>
          </div>
          <OrganSelector selectedOrgan={selectedOrgan} onSelect={handleOrganSelect} />
        </div>
      )}

      {selectedOrgan && (
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
                <p className="text-slate-400">Modality: <span style={{ color: selectedOrgan.color }} className="font-semibold">{selectedOrgan.modality}</span></p>
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
              onFileSelect={handleAnalyze} 
              isLoading={loading}
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
