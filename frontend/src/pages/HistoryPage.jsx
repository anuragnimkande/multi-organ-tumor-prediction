import React, { useEffect, useState } from 'react';
import { ORGANS } from '../config/organs';

export default function HistoryPage() {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    const stored = localStorage.getItem('quantum_history');
    if (stored) {
      setHistory(JSON.parse(stored));
    }
  }, []);

  return (
    <div className="max-w-6xl mx-auto py-12 px-4 space-y-8">
      <div>
        <h1 className="text-4xl font-bold">Analysis History</h1>
        <p className="text-slate-400 mt-2">Recent inferences processed locally on your device.</p>
      </div>

      {history.length === 0 ? (
        <div className="glass-card p-12 text-center text-slate-400">
          No history found. Complete an analysis first.
        </div>
      ) : (
        <div className="glass-card overflow-hidden">
          <table className="w-full text-left text-sm">
            <thead className="bg-white/5 border-b border-white/10">
              <tr>
                <th className="px-6 py-4 font-semibold text-slate-300">Date</th>
                <th className="px-6 py-4 font-semibold text-slate-300">Organ</th>
                <th className="px-6 py-4 font-semibold text-slate-300">Modality</th>
                <th className="px-6 py-4 font-semibold text-slate-300">Prediction</th>
                <th className="px-6 py-4 font-semibold text-slate-300">Confidence</th>
              </tr>
            </thead>
            <tbody className="divide-y divide-white/5">
              {history.map((item) => {
                const organ = ORGANS[item.organ];
                return (
                  <tr key={item.id} className="hover:bg-white/5 transition-colors">
                    <td className="px-6 py-4 text-slate-400">
                      {new Date(item.timestamp).toLocaleString()}
                    </td>
                    <td className="px-6 py-4 flex items-center gap-2">
                      <span>{organ?.icon}</span>
                      <span className="font-medium text-white">{organ?.name || item.organ}</span>
                    </td>
                    <td className="px-6 py-4 text-slate-400">
                      {item.modality}
                    </td>
                    <td className="px-6 py-4">
                      <span className={`px-2 py-1 rounded text-xs font-semibold ${
                        item.prediction.toLowerCase().includes('tumor') || item.prediction.toLowerCase().includes('melanoma') || item.prediction.toLowerCase().includes('malignant')
                          ? 'bg-red-500/20 text-red-400' 
                          : 'bg-emerald-500/20 text-emerald-400'
                      }`}>
                        {item.prediction}
                      </span>
                    </td>
                    <td className="px-6 py-4 font-mono text-quantum-300">
                      {(item.confidence * 100).toFixed(1)}%
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}
    </div>
  );
}
