import React from 'react';
import { ORGANS } from '../config/organs';
import StatusBadge from '../components/StatusBadge';

export default function ModelsPage() {
  const organsList = Object.values(ORGANS);

  return (
    <div className="max-w-5xl mx-auto py-12 px-4 space-y-12">
      <div className="text-center space-y-4">
        <h1 className="text-4xl font-bold">Model Architectures</h1>
        <p className="text-xl text-slate-400">
          Hybrid pipelines combining CNN feature extractors with Variational Quantum Circuits (VQCs).
        </p>
      </div>

      <div className="space-y-8">
        {organsList.map(organ => (
          <div key={organ.id} className="glass-card p-6 border-l-4" style={{ borderLeftColor: organ.color }}>
            <div className="flex justify-between items-start mb-6">
              <div className="flex items-center gap-3">
                <span className="text-3xl">{organ.icon}</span>
                <div>
                  <h2 className="text-2xl font-bold">{organ.name}</h2>
                  <p className="text-slate-400">{organ.modality}</p>
                </div>
              </div>
              <StatusBadge status={organ.status} />
            </div>

            {organ.status === 'active' ? (
              <div className="space-y-6">
                <div className="p-4 bg-white/5 rounded-lg font-mono text-sm overflow-x-auto text-quantum-300">
                  <pre>{`Input (${organ.modality}) → ResNet18 (128 feats) → Dense(4) → 4-qubit VQC → Output`}</pre>
                </div>
                <div>
                  <h3 className="font-semibold text-lg text-white mb-2">Quantum Circuit Details</h3>
                  <ul className="list-disc pl-5 text-slate-400 space-y-1">
                    <li>Device: default.qubit (PennyLane CPU)</li>
                    <li>Encoding: AngleEmbedding (RY)</li>
                    <li>Layers: 3 StronglyEntanglingLayers</li>
                    <li>Trainable Params: 36</li>
                  </ul>
                </div>
              </div>
            ) : (
              <div className="text-slate-500 italic">
                Model architecture currently under development and clinical validation.
              </div>
            )}
          </div>
        ))}
      </div>
    </div>
  );
}
