import React from 'react';

export default function AboutPage() {
  return (
    <div className="max-w-4xl mx-auto py-12 px-4 space-y-8">
      <h1 className="text-4xl font-bold">About QuantumScan</h1>
      
      <div className="glass-card p-8 space-y-6 text-slate-300 leading-relaxed">
        <p>
          <strong className="text-white text-lg">QuantumScan</strong> is a multi-organ tumor detection platform that leverages Hybrid Quantum-Classical machine learning to analyze medical imaging data.
        </p>

        <div>
          <h2 className="text-xl font-bold text-quantum-400 mb-3">The Technology</h2>
          <p>
            By combining state-of-the-art Classical Convolutional Neural Networks (CNNs) for robust feature extraction with Variational Quantum Circuits (VQCs) for high-dimensional classification, QuantumScan aims to explore the frontier of medical artificial intelligence. 
            The system is built using PyTorch and PennyLane.
          </p>
        </div>

        <div>
          <h2 className="text-xl font-bold text-quantum-400 mb-3">Multi-Organ Roadmap</h2>
          <p>
            Our architecture is designed to scale across multiple organs and modalities:
          </p>
          <ul className="list-disc pl-5 mt-2 space-y-1">
            <li><strong>Brain (MRI):</strong> Fully implemented hybrid pipeline for binary tumor classification.</li>
            <li><strong>Lung (CT), Breast (Mammography), Liver (CT), Skin (Dermoscopy):</strong> Under active development.</li>
          </ul>
        </div>

        <div className="p-4 rounded-lg bg-amber-500/10 border border-amber-500/20 text-amber-200">
          <strong>Medical Disclaimer:</strong> This software is a research prototype. It is NOT intended for clinical use, diagnosis, or treatment. Always consult a certified medical professional.
        </div>
      </div>
    </div>
  );
}
