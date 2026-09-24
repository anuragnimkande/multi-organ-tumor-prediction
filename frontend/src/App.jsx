import { Routes, Route, useLocation, Navigate } from 'react-router-dom'
import { useEffect } from 'react'
import Navbar    from './components/Navbar'
import HomePage  from './pages/HomePage'
import UploadPage from './pages/UploadPage'
import ResultPage from './pages/ResultPage'

import DashboardPage from './pages/DashboardPage'
import AnalysisPage from './pages/AnalysisPage'
import AnalysisResultPage from './pages/AnalysisResultPage'
import HistoryPage from './pages/HistoryPage'
import ModelsPage from './pages/ModelsPage'
import AboutPage from './pages/AboutPage'

function App() {
  const location = useLocation()

  // Reset scroll on route change
  useEffect(() => { window.scrollTo(0, 0) }, [location.pathname])

  return (
    <div className="min-h-screen bg-void-950 flex flex-col">
      <Navbar />
      <main className="flex-1">
        <Routes>
          {/* New Multi-Organ Routes */}
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/analysis" element={<AnalysisPage />} />
          <Route path="/analysis/:organId" element={<AnalysisPage />} />
          <Route path="/analysis/:organId/result" element={<AnalysisResultPage />} />
          <Route path="/history" element={<HistoryPage />} />
          <Route path="/models" element={<ModelsPage />} />
          <Route path="/about" element={<AboutPage />} />

          {/* Legacy Routes - kept for backward compatibility or redirected */}
          <Route path="/"        element={<Navigate to="/dashboard" replace />} />
          <Route path="/upload"  element={<Navigate to="/analysis/brain" replace />} />
          <Route path="/result"  element={<ResultPage />} />
        </Routes>
      </main>

      {/* Footer */}
      <footer className="border-t border-quantum-500/10 py-6 text-center text-xs text-slate-600">
        <span className="font-display">
          ⚛️ QuantumScan &nbsp;·&nbsp; Hybrid Quantum-Classical Multi-Organ AI &nbsp;·&nbsp;
          Powered by PyTorch & PennyLane
        </span>
      </footer>
    </div>
  )
}

export default App
