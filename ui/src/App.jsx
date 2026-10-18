import { useEffect, useState } from 'react'
import { Routes, Route, Link } from 'react-router-dom'
import UploadPage from './pages/UploadPage.jsx'
import ProgressPage from './pages/ProgressPage.jsx'
import ResultsPage from './pages/ResultsPage.jsx'
import HistoryPage from './pages/HistoryPage.jsx'

const MAROON = '#501214'
const GOLD   = '#C6973F'

function SunIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <circle cx="12" cy="12" r="4"/>
      <path d="M12 2v2M12 20v2M4.93 4.93l1.41 1.41M17.66 17.66l1.41 1.41M2 12h2M20 12h2M4.93 19.07l1.41-1.41M17.66 6.34l1.41-1.41"/>
    </svg>
  )
}

function MoonIcon() {
  return (
    <svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24"
      fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/>
    </svg>
  )
}

export default function App() {
  const [dark, setDark] = useState(
    () => localStorage.getItem('theme') === 'dark'
      || (!localStorage.getItem('theme') && window.matchMedia('(prefers-color-scheme: dark)').matches)
  )

  useEffect(() => {
    document.documentElement.classList.toggle('dark', dark)
    localStorage.setItem('theme', dark ? 'dark' : 'light')
  }, [dark])

  return (
    <div
      style={dark ? { backgroundColor: '#1A0506' } : {}}
      className="min-h-screen bg-gray-50 text-gray-900 dark:text-gray-100 transition-colors duration-200"
    >
      {/* Header — TXST Maroon */}
      <header style={{ backgroundColor: MAROON }} className="px-5 py-2.5 flex items-center justify-between gap-4">
        {/* Left: TXST logo + app title */}
        <Link to="/" className="flex items-center gap-3 shrink-0">
          <img
            src="/txst_logo.png"
            alt="Texas State University"
            style={{ height: 38, width: 'auto', objectFit: 'contain' }}
          />
          <div style={{ borderLeft: '1px solid rgba(198,151,63,0.4)' }} className="pl-3">
            <p style={{ color: GOLD, fontWeight: 700, fontSize: 14, lineHeight: '1.2' }}>Cattle Vision</p>
            <p style={{ color: 'rgba(255,255,255,0.65)', fontSize: 11, lineHeight: '1.2' }}>Behavior Inference</p>
          </div>
        </Link>

        {/* Right: nav + HiPE logo + dark mode */}
        <div className="flex items-center gap-2">
          <Link
            to="/history"
            style={{ color: GOLD }}
            className="text-sm font-medium px-3 py-1.5 rounded-lg
              hover:bg-white/10 transition-colors"
          >
            Job history
          </Link>

          <button
            onClick={() => setDark(d => !d)}
            aria-label={dark ? 'Switch to light mode' : 'Switch to dark mode'}
            style={{ color: 'rgba(255,255,255,0.8)' }}
            className="flex items-center gap-1.5 text-sm px-3 py-1.5
              rounded-lg hover:bg-white/10 transition-colors"
          >
            {dark ? <SunIcon /> : <MoonIcon />}
            <span className="hidden sm:inline">{dark ? 'Light' : 'Dark'}</span>
          </button>

          <div style={{ borderLeft: '1px solid rgba(198,151,63,0.4)' }} className="pl-3 ml-1">
            <img
              src="/hipe_logo.png"
              alt="HiPE Research Group"
              style={{ height: 36, width: 'auto', objectFit: 'contain' }}
            />
          </div>
        </div>
      </header>

      <Routes>
        <Route path="/" element={<UploadPage />} />
        <Route path="/jobs/:jobId/progress" element={<ProgressPage />} />
        <Route path="/jobs/:jobId/results" element={<ResultsPage />} />
        <Route path="/history" element={<HistoryPage />} />
      </Routes>
    </div>
  )
}
