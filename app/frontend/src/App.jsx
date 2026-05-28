import { BrowserRouter, Routes, Route, NavLink } from 'react-router-dom'
import EmbeddingsPage from './pages/EmbeddingsPage'
import MorphologyPage from './pages/MorphologyPage'

export default function App() {
  return (
    <BrowserRouter>
      <div style={{ display: 'flex', flexDirection: 'column', height: '100vh' }}>
        <Nav />
        <main style={{ flex: 1, overflow: 'hidden' }}>
          <Routes>
            <Route path="/" element={<EmbeddingsPage />} />
            <Route path="/morphology" element={<MorphologyPage />} />
          </Routes>
        </main>
      </div>
    </BrowserRouter>
  )
}

function Nav() {
  const linkStyle = ({ isActive }) => ({
    fontFamily: 'var(--font-mono)',
    fontSize: '12px',
    letterSpacing: '0.08em',
    textTransform: 'uppercase',
    padding: '4px 0',
    borderBottom: isActive ? '2px solid var(--accent)' : '2px solid transparent',
    color: isActive ? 'var(--text)' : 'var(--muted)',
    transition: 'color 0.15s, border-color 0.15s',
  })

  return (
    <nav style={{
      display: 'flex', alignItems: 'center', gap: 32,
      padding: '0 24px', height: 48,
      borderBottom: '1px solid var(--border)',
      background: 'var(--bg2)',
      flexShrink: 0,
    }}>
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: 'var(--accent)', marginRight: 16 }}>
        bel<span style={{ color: 'var(--accent2)' }}>NLP</span>
      </span>
      <NavLink to="/" style={linkStyle}>embeddings</NavLink>
      <NavLink to="/morphology" style={linkStyle}>morphology</NavLink>
    </nav>
  )
}