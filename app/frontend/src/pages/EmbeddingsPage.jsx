import { useState, useCallback } from 'react'
import Plot from 'react-plotly.js'

const API = '/api/embeddings'

const POS_OPTIONS = ['NOUN','VERB','ADJ','ADV','PRON','DET','ADP','CONJ','PART','NUM','X']

export default function EmbeddingsPage() {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', height: '100%', overflow: 'hidden' }}>
      <EmbedViz />
      <Sidebar />
    </div>
  )
}

// ── 3D Visualizer ────────────────────────────────────────────────────────────

function EmbedViz() {
  const [input, setInput] = useState('')
  const [reduction, setReduction] = useState('pca')
  const [dims, setDims] = useState('3')
  const [points, setPoints] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const run = useCallback(async () => {
    const words = input.split(/[\s,]+/).map(w => w.trim()).filter(Boolean)
    if (!words.length) return
    setLoading(true); setError('')
    try {
      const res = await fetch(`${API}/embed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ words, reduction, n_components: parseInt(dims) }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setPoints(data.points)
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }, [input, reduction, dims])

  const is3d = dims === '3'

  const plotData = points.length ? [{
    type: is3d ? 'scatter3d' : 'scatter',
    mode: 'markers+text',
    x: points.map(p => p.x),
    y: points.map(p => p.y),
    ...(is3d ? { z: points.map(p => p.z) } : {}),
    text: points.map(p => p.word),
    textposition: 'top center',
    textfont: { family: 'IBM Plex Mono', size: 11, color: '#e8e8f0' },
    marker: {
      size: is3d ? 6 : 8,
      color: points.map((_, i) => i),
      colorscale: 'Viridis',
      opacity: 0.85,
    },
    hovertemplate: '<b>%{text}</b><extra></extra>',
  }] : []

  const layout = {
    paper_bgcolor: '#0c0c0e',
    plot_bgcolor: '#0c0c0e',
    margin: { l: 0, r: 0, t: 0, b: 0 },
    font: { family: 'IBM Plex Mono', color: '#6b6b80', size: 10 },
    scene: {
      bgcolor: '#0c0c0e',
      xaxis: { gridcolor: '#2a2a32', zerolinecolor: '#2a2a32' },
      yaxis: { gridcolor: '#2a2a32', zerolinecolor: '#2a2a32' },
      zaxis: { gridcolor: '#2a2a32', zerolinecolor: '#2a2a32' },
    },
    xaxis: { gridcolor: '#2a2a32', zerolinecolor: '#2a2a32' },
    yaxis: { gridcolor: '#2a2a32', zerolinecolor: '#2a2a32' },
    showlegend: false,
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', borderRight: '1px solid var(--border)' }}>
      <div style={{ display: 'flex', gap: 8, padding: '12px 16px', borderBottom: '1px solid var(--border)', alignItems: 'flex-end', flexWrap: 'wrap' }}>
        <div style={{ flex: 1, minWidth: 200 }}>
          <label>words (sep/by space)</label>
          <input
            value={input}
            onChange={e => setInput(e.target.value)}
            onKeyDown={e => e.key === 'Enter' && run()}
            placeholder="кот сабака птушка вада агонь..."
            style={{ marginTop: 4 }}
          />
        </div>
        <div>
          <label>method</label>
          <select value={reduction} onChange={e => setReduction(e.target.value)} style={{ marginTop: 4, display: 'block' }}>
            <option value="pca">PCA</option>
            <option value="tsne">t-SNE</option>
            <option value="umap">UMAP</option>
          </select>
        </div>
        <div>
          <label>dims</label>
          <select value={dims} onChange={e => setDims(e.target.value)} style={{ marginTop: 4, display: 'block' }}>
            <option value="3">3D</option>
            <option value="2">2D</option>
          </select>
        </div>
        <button className="primary" onClick={run} disabled={loading}>
          {loading ? 'loading...' : 'visualize'}
        </button>
      </div>

      {error && <div className="error" style={{ padding: '8px 16px' }}>{error}</div>}

      <div style={{ flex: 1, overflow: 'hidden' }}>
        {points.length > 0 ? (
          <Plot
            data={plotData}
            layout={layout}
            config={{ displayModeBar: false, responsive: true }}
            style={{ width: '100%', height: '100%' }}
            useResizeHandler
          />
        ) : (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--muted)', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
            enter words and press "visualize"
          </div>
        )}
      </div>
    </div>
  )
}

// ── Sidebar ──────────────────────────────────────────────────────────────────

function Sidebar() {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 1, overflowY: 'auto', background: 'var(--bg)' }}>
      <SimilarityPanel />
      <NearestPanel />
      <AnalogyPanel />
    </div>
  )
}

function SimilarityPanel() {
  const [a, setA] = useState('')
  const [b, setB] = useState('')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)

  const run = async () => {
    if (!a || !b) return
    setLoading(true)
    try {
      const res = await fetch(`${API}/similarity`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ word_a: a, word_b: b }),
      })
      setResult(await res.json())
    } finally { setLoading(false) }
  }

  return (
    <div className="card" style={{ borderRadius: 0, borderLeft: 'none', borderRight: 'none', borderTop: 'none' }}>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.08em' }}>similarity</div>
      <div style={{ display: 'flex', gap: 6 }}>
        <input value={a} onChange={e => setA(e.target.value)} placeholder="word A" style={{ flex: 1 }} />
        <input value={b} onChange={e => setB(e.target.value)} placeholder="word B" style={{ flex: 1 }} />
        <button onClick={run} disabled={loading} style={{ whiteSpace: 'nowrap' }}>→</button>
      </div>
      {result && (
        <div style={{ marginTop: 10, display: 'flex', gap: 12 }}>
          <Metric label="cosine" value={result.cosine} color={result.cosine > 0.5 ? 'var(--success)' : result.cosine > 0 ? 'var(--accent2)' : 'var(--muted)'} />
          <Metric label="euclidean" value={result.euclidean} color="var(--muted)" />
        </div>
      )}
    </div>
  )
}

function NearestPanel() {
  const [word, setWord] = useState('')
  const [topn, setTopn] = useState(8)
  const [neighbours, setNeighbours] = useState([])
  const [loading, setLoading] = useState(false)

  const run = async () => {
    if (!word) return
    setLoading(true)
    try {
      const res = await fetch(`${API}/nearest`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ word, topn }),
      })
      const data = await res.json()
      setNeighbours(data.neighbours)
    } finally { setLoading(false) }
  }

  return (
    <div className="card" style={{ borderRadius: 0, borderLeft: 'none', borderRight: 'none' }}>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.08em' }}>nearest neighbours</div>
      <div style={{ display: 'flex', gap: 6 }}>
        <input value={word} onChange={e => setWord(e.target.value)} onKeyDown={e => e.key === 'Enter' && run()} placeholder="word" style={{ flex: 1 }} />
        <select value={topn} onChange={e => setTopn(Number(e.target.value))}>
          {[5,8,10,15].map(n => <option key={n} value={n}>top {n}</option>)}
        </select>
        <button onClick={run} disabled={loading}>→</button>
      </div>
      {neighbours.length > 0 && (
        <div style={{ marginTop: 10, display: 'flex', flexWrap: 'wrap', gap: 6 }}>
          {neighbours.map((w, i) => (
            <span key={w} style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: `hsl(${260 + i * 10}, 70%, 70%)`, cursor: 'pointer' }}
              onClick={() => setWord(w)}>
              {w}
            </span>
          ))}
        </div>
      )}
    </div>
  )
}

function AnalogyPanel() {
  const [a, setA] = useState('')
  const [b, setB] = useState('')
  const [c, setC] = useState('')
  const [results, setResults] = useState([])
  const [loading, setLoading] = useState(false)

  const run = async () => {
    if (!a || !b || !c) return
    setLoading(true)
    try {
      const res = await fetch(`${API}/analogy`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ a, b, c }),
      })
      const data = await res.json()
      setResults(data.results)
    } finally { setLoading(false) }
  }

  return (
    <div className="card" style={{ borderRadius: 0, borderLeft: 'none', borderRight: 'none', borderBottom: 'none' }}>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)', marginBottom: 10, textTransform: 'uppercase', letterSpacing: '0.08em' }}>analogy  a : b = c : ?</div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr auto', gap: 6, alignItems: 'end' }}>
        <div><label>A</label><input value={a} onChange={e => setA(e.target.value)} placeholder="кароль" style={{ marginTop: 4 }} /></div>
        <div><label>B</label><input value={b} onChange={e => setB(e.target.value)} placeholder="каралева" style={{ marginTop: 4 }} /></div>
        <div><label>C</label><input value={c} onChange={e => setC(e.target.value)} placeholder="цар" style={{ marginTop: 4 }} /></div>
        <button onClick={run} disabled={loading}>→</button>
      </div>
      {results.length > 0 && (
        <div style={{ marginTop: 12 }}>
          <div style={{ fontSize: 11, color: 'var(--muted)', fontFamily: 'var(--font-mono)', marginBottom: 6 }}>вынік:</div>
          {results.map((w, i) => (
            <div key={w} style={{ fontFamily: 'var(--font-mono)', fontSize: 13, color: i === 0 ? 'var(--accent2)' : 'var(--text)', padding: '3px 0' }}>
              {i === 0 ? '→ ' : '   '}{w}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}

function Metric({ label, value, color }) {
  return (
    <div>
      <div style={{ fontSize: 10, color: 'var(--muted)', fontFamily: 'var(--font-mono)', textTransform: 'uppercase' }}>{label}</div>
      <div style={{ fontSize: 20, fontFamily: 'var(--font-mono)', color, fontWeight: 500 }}>{value}</div>
    </div>
  )
}