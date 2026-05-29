import { useState, useEffect, useCallback, useMemo } from 'react'
import Plot from 'react-plotly.js'

const API = '/api/embeddings'
const SLICES = [1, 5, 10, 25, 50, 100]
const CLUSTER_K_OPTIONS = [3, 5, 8, 10]
const CLUSTER_COLORS = [
  '#ff2d55', '#0af',    '#f5a623', '#4cd964', '#bf5af2',
  '#ff9f0a', '#00e5ff', '#ff375f', '#30d158', '#da8fff',
]

// ── k-means++ ────────────────────────────────────────────────────────────────
function kmeans(pts, k, iters = 50) {
  const vec  = p => [p.x, p.y, p.z ?? 0]
  const dist2 = (a, b) => a.reduce((s, v, i) => s + (v - b[i]) ** 2, 0)
  if (pts.length <= k) return pts.map((_, i) => i % k)

  // k-means++ initialisation
  const centroids = [vec(pts[Math.floor(Math.random() * pts.length)])]
  while (centroids.length < k) {
    const dists = pts.map(p => Math.min(...centroids.map(c => dist2(vec(p), c))))
    const total = dists.reduce((s, d) => s + d, 0)
    let r = Math.random() * total
    for (let i = 0; i < pts.length; i++) {
      r -= dists[i]
      if (r <= 0) { centroids.push(vec(pts[i])); break }
    }
    if (centroids.length < k) centroids.push(vec(pts[pts.length - 1]))
  }

  let labels = new Array(pts.length).fill(0)
  for (let iter = 0; iter < iters; iter++) {
    for (let i = 0; i < pts.length; i++) {
      let best = 0, bestD = Infinity
      for (let j = 0; j < k; j++) {
        const d = dist2(vec(pts[i]), centroids[j])
        if (d < bestD) { bestD = d; best = j }
      }
      labels[i] = best
    }
    const sums   = Array.from({ length: k }, () => [0, 0, 0])
    const counts = new Array(k).fill(0)
    pts.forEach((p, i) => { vec(p).forEach((v, d) => { sums[labels[i]][d] += v }); counts[labels[i]]++ })
    centroids.forEach((c, j) => { if (counts[j]) c.forEach((_, d) => { c[d] = sums[j][d] / counts[j] }) })
  }
  return labels
}

// ── Page ─────────────────────────────────────────────────────────────────────
export default function WordsPage() {
  const [allWords,  setAllWords]  = useState([])
  const [allPoints, setAllPoints] = useState([])
  const [reduction, setReduction] = useState('pca')
  const [dims,      setDims]      = useState('3')
  const [slicePct,  setSlicePct]  = useState(100)
  const [search,      setSearch]      = useState('')   // raw input — updates instantly
  const [searchLower, setSearchLower] = useState('')   // debounced — drives the plot
  const [showLabels, setShowLabels] = useState(true)
  const [clusterOn, setClusterOn] = useState(false)
  const [clusterK,  setClusterK]  = useState(5)
  const [status,    setStatus]    = useState('idle')  // idle | loading | done | error
  const [error,     setError]     = useState('')

  // ── load words.txt ─────────────────────────────────────────────────────────
  useEffect(() => {
    ;(async () => {
      setStatus('loading')
      try {
        const res = await fetch('/words.txt')
        if (!res.ok) throw new Error(`words.txt not found (${res.status})`)
        const words = (await res.text())
          .split('\n').map(l => l.trim()).filter(l => l && !l.startsWith('#'))
        setAllWords(words)
      } catch (e) { setError(e.message); setStatus('error') }
    })()
  }, [])

  // ── embed ──────────────────────────────────────────────────────────────────
  const embed = useCallback(async (words, red, n) => {
    if (!words.length) return
    setStatus('loading'); setError('')
    try {
      const res = await fetch(`${API}/embed`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ words, reduction: red, n_components: parseInt(n) }),
      })
      if (!res.ok) throw new Error(await res.text())
      setAllPoints((await res.json()).points)
      setStatus('done')
    } catch (e) { setError(e.message); setStatus('error') }
  }, [])

  useEffect(() => { if (allWords.length) embed(allWords, reduction, dims) }, [allWords])

  // debounce search → plot update (300 ms)
  useEffect(() => {
    const t = setTimeout(() => setSearchLower(search.trim().toLowerCase()), 300)
    return () => clearTimeout(t)
  }, [search])

  // ── derived state ──────────────────────────────────────────────────────────
  const visiblePoints = useMemo(() => {
    const n = Math.max(1, Math.ceil(allPoints.length * slicePct / 100))
    return allPoints.slice(0, n)
  }, [allPoints, slicePct])

  const clusterLabels = useMemo(() =>
    clusterOn && visiblePoints.length >= clusterK
      ? kmeans(visiblePoints, clusterK)
      : null,
    [clusterOn, clusterK, visiblePoints]
  )

  const matchCount  = searchLower
    ? visiblePoints.filter(p => p.word.includes(searchLower)).length
    : 0

  // ── plotly ─────────────────────────────────────────────────────────────────
  const is3d     = dims === '3'
  const loading  = status === 'loading'
  const tooMany  = visiblePoints.length > 150  // hide labels when crowded

  const plotData = useMemo(() => {
    if (!visiblePoints.length) return []

    const base = { type: is3d ? 'scatter3d' : 'scatter', textposition: 'top center', hovertemplate: '<b>%{text}</b><extra></extra>' }
    const xyz  = pts => ({
      x: pts.map(p => p.x), y: pts.map(p => p.y),
      ...(is3d ? { z: pts.map(p => p.z) } : {}),
    })

    // ── search mode: two traces, labels only on matches ────────────────────
    if (searchLower) {
      const hit  = visiblePoints.filter(p =>  p.word.includes(searchLower))
      const miss = visiblePoints.filter(p => !p.word.includes(searchLower))
      return [
        // dimmed background points — no labels
        ...(miss.length ? [{
          ...base, ...xyz(miss),
          mode: 'markers',
          text: miss.map(p => p.word),
          marker: { size: is3d ? 3 : 4, color: '#1e1e2e', opacity: 0.5, line: { width: 0 } },
        }] : []),
        // highlighted matches — labels visible only if showLabels
        ...(hit.length ? [{
          ...base, ...xyz(hit),
          mode: showLabels ? 'markers+text' : 'markers',
          text: hit.map(p => p.word),
          textfont: { family: 'IBM Plex Mono', size: 11, color: '#ffffff' },
          marker: { size: is3d ? 8 : 10, color: '#ffffff', opacity: 1, line: { width: 0 } },
        }] : []),
      ]
    }

    // ── normal mode ────────────────────────────────────────────────────────
    const markerColors = visiblePoints.map((_, i) =>
      clusterLabels
        ? CLUSTER_COLORS[clusterLabels[i] % CLUSTER_COLORS.length]
        : `hsl(${(i / visiblePoints.length) * 260 + 220}, 70%, 65%)`
    )
    const textColors = clusterLabels ? markerColors : visiblePoints.map(() => '#c0c0d8')

    return [{
      ...base, ...xyz(visiblePoints),
      mode: (showLabels && !tooMany) ? 'markers+text' : 'markers',
      text: visiblePoints.map(p => p.word),
      textfont: { family: 'IBM Plex Mono', size: 9, color: textColors },
      marker: { size: is3d ? 4 : 6, color: markerColors, opacity: 0.9, line: { width: 0 } },
    }]
  }, [visiblePoints, clusterLabels, searchLower, is3d, tooMany, showLabels])

  const layout = useMemo(() => ({
    paper_bgcolor: '#0c0c0e',
    plot_bgcolor:  '#0c0c0e',
    margin: { l: 0, r: 0, t: 0, b: 0 },
    font:   { family: 'IBM Plex Mono', color: '#6b6b80', size: 10 },
    scene: {
      bgcolor: '#0c0c0e',
      camera:  { eye: { x: 1.4, y: 1.4, z: 1.1 }, center: { x: 0, y: 0, z: -0.15 } },
      xaxis: { gridcolor: '#2a2a32', zerolinecolor: '#2a2a32' },
      yaxis: { gridcolor: '#2a2a32', zerolinecolor: '#2a2a32' },
      zaxis: { gridcolor: '#2a2a32', zerolinecolor: '#2a2a32' },
    },
    xaxis: { gridcolor: '#2a2a32', zerolinecolor: '#2a2a32' },
    yaxis: { gridcolor: '#2a2a32', zerolinecolor: '#2a2a32' },
    showlegend: false,
  }), [])

  // ── render ─────────────────────────────────────────────────────────────────
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 280px', height: '100%', overflow: 'hidden' }}>

      {/* ── plot column ── */}
      <div style={{ display: 'flex', flexDirection: 'column', height: '100%', borderRight: '1px solid var(--border)' }}>

        {/* toolbar */}
        <div style={{
          display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap',
          padding: '10px 16px', borderBottom: '1px solid var(--border)',
          background: 'var(--bg2)', flexShrink: 0,
        }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)' }}>
            {loading
              ? 'embedding…'
              : status === 'done'
                ? <><span style={{ color: 'var(--text)' }}>{visiblePoints.length}</span>{' / '}{allPoints.length} words{tooMany && showLabels && <span style={{ marginLeft: 8, color: 'var(--border)' }}>(too many for labels)</span>}</>
                : '—'}
          </span>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 8, alignItems: 'center' }}>
            <label style={{
              display: 'flex', alignItems: 'center', gap: 5,
              fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)',
              cursor: 'pointer', userSelect: 'none',
            }}>
              <input type="checkbox" checked={showLabels} onChange={e => setShowLabels(e.target.checked)} />
              labels
            </label>
            <select value={reduction} onChange={e => setReduction(e.target.value)} disabled={loading}>
              <option value="pca">PCA</option>
              <option value="tsne">t-SNE</option>
              <option value="umap">UMAP</option>
            </select>
            <select value={dims} onChange={e => setDims(e.target.value)} disabled={loading}>
              <option value="3">3D</option>
              <option value="2">2D</option>
            </select>
            <button className="primary" onClick={() => embed(allWords, reduction, dims)} disabled={loading || !allWords.length}>
              {loading ? '…' : 'replot'}
            </button>
          </div>
        </div>

        {error && <div className="error" style={{ padding: '8px 16px', flexShrink: 0 }}>{error}</div>}

        <div style={{ flex: 1, overflow: 'hidden' }}>
          {visiblePoints.length > 0
            ? <Plot data={plotData} layout={layout} config={{ displayModeBar: false, responsive: true }} style={{ width: '100%', height: '100%' }} useResizeHandler />
            : !loading && (
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%', color: 'var(--muted)', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
                add words to public/words.txt and press replot
              </div>
            )}
        </div>
      </div>

      {/* ── right sidebar ── */}
      <div style={{ display: 'flex', flexDirection: 'column', overflowY: 'auto', background: 'var(--bg)' }}>

        {/* data slice */}
        <Panel label="data slice">
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)', marginBottom: 8 }}>
            {Math.max(1, Math.ceil(allPoints.length * slicePct / 100))} of {allPoints.length} words
          </div>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 4 }}>
            {SLICES.map(pct => (
              <ChipBtn key={pct} active={slicePct === pct} onClick={() => setSlicePct(pct)}>
                {pct}%
              </ChipBtn>
            ))}
          </div>
        </Panel>

        {/* highlight */}
        <Panel label="highlight word">
          <input
            value={search}
            onChange={e => setSearch(e.target.value)}
            placeholder="type to highlight…"
            style={{ width: '100%', boxSizing: 'border-box' }}
          />
          {searchLower && (
            <div style={{ marginTop: 6, fontFamily: 'var(--font-mono)', fontSize: 11, color: matchCount ? 'var(--accent2)' : 'var(--muted)' }}>
              {matchCount} match{matchCount !== 1 ? 'es' : ''}
            </div>
          )}
        </Panel>

        {/* clustering */}
        <Panel label="clustering">
          <label style={{ display: 'flex', alignItems: 'center', gap: 8, fontFamily: 'var(--font-mono)', fontSize: 12, cursor: 'pointer', marginBottom: clusterOn ? 10 : 0 }}>
            <input type="checkbox" checked={clusterOn} onChange={e => setClusterOn(e.target.checked)} />
            color by cluster
          </label>
          {clusterOn && (
            <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap' }}>
              {CLUSTER_K_OPTIONS.map(k => (
                <ChipBtn key={k} active={clusterK === k} onClick={() => setClusterK(k)}>k={k}</ChipBtn>
              ))}
            </div>
          )}
        </Panel>

        {/* word list */}
        <Panel label={`words (${visiblePoints.length})`} flex>
          <div style={{ overflowY: 'auto', flex: 1 }}>
            {visiblePoints.map((p, i) => {
              const isMatch   = searchLower && p.word.includes(searchLower)
              const isActive  = search === p.word
              const color = clusterLabels
                ? CLUSTER_COLORS[clusterLabels[i] % CLUSTER_COLORS.length]
                : isMatch ? 'var(--text)' : 'var(--muted)'
              return (
                <div key={p.word}
                  onClick={() => setSearch(isActive ? '' : p.word)}
                  style={{
                    fontFamily: 'var(--font-mono)', fontSize: 12,
                    padding: '3px 6px', borderRadius: 3, cursor: 'pointer',
                    color, background: isActive ? 'var(--bg3)' : 'transparent',
                    transition: 'background 0.1s',
                  }}
                  onMouseEnter={e => { if (!isActive) e.currentTarget.style.background = 'var(--bg2)' }}
                  onMouseLeave={e => { if (!isActive) e.currentTarget.style.background = 'transparent' }}
                >
                  {p.word}
                </div>
              )
            })}
          </div>
        </Panel>
      </div>
    </div>
  )
}

// ── helpers ───────────────────────────────────────────────────────────────────
function Panel({ label, children, flex }) {
  return (
    <div style={{
      padding: '14px 16px', borderBottom: '1px solid var(--border)',
      display: 'flex', flexDirection: 'column',
      ...(flex ? { flex: 1, minHeight: 0 } : {}),
    }}>
      <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 10 }}>
        {label}
      </div>
      {children}
    </div>
  )
}

function ChipBtn({ active, onClick, children }) {
  return (
    <button onClick={onClick} style={{
      fontSize: 11, padding: '3px 10px', fontFamily: 'var(--font-mono)',
      background: active ? 'var(--accent)' : 'var(--bg3)',
      color: active ? '#fff' : 'var(--muted)',
      border: '1px solid var(--border)', borderRadius: 4, cursor: 'pointer',
    }}>
      {children}
    </button>
  )
}
