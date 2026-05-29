import { useState } from 'react'

const API = '/api/morphology'

const POS_OPTIONS = ['NOUN','VERB','ADJ','ADV','PRON','DET','ADP','CONJ','PART','NUM','X']

const POS_COLOR = {
  NOUN: 'noun', VERB: 'verb', ADJ: 'adj', ADV: 'adv',
}
const posClass = p => POS_COLOR[p] || 'other'

export default function MorphologyPage() {
  return (
    <div style={{ display: 'grid', gridTemplateColumns: '1fr 380px', height: '100%', overflow: 'hidden' }}>
      <AnalyzePanel />
      <LemmatizePanel />
    </div>
  )
}

// ── Full text analysis ────────────────────────────────────────────────────────

function AnalyzePanel() {
  const [text, setText] = useState('')
  const [tokens, setTokens] = useState([])
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')

  const run = async () => {
    if (!text.trim()) return
    setLoading(true); setError('')
    try {
      const res = await fetch(`${API}/analyze`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setTokens(data.tokens)
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  const posCounts = tokens.reduce((acc, t) => {
    if (t.pos) acc[t.pos] = (acc[t.pos] || 0) + 1
    return acc
  }, {})

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', borderRight: '1px solid var(--border)' }}>
      <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)' }}>
        <label>text for analysis</label>
        <textarea
          value={text}
          onChange={e => setText(e.target.value)}
          placeholder="Увядзіце беларускі тэкст..."
          rows={4}
          style={{ marginTop: 4, resize: 'vertical' }}
        />
        <div style={{ display: 'flex', gap: 8, marginTop: 8, alignItems: 'center' }}>
          <button className="primary" onClick={run} disabled={loading}>
            {loading ? 'analysing...' : 'analyse'}
          </button>
          {tokens.length > 0 && (
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)' }}>
              {tokens.length} tokens
            </span>
          )}
        </div>
        {error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}
      </div>

      {tokens.length > 0 && (
        <>
          <div style={{ padding: '10px 16px', borderBottom: '1px solid var(--border)', display: 'flex', flexWrap: 'wrap', gap: 8 }}>
            {Object.entries(posCounts).sort((a,b) => b[1]-a[1]).map(([pos, count]) => (
              <span key={pos} className={`tag ${posClass(pos)}`}>{pos} <span style={{ opacity: 0.6 }}>{count}</span></span>
            ))}
          </div>

          <div style={{ flex: 1, overflowY: 'auto', padding: 16 }}>
            <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'flex-start' }}>
              {tokens.map((t, i) => (
                <TokenCard key={i} token={t} />
              ))}
            </div>
          </div>
        </>
      )}

      {!tokens.length && !loading && (
        <div style={{ flex: 1, display: 'flex', alignItems: 'center', justifyContent: 'center', color: 'var(--muted)', fontFamily: 'var(--font-mono)', fontSize: 12 }}>
          enter text and press "analyse"
        </div>
      )}
    </div>
  )
}

function TokenCard({ token }) {
  return (
    <div style={{
      background: 'var(--bg2)',
      border: '1px solid var(--border)',
      borderRadius: 6,
      padding: '8px 10px',
      minWidth: 80,
      display: 'flex', flexDirection: 'column', gap: 4,
      transition: 'border-color 0.15s',
    }}
      onMouseEnter={e => e.currentTarget.style.borderColor = 'var(--accent)'}
      onMouseLeave={e => e.currentTarget.style.borderColor = 'var(--border)'}
    >
      <span style={{ fontFamily: 'var(--font-mono)', fontSize: 14, fontWeight: 500 }}>{token.text}</span>
      {token.pos && <span className={`tag ${posClass(token.pos)}`} style={{ alignSelf: 'flex-start' }}>{token.pos}</span>}
      {token.lemma && (
        <span style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: token.lemma !== token.text ? 'var(--muted)' : 'var(--border)' }}>
          → {token.lemma}
        </span>
      )}
    </div>
  )
}

// ── Single word lemmatizer ────────────────────────────────────────────────────

function LemmatizePanel() {
  const [word, setWord] = useState('')
  const [pos, setPos] = useState('VERB')
  const [result, setResult] = useState(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [history, setHistory] = useState([])

  const run = async () => {
    if (!word.trim()) return
    setLoading(true); setError('')
    try {
      const res = await fetch(`${API}/lemmatize`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ word, pos }),
      })
      if (!res.ok) throw new Error(await res.text())
      const data = await res.json()
      setResult(data)
      setHistory(h => [data, ...h].slice(0, 20))
    } catch (e) { setError(e.message) }
    finally { setLoading(false) }
  }

  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%', overflowY: 'auto' }}>
      <div style={{ padding: '12px 16px', borderBottom: '1px solid var(--border)' }}>
        <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 10 }}>lemmatizer</div>
        <div style={{ display: 'flex', gap: 6, alignItems: 'flex-end' }}>
          <div style={{ flex: 1 }}>
            <label>word</label>
            <input
              value={word}
              onChange={e => setWord(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && run()}
              placeholder="іду, бяжыш, прыйшлі..."
              style={{ marginTop: 4 }}
            />
          </div>
          <div>
            <label>POS</label>
            <select value={pos} onChange={e => setPos(e.target.value)} style={{ marginTop: 4, display: 'block' }}>
              {POS_OPTIONS.map(p => <option key={p} value={p}>{p}</option>)}
            </select>
          </div>
          <button className="primary" onClick={run} disabled={loading}>→</button>
        </div>
        {error && <div className="error" style={{ marginTop: 8 }}>{error}</div>}
      </div>

      {result && (
        <div style={{ padding: '16px', borderBottom: '1px solid var(--border)' }}>
          <div style={{ display: 'flex', alignItems: 'center', gap: 12 }}>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 20, color: 'var(--muted)' }}>{result.word}</span>
            <span style={{ color: 'var(--muted)', fontSize: 18 }}>→</span>
            <span style={{ fontFamily: 'var(--font-mono)', fontSize: 20, color: 'var(--accent2)', fontWeight: 500 }}>{result.lemma}</span>
            <span className={`tag ${posClass(result.pos)}`}>{result.pos}</span>
          </div>
        </div>
      )}

      {history.length > 0 && (
        <div style={{ padding: '12px 16px', flex: 1 }}>
          <div style={{ fontFamily: 'var(--font-mono)', fontSize: 11, color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.08em', marginBottom: 10 }}>history</div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
            {history.map((h, i) => (
              <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 8, cursor: 'pointer', padding: '6px 8px', borderRadius: 4, transition: 'background 0.1s' }}
                onClick={() => { setWord(h.word); setPos(h.pos); setResult(h) }}
                onMouseEnter={e => e.currentTarget.style.background = 'var(--bg3)'}
                onMouseLeave={e => e.currentTarget.style.background = 'transparent'}
              >
                <span className={`tag ${posClass(h.pos)}`}>{h.pos}</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12 }}>{h.word}</span>
                <span style={{ color: 'var(--muted)', fontSize: 12 }}>→</span>
                <span style={{ fontFamily: 'var(--font-mono)', fontSize: 12, color: 'var(--accent2)' }}>{h.lemma}</span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  )
}