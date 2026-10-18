import { useEffect, useRef, useState } from 'react'
import { useParams, Link } from 'react-router-dom'

// Behavior palette — matches render_behavior_video.py RGB values
const BEHAVIOR_COLORS = {
  0: '#50C850', // Standing  — green
  1: '#5082FF', // Lying     — blue
  2: '#32DCDC', // Foraging  — cyan
  3: '#FFC832', // Drinking  — yellow
  4: '#C850FF', // Ruminating — purple
  5: '#FF8C32', // Grooming  — orange
  6: '#D97706', // Other     — amber (was grey #A0A0A0, invisible in dark mode)
}
const FALLBACK_COLOR = '#A0A0A0'

function behaviorColor(labelId) {
  return BEHAVIOR_COLORS[labelId] ?? FALLBACK_COLOR
}

function shortId(tid) {
  // Show last ~10 chars to keep rows compact
  return tid.length > 12 ? '…' + tid.slice(-10) : tid
}

// ── Tooltip ──────────────────────────────────────────────────────────────────
function Tooltip({ children, text }) {
  const [show, setShow] = useState(false)
  return (
    <span
      className="relative"
      onMouseEnter={() => setShow(true)}
      onMouseLeave={() => setShow(false)}
    >
      {children}
      {show && text && (
        <span className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1.5 z-50
          bg-gray-900 dark:bg-[#4D1C1E] text-white text-xs rounded px-2 py-1 whitespace-nowrap pointer-events-none shadow-lg">
          {text}
        </span>
      )}
    </span>
  )
}

// ── Timeline Gantt chart ──────────────────────────────────────────────────────
const ROW_H = 28
const BAR_H = ROW_H - 6
const LABEL_W = 110

function TimelineChart({ timelines, totalDuration }) {
  if (!timelines?.length) return null
  const dur = totalDuration || 1

  return (
    <div>
      <h2 className="text-base font-semibold text-gray-700 dark:text-gray-300 mb-3">
        Behavioral Timeline
      </h2>
      <div style={{ overflowX: 'auto' }}>
        <div style={{ minWidth: 500 }}>
          {/* x-axis labels */}
          <div style={{ display: 'flex', paddingLeft: LABEL_W, marginBottom: 4, fontSize: 11, color: '#9ca3af', userSelect: 'none' }}>
            {[0, 0.25, 0.5, 0.75, 1].map(t => (
              <span key={t} style={{ flex: 1, textAlign: 'center' }}>{(t * dur).toFixed(1)}s</span>
            ))}
          </div>
          {timelines.map(({ track_id, is_outlier, segments }) => (
            <div key={track_id} style={{ display: 'flex', alignItems: 'center', marginBottom: 4, height: ROW_H }}>
              {/* Row label */}
              <div
                style={{
                  width: LABEL_W, minWidth: LABEL_W,
                  fontSize: 11, fontFamily: 'monospace', textAlign: 'right',
                  paddingRight: 8, overflow: 'hidden', whiteSpace: 'nowrap',
                  userSelect: 'none', fontWeight: is_outlier ? 700 : 400,
                  color: is_outlier ? '#ef4444' : '#6b7280',
                }}
                title={track_id}
              >
                {shortId(track_id)}{is_outlier && ' ⚠'}
              </div>
              {/* Segment bar */}
              <div style={{ position: 'relative', flex: 1, height: BAR_H, borderRadius: 4, overflow: 'hidden', backgroundColor: 'var(--chart-track)' }}>
                {segments.map((seg, i) => {
                  const left = (seg.start_sec / dur) * 100
                  const width = Math.max((seg.duration_sec / dur) * 100, 0.3)
                  return (
                    <div
                      key={i}
                      title={`${seg.behavior} | ${seg.start_sec.toFixed(2)}s–${seg.end_sec.toFixed(2)}s (${seg.duration_sec.toFixed(2)}s)`}
                      style={{
                        position: 'absolute', top: 0, bottom: 0,
                        left: `${left}%`, width: `${width}%`,
                        backgroundColor: behaviorColor(seg.label_id),
                        opacity: 0.85, borderRadius: 2, cursor: 'default',
                      }}
                    />
                  )
                })}
              </div>
            </div>
          ))}
        </div>
      </div>
      {/* Legend */}
      <div style={{ display: 'flex', flexWrap: 'wrap', gap: 12, marginTop: 12 }}>
        {Object.entries(BEHAVIOR_COLORS).map(([id, color]) => {
          const names = ['Standing','Lying','Foraging','Drinking','Ruminating','Grooming','Other']
          return (
            <div key={id} style={{ display: 'flex', alignItems: 'center', gap: 6, fontSize: 11, color: '#9ca3af' }}>
              <span style={{ width: 12, height: 12, borderRadius: 2, display: 'inline-block', backgroundColor: color }} />
              {names[+id]}
            </div>
          )
        })}
      </div>
    </div>
  )
}

// ── Activity Budget chart ─────────────────────────────────────────────────────
function BudgetRow({ track_id, is_outlier, behaviors, isHerd }) {
  const nonzero = (behaviors || [])
    .filter(b => b.pct_time > 0)
    .sort((a, b) => a.label_id - b.label_id)

  // Compute cumulative left offsets for absolute-positioned bars
  let offset = 0
  const bars = nonzero.map(b => {
    const left = offset
    offset += b.pct_time
    return { ...b, left }
  })

  const labelColor = isHerd ? '#6b7280' : is_outlier ? '#ef4444' : '#6b7280'
  const labelStyle = {
    width: LABEL_W, minWidth: LABEL_W,
    fontSize: 11, fontFamily: 'monospace', textAlign: 'right',
    paddingRight: 8, overflow: 'hidden', whiteSpace: 'nowrap',
    userSelect: 'none', color: labelColor,
    fontWeight: (!isHerd && is_outlier) ? 700 : 400,
    fontStyle: isHerd ? 'italic' : 'normal',
  }

  return (
    <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 6 }}>
      <div style={labelStyle} title={track_id}>
        {isHerd ? track_id : shortId(track_id)}{!isHerd && is_outlier && ' ⚠'}
      </div>
      <div style={{ position: 'relative', flex: 1, height: 20, borderRadius: 4, overflow: 'hidden', backgroundColor: 'var(--chart-track)' }}>
        {bars.map((b, i) => (
          <div
            key={i}
            title={`${b.behavior}: ${b.pct_time.toFixed(1)}%`}
            style={{
              position: 'absolute', top: 0, bottom: 0,
              left: `${b.left}%`, width: `${b.pct_time}%`,
              backgroundColor: behaviorColor(b.label_id),
              opacity: isHerd ? 0.6 : 0.85,
              cursor: 'default',
            }}
          />
        ))}
      </div>
    </div>
  )
}

function BudgetChart({ activityBudget }) {
  if (!activityBudget?.length) return null

  const behaviorTotals = {}
  for (const track of activityBudget) {
    for (const b of track.behaviors) {
      behaviorTotals[b.label_id] = (behaviorTotals[b.label_id] || 0) + b.pct_time
    }
  }
  const n = activityBudget.length
  const herdBehaviors = Object.entries(behaviorTotals).map(([id, total]) => ({
    label_id: +id,
    behavior: ['Standing','Lying','Foraging','Drinking','Ruminating','Grooming','Other'][+id] ?? 'Unknown',
    pct_time: total / n,
  }))

  return (
    <div>
      <h2 className="text-base font-semibold text-gray-700 dark:text-gray-300 mb-3">
        Activity Budget
      </h2>
      <div style={{ overflowX: 'auto' }}>
        <div style={{ minWidth: 400 }}>
          {activityBudget.map(row => (
            <BudgetRow key={row.track_id} {...row} isHerd={false} />
          ))}
          <BudgetRow
            key="herd"
            track_id="Herd Average"
            is_outlier={false}
            behaviors={herdBehaviors}
            isHerd={true}
          />
        </div>
      </div>
    </div>
  )
}

// ── Outlier table ─────────────────────────────────────────────────────────────
function OutlierTable({ outliers }) {
  if (!outliers?.length) return null
  return (
    <div>
      <h2 className="text-base font-semibold text-gray-700 dark:text-gray-300 mb-3">
        Outlier Alerts
      </h2>
      <div className="overflow-x-auto">
        <table className="w-full text-sm border-collapse">
          <thead>
            <tr className="text-left text-xs text-gray-400 dark:text-gray-500 uppercase tracking-wide border-b border-gray-200 dark:border-[#4D1C1E]">
              <th className="pb-2 pr-4">Track ID</th>
              <th className="pb-2 pr-4">Behavior</th>
              <th className="pb-2 pr-4 text-right">Animal %</th>
              <th className="pb-2 pr-4 text-right">Herd Median %</th>
              <th className="pb-2 text-right">Deviation</th>
            </tr>
          </thead>
          <tbody>
            {outliers.map((row, i) => (
              <tr
                key={i}
                className="border-b border-gray-100 dark:border-[#3D1517]
                  text-red-700 dark:text-red-400 font-medium"
              >
                <td className="py-1.5 pr-4 font-mono text-xs" title={row.track_id}>{shortId(row.track_id)}</td>
                <td className="py-1.5 pr-4">{row.behavior}</td>
                <td className="py-1.5 pr-4 text-right">{row.pct_time.toFixed(1)}%</td>
                <td className="py-1.5 pr-4 text-right">{row.baseline_median.toFixed(1)}%</td>
                <td className="py-1.5 text-right">{row.deviation.toFixed(1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}

// ── Download panel ────────────────────────────────────────────────────────────
function DownloadPanel({ manifest }) {
  const zipUrl = manifest.timelines?.length
    ? `/results/${manifest.annotated_video.split('/')[2]}/timelines_zip`
    : null

  function DL({ href, label }) {
    return (
      <a
        href={href}
        download
        style={{ color: '#C6973F' }}
        className="flex items-center gap-2 text-sm hover:underline hover:brightness-110"
      >
        <span>↓</span>{label}
      </a>
    )
  }

  return (
    <div>
      <h2 className="text-base font-semibold text-gray-700 dark:text-gray-300 mb-3">Downloads</h2>
      <div className="space-y-2">
        <DL href={manifest.annotated_video} label="Annotated video (.mp4)" />
        <DL href={manifest.activity_budget} label="Activity budget (.csv)" />
        <DL href={manifest.behavior_deviation} label="Behavior deviation (.csv)" />
        {manifest.timelines?.length > 0 && (
          <details className="mt-1">
            <summary className="text-xs text-gray-400 dark:text-gray-500 cursor-pointer hover:text-gray-600 dark:hover:text-gray-300 select-none">
              Timeline CSVs ({manifest.timelines.length} files)
            </summary>
            <div className="pl-3 mt-1 space-y-1.5">
              {manifest.timelines.map((url) => (
                <DL key={url} href={url} label={url.split('/').pop()} />
              ))}
            </div>
          </details>
        )}
      </div>
    </div>
  )
}

// ── Main page ─────────────────────────────────────────────────────────────────
export default function ResultsPage() {
  const { jobId } = useParams()
  const [manifest, setManifest] = useState(null)
  const [analytics, setAnalytics] = useState(null)
  const [error, setError] = useState(null)

  useEffect(() => {
    fetch(`/jobs/${jobId}/results`)
      .then(r => { if (!r.ok) throw new Error(`Status ${r.status}`); return r.json() })
      .then(setManifest)
      .catch(e => setError(e.message))

    fetch(`/jobs/${jobId}/analytics`)
      .then(r => r.ok ? r.json() : null)
      .then(data => { if (data) setAnalytics(data) })
      .catch(() => {})
  }, [jobId])

  if (error) return (
    <div className="max-w-3xl mx-auto px-4 py-10">
      <h1 className="text-2xl font-semibold mb-4">Results</h1>
      <div className="bg-red-50 dark:bg-[#2A0D0E] border border-red-200 dark:border-red-800
        text-red-700 dark:text-red-400 rounded-lg px-4 py-3 mb-6 text-sm">
        {error}
      </div>
      <Link to="/">
        <button className="bg-gray-200 dark:bg-[#3D1517] text-gray-700 dark:text-gray-300
          font-semibold px-5 py-2.5 rounded-lg hover:bg-gray-300 dark:hover:bg-[#4D1C1E] transition-colors">
          Back to upload
        </button>
      </Link>
    </div>
  )

  if (!manifest) return (
    <div className="max-w-3xl mx-auto px-4 py-10 text-gray-400 dark:text-gray-500">
      Loading results…
    </div>
  )

  return (
    <div className="max-w-3xl mx-auto px-4 py-10 space-y-8">
      <div>
        <h1 className="text-2xl font-semibold mb-1">Inference complete</h1>
        <p className="text-xs text-gray-400 dark:text-gray-500 font-mono">{jobId}</p>
      </div>

      {/* Annotated video */}
      <section className="bg-white dark:bg-[#2A0D0E] border border-gray-100 dark:border-[#3D1517] rounded-xl p-5">
        <h2 className="text-base font-semibold text-gray-700 dark:text-gray-300 mb-3">Annotated Video</h2>
        <video
          controls
          className="w-full rounded-lg bg-black"
          src={manifest.annotated_video}
        >
          Your browser does not support the video tag.
        </video>
      </section>

      {/* Timeline chart */}
      {analytics?.timelines?.length > 0 && (
        <section className="bg-white dark:bg-[#2A0D0E] border border-gray-100 dark:border-[#3D1517] rounded-xl p-5">
          <TimelineChart
            timelines={analytics.timelines}
            totalDuration={analytics.total_duration_sec}
          />
        </section>
      )}

      {/* Activity budget */}
      {analytics?.activity_budget?.length > 0 && (
        <section className="bg-white dark:bg-[#2A0D0E] border border-gray-100 dark:border-[#3D1517] rounded-xl p-5">
          <BudgetChart activityBudget={analytics.activity_budget} />
        </section>
      )}

      {/* Outlier table */}
      {analytics?.outliers?.length > 0 && (
        <section className="bg-white dark:bg-[#2A0D0E] border border-gray-100 dark:border-[#3D1517] rounded-xl p-5">
          <OutlierTable outliers={analytics.outliers} />
        </section>
      )}

      {/* Downloads */}
      <section className="bg-white dark:bg-[#2A0D0E] border border-gray-100 dark:border-[#3D1517] rounded-xl p-5">
        <DownloadPanel manifest={manifest} />
      </section>

      <div className="flex gap-3">
        <Link to="/history">
          <button className="bg-gray-200 dark:bg-gray-800 text-gray-700 dark:text-gray-300
            font-semibold px-5 py-2.5 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-700 transition-colors">
            Job history
          </button>
        </Link>
        <Link to="/">
          <button className="bg-gray-200 dark:bg-gray-800 text-gray-700 dark:text-gray-300
            font-semibold px-5 py-2.5 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-700 transition-colors">
            Run another video
          </button>
        </Link>
      </div>
    </div>
  )
}
