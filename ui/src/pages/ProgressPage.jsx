import { useEffect, useRef, useState } from 'react'
import { useNavigate, useParams, Link } from 'react-router-dom'

const STAGE_NAMES = {
  1: 'Ingest', 2: 'Detect', 3: 'Track',
  4: 'Extract', 5: 'Classify', 6: 'Analyze', 7: 'Render',
}

function ProgressBar({ value, max }) {
  const pct = max > 0 ? Math.min(100, (value / max) * 100) : 0
  return (
    <div className="w-full bg-gray-100 dark:bg-[#3D1517] rounded-full h-2.5 overflow-hidden">
      <div
        style={{ width: `${pct}%`, backgroundColor: '#C6973F' }}
        className="h-full rounded-full transition-all duration-300"
        role="progressbar"
        aria-valuenow={value}
        aria-valuemax={max}
      />
    </div>
  )
}

export default function ProgressPage() {
  const { jobId } = useParams()
  const navigate = useNavigate()
  const esRef = useRef(null)

  const [event, setEvent] = useState(null)
  const [failed, setFailed] = useState(null)

  useEffect(() => {
    const es = new EventSource(`/jobs/${jobId}/stream`)
    esRef.current = es

    es.onmessage = (e) => {
      let data
      try { data = JSON.parse(e.data) } catch { return }

      if (data.status === 'complete') {
        es.close()
        navigate(`/jobs/${jobId}/results`)
        return
      }
      if (data.status === 'failed') {
        es.close()
        setFailed(data.error || 'Unknown error')
        return
      }
      setEvent(data)
    }

    es.onerror = () => {
      es.close()
      setFailed('Lost connection to server. The job may still be running — refresh to check.')
    }

    return () => es.close()
  }, [jobId, navigate])

  const stageNum = event?.stage ?? 0
  const stageName = event?.stage_name ?? STAGE_NAMES[stageNum] ?? '…'
  const totalStages = event?.total_stages ?? 7
  const frame = event?.frame ?? 0
  const totalFrames = event?.total_frames ?? 0

  if (failed) {
    return (
      <div className="max-w-xl mx-auto px-4 py-10">
        <h1 className="text-2xl font-semibold mb-4">Job failed</h1>
        <div className="bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800
          text-red-700 dark:text-red-400 rounded-lg px-4 py-3 mb-6 text-sm break-words">
          {failed}
        </div>
        <Link to="/">
          <button className="bg-gray-200 dark:bg-[#3D1517] text-gray-700 dark:text-gray-300
            font-semibold px-5 py-2.5 rounded-lg hover:bg-gray-300 dark:hover:bg-[#4D1C1E] transition-colors">
            Back to upload
          </button>
        </Link>
      </div>
    )
  }

  return (
    <div className="max-w-xl mx-auto px-4 py-10">
      <h1 className="text-2xl font-semibold mb-1">Running inference…</h1>
      <p className="text-xs text-gray-400 dark:text-gray-500 mb-8 font-mono">{jobId}</p>

      <div className="bg-white dark:bg-[#2A0D0E] border border-gray-100 dark:border-[#3D1517] rounded-xl p-5 mb-4 space-y-2">
        <p className="text-sm text-gray-500 dark:text-gray-400">
          {stageNum > 0
            ? `Stage ${stageNum} / ${totalStages}: ${stageName}`
            : 'Starting…'}
        </p>
        <ProgressBar value={stageNum} max={totalStages} />
      </div>

      <div className="bg-white dark:bg-[#2A0D0E] border border-gray-100 dark:border-[#3D1517] rounded-xl p-5 space-y-2">
        <p className="text-sm text-gray-500 dark:text-gray-400">
          {totalFrames > 0
            ? `Frame ${frame.toLocaleString()} / ${totalFrames.toLocaleString()}`
            : 'Waiting for first event…'}
        </p>
        <ProgressBar value={frame} max={totalFrames} />
      </div>
    </div>
  )
}
