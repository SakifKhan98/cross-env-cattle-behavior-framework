import { useEffect, useState } from 'react'
import { Link, useNavigate } from 'react-router-dom'

const STATUS_STYLES = {
  complete: 'bg-green-100 dark:bg-green-900 text-green-700 dark:text-green-300',
  running:  'bg-amber-100 dark:bg-amber-900 text-amber-700 dark:text-amber-300',
  pending:  'bg-gray-100 dark:bg-[#3D1517] text-gray-600 dark:text-gray-400',
  failed:   'bg-red-100 dark:bg-red-900 text-red-700 dark:text-red-400',
}

function StatusBadge({ status }) {
  return (
    <span className={`text-xs font-semibold px-2 py-0.5 rounded-full ${STATUS_STYLES[status] ?? STATUS_STYLES.pending}`}>
      {status}
    </span>
  )
}

function formatTime(iso) {
  if (!iso) return '—'
  try {
    return new Date(iso).toLocaleString()
  } catch {
    return iso
  }
}

export default function HistoryPage() {
  const [jobs, setJobs] = useState(null)
  const [error, setError] = useState(null)
  const navigate = useNavigate()

  useEffect(() => {
    fetch('/jobs')
      .then(r => { if (!r.ok) throw new Error(`Status ${r.status}`); return r.json() })
      .then(data => setJobs([...data].sort((a, b) => b.created_at.localeCompare(a.created_at))))
      .catch(e => setError(e.message))
  }, [])

  function handleRowClick(job) {
    if (job.status === 'complete') {
      navigate(`/jobs/${job.job_id}/results`)
    }
  }

  return (
    <div className="max-w-3xl mx-auto px-4 py-10">
      <div className="flex items-center justify-between mb-6">
        <h1 className="text-2xl font-semibold">Job History</h1>
        <Link to="/">
          <button className="text-sm bg-gray-200 dark:bg-[#3D1517] text-gray-700 dark:text-gray-300
            font-semibold px-4 py-2 rounded-lg hover:bg-gray-300 dark:hover:bg-[#4D1C1E] transition-colors">
            + New job
          </button>
        </Link>
      </div>

      {error && (
        <div className="bg-red-50 dark:bg-red-950 border border-red-200 dark:border-red-800
          text-red-700 dark:text-red-400 rounded-lg px-4 py-3 mb-6 text-sm">
          {error}
        </div>
      )}

      {jobs === null && !error && (
        <p className="text-gray-400 dark:text-gray-500">Loading…</p>
      )}

      {jobs?.length === 0 && (
        <p className="text-gray-400 dark:text-gray-500">No jobs yet.</p>
      )}

      {jobs?.length > 0 && (
        <div className="bg-white dark:bg-[#2A0D0E] border border-gray-100 dark:border-[#3D1517] rounded-xl overflow-hidden">
          <table className="w-full text-sm border-collapse">
            <thead>
              <tr className="text-left text-xs text-gray-400 dark:text-gray-500 uppercase tracking-wide
                border-b border-gray-100 dark:border-[#3D1517] bg-gray-50 dark:bg-[#220A0B]">
                <th className="px-4 py-2.5">Job ID</th>
                <th className="px-4 py-2.5">Video</th>
                <th className="px-4 py-2.5">Status</th>
                <th className="px-4 py-2.5">Submitted</th>
              </tr>
            </thead>
            <tbody>
              {jobs.map((job, i) => (
                <tr
                  key={job.job_id}
                  onClick={() => handleRowClick(job)}
                  className={`border-b border-gray-50 dark:border-[#3D1517]
                    ${job.status === 'complete'
                      ? 'cursor-pointer hover:bg-rose-50 dark:hover:bg-[#3D1517]'
                      : 'cursor-default'}
                    ${i % 2 === 0 ? '' : 'bg-gray-50/50 dark:bg-[#220A0B]'}`}
                >
                  <td className="px-4 py-2.5 font-mono text-xs text-gray-400 dark:text-gray-500">
                    {job.job_id.slice(0, 8)}…
                  </td>
                  <td className="px-4 py-2.5 text-gray-700 dark:text-gray-300 truncate max-w-[180px]">
                    {job.video_filename || '—'}
                  </td>
                  <td className="px-4 py-2.5">
                    <StatusBadge status={job.status} />
                  </td>
                  <td className="px-4 py-2.5 text-gray-500 dark:text-gray-400 text-xs">
                    {formatTime(job.created_at)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </div>
  )
}
