import React, { useState, useEffect } from 'react'
import { Shield, ShieldAlert, Video, History, Settings, Bell, Camera, Disc } from 'lucide-react'

function App() {
    const [safehouseMode, setSafehouseMode] = useState(false)
    const [status, setStatus] = useState("System Ready")
    const [lastFace, setLastFace] = useState("None")
    const [lastGesture, setLastGesture] = useState("None")
    const [logs, setLogs] = useState([])

    useEffect(() => {
        const ws = new WebSocket('ws://localhost:8000/ws')
        ws.onmessage = (event) => {
            const data = JSON.parse(event.data)
            if (data.type === 'state_update') {
                setSafehouseMode(data.safehouse_mode)
                setStatus(data.status)
                setLastFace(data.last_face)
                setLastGesture(data.last_gesture)
                if (data.logs) setLogs(data.logs)
            }
        }
        return () => ws.close()
    }, [])

    return (
        <div className="dashboard">
            <nav className="sidebar">
                <div className="logo">
                    <Shield color={safehouseMode ? "#10b981" : "#3b82f6"} size={32} />
                    <span>SafeHomeCam</span>
                </div>
                <div className="nav-items">
                    <div className="nav-item active"><Video size={20} /> Live Feed</div>
                    <div className="nav-item"><History size={20} /> Logs</div>
                    <div className="nav-item"><Settings size={20} /> Settings</div>
                </div>
                <div className="status-card">
                    <div className="status-label">Protective Shield</div>
                    <div className={`status-value ${safehouseMode ? 'safe' : 'inactive'}`}>
                        {safehouseMode ? 'SHIELD ACTIVE' : 'SHIELD INACTIVE'}
                    </div>
                </div>
            </nav>

            <main className="content">
                <header className="top-bar">
                    <div className="search-bar">Search incidents...</div>
                    <div className="actions">
                        <Bell size={20} />
                        <div className="user-profile">{lastFace[0] || 'U'}</div>
                    </div>
                </header>

                <div className="grid">
                    <div className="video-section">
                        <div className="video-card">
                            <div className="card-header">
                                <div className="title">
                                    <Disc size={16} className="rec-icon" /> Live Feed - Front Camera
                                </div>
                                <div className="badge">HD Live</div>
                            </div>
                            <div className="video-placeholder">
                                <img src="http://localhost:8000/video_feed" alt="Live stream" className="live-stream" />
                                <div className="overlay northern-overlay">
                                    MONITORING: {lastFace.toUpperCase()} • {new Date().toLocaleTimeString()}
                                </div>
                            </div>
                            <div className="video-controls">
                                <button className="control-btn"><Camera size={18} /></button>
                                <div className="status-pill">
                                    {lastGesture !== "None" ? `GESTURE: ${lastGesture}` : status}
                                </div>
                            </div>
                        </div>
                    </div>

                    <div className="activity-section">
                        <div className="card activity-card">
                            <h3>Recent Activity</h3>
                            <div className="activity-list">
                                {logs.length > 0 ? logs.map(log => (
                                    <div key={log.id} className="activity-item">
                                        <div className="time">{log.time}</div>
                                        <div className="event">{log.event}</div>
                                        <div className="result">{log.status}</div>
                                    </div>
                                )) : (
                                    <div className="empty-logs">Monitoring for events...</div>
                                )}
                            </div>
                        </div>
                    </div>
                </div>
            </main>
        </div>
    )
}

export default App
