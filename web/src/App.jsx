import React, { useState } from 'react'
import WSGraphVisualization from './components/WSGraphVisualization'
import SmallWorldMetrics from './components/SmallWorldMetrics'
import TrainingDashboard from './components/TrainingDashboard'
import SparsityEvolution from './components/SparsityEvolution'

const tabs = [
  { id: 'ws-graph', label: 'WS Network', component: WSGraphVisualization },
  { id: 'metrics', label: 'Small-World Metrics', component: SmallWorldMetrics },
  { id: 'training', label: 'Training Dashboard', component: TrainingDashboard },
  { id: 'sparsity', label: 'Sparsity Evolution', component: SparsityEvolution },
]

function App() {
  const [activeTab, setActiveTab] = useState('ws-graph')

  const ActiveComponent = tabs.find(t => t.id === activeTab)?.component || WSGraphVisualization

  return (
    <div style={styles.container}>
      <header style={styles.header}>
        <h1 style={styles.title}>Multi-Modal WS Foundational Training</h1>
        <p style={styles.subtitle}>Interactive Network Visualizations</p>
      </header>

      <nav style={styles.nav}>
        {tabs.map(tab => (
          <button
            key={tab.id}
            style={{
              ...styles.tab,
              ...(activeTab === tab.id ? styles.activeTab : {})
            }}
            onClick={() => setActiveTab(tab.id)}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      <main style={styles.main}>
        <ActiveComponent />
      </main>

      <footer style={styles.footer}>
        <p>Multi-Modal WS Foundational Training - Interactive Visualizations</p>
      </footer>
    </div>
  )
}

const styles = {
  container: {
    minHeight: '100vh',
    display: 'flex',
    flexDirection: 'column',
  },
  header: {
    padding: '20px 40px',
    background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
    textAlign: 'center',
  },
  title: {
    fontSize: '2rem',
    fontWeight: 700,
    marginBottom: '5px',
  },
  subtitle: {
    fontSize: '1rem',
    opacity: 0.9,
  },
  nav: {
    display: 'flex',
    justifyContent: 'center',
    gap: '10px',
    padding: '15px',
    background: '#16213e',
    borderBottom: '1px solid #333',
  },
  tab: {
    padding: '10px 20px',
    border: 'none',
    borderRadius: '5px',
    background: '#0f3460',
    color: '#fff',
    cursor: 'pointer',
    fontSize: '0.9rem',
    transition: 'all 0.2s',
  },
  activeTab: {
    background: '#e94560',
  },
  main: {
    flex: 1,
    padding: '20px',
  },
  footer: {
    padding: '15px',
    textAlign: 'center',
    background: '#16213e',
    fontSize: '0.8rem',
    opacity: 0.7,
  },
}

export default App
