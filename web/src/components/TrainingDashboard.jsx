import React, { useState, useEffect, useRef } from 'react'
import Plot from 'react-plotly.js'

/**
 * Simulated training dashboard showing loss curves, accuracy, and learning rate.
 */
function TrainingDashboard() {
  const [isTraining, setIsTraining] = useState(false)
  const [epoch, setEpoch] = useState(0)
  const [history, setHistory] = useState({
    trainLoss: [],
    valLoss: [],
    trainAcc: [],
    valAcc: [],
    lr: [],
  })
  const intervalRef = useRef(null)

  // Simulate training step
  const simulateTrainingStep = (currentEpoch) => {
    // Simulated loss curves (exponential decay with noise)
    const trainLoss = 2.5 * Math.exp(-0.15 * currentEpoch) + 0.1 + Math.random() * 0.1
    const valLoss = 2.5 * Math.exp(-0.12 * currentEpoch) + 0.15 + Math.random() * 0.15

    // Simulated accuracy (sigmoid growth)
    const trainAcc = 95 / (1 + Math.exp(-0.2 * (currentEpoch - 15))) + Math.random() * 2
    const valAcc = 90 / (1 + Math.exp(-0.18 * (currentEpoch - 17))) + Math.random() * 3

    // Cosine annealing learning rate
    const lr = 0.001 * (1 + Math.cos(Math.PI * currentEpoch / 50)) / 2

    return { trainLoss, valLoss, trainAcc, valAcc, lr }
  }

  // Start training simulation
  const startTraining = () => {
    setIsTraining(true)
    setEpoch(0)
    setHistory({
      trainLoss: [],
      valLoss: [],
      trainAcc: [],
      valAcc: [],
      lr: [],
    })
  }

  // Stop training
  const stopTraining = () => {
    setIsTraining(false)
    if (intervalRef.current) {
      clearInterval(intervalRef.current)
    }
  }

  // Reset
  const reset = () => {
    stopTraining()
    setEpoch(0)
    setHistory({
      trainLoss: [],
      valLoss: [],
      trainAcc: [],
      valAcc: [],
      lr: [],
    })
  }

  // Training loop
  useEffect(() => {
    if (isTraining && epoch < 50) {
      intervalRef.current = setTimeout(() => {
        const metrics = simulateTrainingStep(epoch)
        setHistory(prev => ({
          trainLoss: [...prev.trainLoss, metrics.trainLoss],
          valLoss: [...prev.valLoss, metrics.valLoss],
          trainAcc: [...prev.trainAcc, metrics.trainAcc],
          valAcc: [...prev.valAcc, metrics.valAcc],
          lr: [...prev.lr, metrics.lr],
        }))
        setEpoch(e => e + 1)
      }, 200)
    } else if (epoch >= 50) {
      setIsTraining(false)
    }

    return () => {
      if (intervalRef.current) {
        clearTimeout(intervalRef.current)
      }
    }
  }, [isTraining, epoch])

  const epochs = Array.from({ length: history.trainLoss.length }, (_, i) => i + 1)

  return (
    <div style={styles.container}>
      <div style={styles.controls}>
        <button
          style={{...styles.button, background: isTraining ? '#e94560' : '#667eea'}}
          onClick={isTraining ? stopTraining : startTraining}
        >
          {isTraining ? 'Pause' : epoch > 0 ? 'Resume' : 'Start Training'}
        </button>
        <button style={styles.button} onClick={reset}>
          Reset
        </button>
        <div style={styles.status}>
          Epoch: <strong>{epoch}</strong> / 50
        </div>
      </div>

      <div style={styles.plots}>
        {/* Loss Plot */}
        <div style={styles.plotContainer}>
          <Plot
            data={[
              {
                x: epochs,
                y: history.trainLoss,
                type: 'scatter',
                mode: 'lines',
                name: 'Train Loss',
                line: { color: '#667eea', width: 2 },
              },
              {
                x: epochs,
                y: history.valLoss,
                type: 'scatter',
                mode: 'lines',
                name: 'Val Loss',
                line: { color: '#e94560', width: 2 },
              },
            ]}
            layout={{
              title: 'Loss Curves',
              xaxis: { title: 'Epoch', gridcolor: '#333', range: [0, 50] },
              yaxis: { title: 'Loss', gridcolor: '#333', range: [0, 3] },
              paper_bgcolor: '#0f3460',
              plot_bgcolor: '#0f3460',
              font: { color: '#fff', size: 10 },
              legend: { x: 0.7, y: 0.95 },
              margin: { t: 40, b: 40, l: 50, r: 20 },
            }}
            style={{ width: '100%', height: '250px' }}
            config={{ displayModeBar: false }}
          />
        </div>

        {/* Accuracy Plot */}
        <div style={styles.plotContainer}>
          <Plot
            data={[
              {
                x: epochs,
                y: history.trainAcc,
                type: 'scatter',
                mode: 'lines',
                name: 'Train Acc',
                line: { color: '#667eea', width: 2 },
              },
              {
                x: epochs,
                y: history.valAcc,
                type: 'scatter',
                mode: 'lines',
                name: 'Val Acc',
                line: { color: '#e94560', width: 2 },
              },
            ]}
            layout={{
              title: 'Accuracy',
              xaxis: { title: 'Epoch', gridcolor: '#333', range: [0, 50] },
              yaxis: { title: 'Accuracy (%)', gridcolor: '#333', range: [0, 100] },
              paper_bgcolor: '#0f3460',
              plot_bgcolor: '#0f3460',
              font: { color: '#fff', size: 10 },
              legend: { x: 0.7, y: 0.2 },
              margin: { t: 40, b: 40, l: 50, r: 20 },
            }}
            style={{ width: '100%', height: '250px' }}
            config={{ displayModeBar: false }}
          />
        </div>

        {/* Learning Rate Plot */}
        <div style={styles.plotContainer}>
          <Plot
            data={[
              {
                x: epochs,
                y: history.lr,
                type: 'scatter',
                mode: 'lines',
                name: 'Learning Rate',
                line: { color: '#4ade80', width: 2 },
                fill: 'tozeroy',
                fillcolor: 'rgba(74, 222, 128, 0.2)',
              },
            ]}
            layout={{
              title: 'Learning Rate (Cosine Annealing)',
              xaxis: { title: 'Epoch', gridcolor: '#333', range: [0, 50] },
              yaxis: { title: 'LR', gridcolor: '#333' },
              paper_bgcolor: '#0f3460',
              plot_bgcolor: '#0f3460',
              font: { color: '#fff', size: 10 },
              margin: { t: 40, b: 40, l: 50, r: 20 },
            }}
            style={{ width: '100%', height: '200px' }}
            config={{ displayModeBar: false }}
          />
        </div>
      </div>

      {/* Current Metrics */}
      <div style={styles.metricsRow}>
        <div style={styles.metricCard}>
          <div style={styles.metricLabel}>Train Loss</div>
          <div style={styles.metricValue}>
            {history.trainLoss.length > 0
              ? history.trainLoss[history.trainLoss.length - 1].toFixed(4)
              : '-'}
          </div>
        </div>
        <div style={styles.metricCard}>
          <div style={styles.metricLabel}>Val Loss</div>
          <div style={styles.metricValue}>
            {history.valLoss.length > 0
              ? history.valLoss[history.valLoss.length - 1].toFixed(4)
              : '-'}
          </div>
        </div>
        <div style={styles.metricCard}>
          <div style={styles.metricLabel}>Train Acc</div>
          <div style={styles.metricValue}>
            {history.trainAcc.length > 0
              ? history.trainAcc[history.trainAcc.length - 1].toFixed(1) + '%'
              : '-'}
          </div>
        </div>
        <div style={styles.metricCard}>
          <div style={styles.metricLabel}>Val Acc</div>
          <div style={styles.metricValue}>
            {history.valAcc.length > 0
              ? history.valAcc[history.valAcc.length - 1].toFixed(1) + '%'
              : '-'}
          </div>
        </div>
      </div>
    </div>
  )
}

const styles = {
  container: {
    maxWidth: '900px',
    margin: '0 auto',
  },
  controls: {
    display: 'flex',
    gap: '15px',
    alignItems: 'center',
    padding: '15px',
    background: '#16213e',
    borderRadius: '8px',
    marginBottom: '20px',
  },
  button: {
    padding: '10px 20px',
    background: '#667eea',
    border: 'none',
    borderRadius: '5px',
    color: '#fff',
    cursor: 'pointer',
    fontWeight: 'bold',
  },
  status: {
    marginLeft: 'auto',
    fontSize: '1.1rem',
  },
  plots: {
    display: 'flex',
    flexDirection: 'column',
    gap: '15px',
  },
  plotContainer: {
    background: '#0f3460',
    borderRadius: '8px',
    padding: '10px',
  },
  metricsRow: {
    display: 'flex',
    gap: '15px',
    marginTop: '20px',
  },
  metricCard: {
    flex: 1,
    padding: '15px',
    background: '#16213e',
    borderRadius: '8px',
    textAlign: 'center',
  },
  metricLabel: {
    fontSize: '0.9rem',
    opacity: 0.7,
    marginBottom: '5px',
  },
  metricValue: {
    fontSize: '1.5rem',
    fontWeight: 'bold',
  },
}

export default TrainingDashboard
