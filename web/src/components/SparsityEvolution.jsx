import React, { useState, useEffect, useRef } from 'react'
import * as d3 from 'd3'
import Plot from 'react-plotly.js'

/**
 * Visualization of sparse network topology evolution during training.
 * Demonstrates SET (Sparse Evolutionary Training) rewiring.
 */
function SparsityEvolution() {
  const svgRef = useRef(null)
  const [step, setStep] = useState(0)
  const [isAnimating, setIsAnimating] = useState(false)
  const [sparsity, setSparsity] = useState(0.7)
  const [pruneRate, setPruneRate] = useState(0.3)
  const [maskHistory, setMaskHistory] = useState([])
  const intervalRef = useRef(null)

  const matrixSize = 16

  // Generate initial random sparse mask
  const generateMask = (size, density) => {
    const mask = []
    for (let i = 0; i < size; i++) {
      const row = []
      for (let j = 0; j < size; j++) {
        row.push(Math.random() < density ? 1 : 0)
      }
      mask.push(row)
    }
    return mask
  }

  // Generate random weights
  const generateWeights = (size) => {
    const weights = []
    for (let i = 0; i < size; i++) {
      const row = []
      for (let j = 0; j < size; j++) {
        row.push((Math.random() - 0.5) * 2)
      }
      weights.push(row)
    }
    return weights
  }

  // SET rewiring step
  const setRewireStep = (mask, weights, pruneRate) => {
    const size = mask.length
    const newMask = mask.map(row => [...row])

    // Find active connections
    const activeConnections = []
    for (let i = 0; i < size; i++) {
      for (let j = 0; j < size; j++) {
        if (mask[i][j]) {
          activeConnections.push({
            i, j,
            magnitude: Math.abs(weights[i][j]),
          })
        }
      }
    }

    // Sort by magnitude (smallest first)
    activeConnections.sort((a, b) => a.magnitude - b.magnitude)

    // Prune smallest
    const numToPrune = Math.floor(activeConnections.length * pruneRate)
    const toPrune = activeConnections.slice(0, numToPrune)

    toPrune.forEach(conn => {
      newMask[conn.i][conn.j] = 0
    })

    // Grow random new connections
    let grown = 0
    while (grown < numToPrune) {
      const i = Math.floor(Math.random() * size)
      const j = Math.floor(Math.random() * size)
      if (newMask[i][j] === 0) {
        newMask[i][j] = 1
        grown++
      }
    }

    return newMask
  }

  // Initialize
  useEffect(() => {
    const initialMask = generateMask(matrixSize, 1 - sparsity)
    setMaskHistory([initialMask])
    setStep(0)
  }, [sparsity])

  // Animation loop
  useEffect(() => {
    if (isAnimating && step < 20) {
      intervalRef.current = setTimeout(() => {
        const weights = generateWeights(matrixSize)
        const currentMask = maskHistory[maskHistory.length - 1]
        const newMask = setRewireStep(currentMask, weights, pruneRate)

        setMaskHistory(prev => [...prev, newMask])
        setStep(s => s + 1)
      }, 500)
    } else if (step >= 20) {
      setIsAnimating(false)
    }

    return () => {
      if (intervalRef.current) clearTimeout(intervalRef.current)
    }
  }, [isAnimating, step, maskHistory, pruneRate])

  // D3 visualization
  useEffect(() => {
    if (!svgRef.current || maskHistory.length === 0) return

    const svg = d3.select(svgRef.current)
    const size = 300
    const cellSize = size / matrixSize
    const mask = maskHistory[maskHistory.length - 1]

    svg.selectAll('*').remove()

    const g = svg.append('g').attr('transform', 'translate(10, 10)')

    // Draw cells
    for (let i = 0; i < matrixSize; i++) {
      for (let j = 0; j < matrixSize; j++) {
        g.append('rect')
          .attr('x', j * cellSize)
          .attr('y', i * cellSize)
          .attr('width', cellSize - 1)
          .attr('height', cellSize - 1)
          .attr('fill', mask[i][j] ? '#667eea' : '#1a1a2e')
          .attr('rx', 2)
      }
    }

    // Labels
    svg.append('text')
      .attr('x', size / 2 + 10)
      .attr('y', size + 30)
      .attr('text-anchor', 'middle')
      .attr('fill', '#888')
      .text('Weight Matrix Connectivity')

  }, [maskHistory])

  // Calculate sparsity history
  const sparsityHistory = maskHistory.map(mask => {
    const total = matrixSize * matrixSize
    const nonzero = mask.flat().filter(v => v === 1).length
    return ((total - nonzero) / total * 100).toFixed(1)
  })

  const reset = () => {
    setIsAnimating(false)
    const initialMask = generateMask(matrixSize, 1 - sparsity)
    setMaskHistory([initialMask])
    setStep(0)
  }

  return (
    <div style={styles.container}>
      <div style={styles.controls}>
        <div style={styles.control}>
          <label>Initial Sparsity: {(sparsity * 100).toFixed(0)}%</label>
          <input
            type="range"
            min="0.5"
            max="0.95"
            step="0.05"
            value={sparsity}
            onChange={e => setSparsity(parseFloat(e.target.value))}
            disabled={isAnimating}
          />
        </div>
        <div style={styles.control}>
          <label>Prune Rate: {(pruneRate * 100).toFixed(0)}%</label>
          <input
            type="range"
            min="0.1"
            max="0.5"
            step="0.05"
            value={pruneRate}
            onChange={e => setPruneRate(parseFloat(e.target.value))}
            disabled={isAnimating}
          />
        </div>
        <button
          style={{...styles.button, background: isAnimating ? '#e94560' : '#667eea'}}
          onClick={() => setIsAnimating(!isAnimating)}
        >
          {isAnimating ? 'Pause' : step > 0 ? 'Resume' : 'Start SET'}
        </button>
        <button style={styles.button} onClick={reset}>
          Reset
        </button>
        <div style={styles.status}>
          Step: <strong>{step}</strong> / 20
        </div>
      </div>

      <div style={styles.visualization}>
        <div style={styles.matrixContainer}>
          <svg ref={svgRef} width={320} height={350} style={styles.svg} />
        </div>

        <div style={styles.plotContainer}>
          <Plot
            data={[
              {
                x: Array.from({ length: sparsityHistory.length }, (_, i) => i),
                y: sparsityHistory,
                type: 'scatter',
                mode: 'lines+markers',
                name: 'Sparsity',
                line: { color: '#667eea', width: 2 },
                marker: { size: 6 },
              },
            ]}
            layout={{
              title: 'Sparsity Over Time',
              xaxis: { title: 'SET Step', gridcolor: '#333', range: [-0.5, 20] },
              yaxis: { title: 'Sparsity (%)', gridcolor: '#333', range: [40, 100] },
              paper_bgcolor: '#0f3460',
              plot_bgcolor: '#0f3460',
              font: { color: '#fff', size: 10 },
              margin: { t: 40, b: 50, l: 50, r: 20 },
            }}
            style={{ width: '100%', height: '300px' }}
            config={{ displayModeBar: false }}
          />
        </div>
      </div>

      <div style={styles.explanation}>
        <h3>SET: Sparse Evolutionary Training</h3>
        <p>
          The SET algorithm maintains sparse networks by periodically:
        </p>
        <ol>
          <li><strong>Prune</strong>: Remove connections with smallest weight magnitude</li>
          <li><strong>Grow</strong>: Add new random connections to maintain density</li>
        </ol>
        <p>
          Key insight: The network "explores" different topologies during training,
          potentially finding better sparse structures than static pruning.
        </p>
        <div style={styles.legend}>
          <span style={{...styles.legendItem, background: '#667eea'}}>Active Connection</span>
          <span style={{...styles.legendItem, background: '#1a1a2e', border: '1px solid #333'}}>Pruned</span>
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
    gap: '20px',
    alignItems: 'center',
    padding: '15px',
    background: '#16213e',
    borderRadius: '8px',
    marginBottom: '20px',
    flexWrap: 'wrap',
  },
  control: {
    display: 'flex',
    flexDirection: 'column',
    gap: '5px',
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
  visualization: {
    display: 'flex',
    gap: '20px',
    alignItems: 'flex-start',
  },
  matrixContainer: {
    background: '#0f3460',
    borderRadius: '8px',
    padding: '10px',
  },
  svg: {
    display: 'block',
  },
  plotContainer: {
    flex: 1,
    background: '#0f3460',
    borderRadius: '8px',
    padding: '10px',
  },
  explanation: {
    marginTop: '20px',
    padding: '15px',
    background: '#16213e',
    borderRadius: '8px',
    lineHeight: 1.6,
  },
  legend: {
    display: 'flex',
    gap: '15px',
    marginTop: '15px',
  },
  legendItem: {
    padding: '5px 10px',
    borderRadius: '4px',
    fontSize: '12px',
  },
}

export default SparsityEvolution
