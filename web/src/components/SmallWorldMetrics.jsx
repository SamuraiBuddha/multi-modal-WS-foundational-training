import React, { useState, useEffect } from 'react'
import Plot from 'react-plotly.js'

/**
 * Interactive visualization of small-world metrics vs beta.
 * Shows how clustering and path length change with rewiring probability.
 */
function SmallWorldMetrics() {
  const [n, setN] = useState(50)
  const [k, setK] = useState(4)
  const [metricsData, setMetricsData] = useState(null)
  const [loading, setLoading] = useState(false)

  // Generate WS graph and calculate metrics
  const calculateWSMetrics = (n, k, beta) => {
    // Create adjacency matrix
    const adj = Array(n).fill(null).map(() => Array(n).fill(0))

    // Ring lattice
    for (let i = 0; i < n; i++) {
      for (let j = 1; j <= k / 2; j++) {
        const target = (i + j) % n
        adj[i][target] = 1
        adj[target][i] = 1
      }
    }

    // Rewire
    for (let i = 0; i < n; i++) {
      for (let j = 1; j <= k / 2; j++) {
        const target = (i + j) % n
        if (Math.random() < beta && adj[i][target]) {
          // Remove original edge
          adj[i][target] = 0
          adj[target][i] = 0

          // Add random edge
          let newTarget
          let attempts = 0
          do {
            newTarget = Math.floor(Math.random() * n)
            attempts++
          } while ((newTarget === i || adj[i][newTarget]) && attempts < 100)

          if (attempts < 100) {
            adj[i][newTarget] = 1
            adj[newTarget][i] = 1
          }
        }
      }
    }

    // Calculate clustering coefficient
    let totalTriangles = 0
    let totalTriplets = 0

    for (let i = 0; i < n; i++) {
      const neighbors = []
      for (let j = 0; j < n; j++) {
        if (adj[i][j]) neighbors.push(j)
      }

      const ki = neighbors.length
      if (ki >= 2) {
        totalTriplets += (ki * (ki - 1)) / 2
        for (let a = 0; a < neighbors.length; a++) {
          for (let b = a + 1; b < neighbors.length; b++) {
            if (adj[neighbors[a]][neighbors[b]]) {
              totalTriangles++
            }
          }
        }
      }
    }

    const clustering = totalTriplets > 0 ? totalTriangles / totalTriplets : 0

    // Calculate average path length (BFS)
    let totalPath = 0
    let pathCount = 0

    for (let start = 0; start < n; start++) {
      const dist = Array(n).fill(Infinity)
      dist[start] = 0
      const queue = [start]

      while (queue.length > 0) {
        const current = queue.shift()
        for (let neighbor = 0; neighbor < n; neighbor++) {
          if (adj[current][neighbor] && dist[neighbor] === Infinity) {
            dist[neighbor] = dist[current] + 1
            queue.push(neighbor)
          }
        }
      }

      for (let end = start + 1; end < n; end++) {
        if (dist[end] !== Infinity) {
          totalPath += dist[end]
          pathCount++
        }
      }
    }

    const avgPathLength = pathCount > 0 ? totalPath / pathCount : 0

    return { clustering, avgPathLength }
  }

  // Compute metrics for range of beta values
  const computeAllMetrics = () => {
    setLoading(true)

    setTimeout(() => {
      const betas = [0, 0.001, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
      const nTrials = 3

      const results = betas.map(beta => {
        let cSum = 0
        let lSum = 0

        for (let t = 0; t < nTrials; t++) {
          const metrics = calculateWSMetrics(n, k, beta)
          cSum += metrics.clustering
          lSum += metrics.avgPathLength
        }

        return {
          beta,
          clustering: cSum / nTrials,
          pathLength: lSum / nTrials,
        }
      })

      // Normalize by beta=0 values
      const c0 = results[0].clustering || 1
      const l0 = results[0].pathLength || 1

      const normalizedData = results.map(r => ({
        ...r,
        cNorm: r.clustering / c0,
        lNorm: r.pathLength / l0,
      }))

      setMetricsData(normalizedData)
      setLoading(false)
    }, 100)
  }

  useEffect(() => {
    computeAllMetrics()
  }, [n, k])

  return (
    <div style={styles.container}>
      <div style={styles.controls}>
        <div style={styles.control}>
          <label>Nodes (n): {n}</label>
          <input
            type="range"
            min="20"
            max="100"
            step="10"
            value={n}
            onChange={e => setN(parseInt(e.target.value))}
          />
        </div>
        <div style={styles.control}>
          <label>Neighbors (k): {k}</label>
          <input
            type="range"
            min="2"
            max="10"
            step="2"
            value={k}
            onChange={e => setK(parseInt(e.target.value))}
          />
        </div>
        <button style={styles.button} onClick={computeAllMetrics}>
          {loading ? 'Computing...' : 'Recompute'}
        </button>
      </div>

      {metricsData && (
        <div style={styles.plotContainer}>
          <Plot
            data={[
              {
                x: metricsData.map(d => d.beta),
                y: metricsData.map(d => d.cNorm),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'C(beta) / C(0)',
                line: { color: '#667eea', width: 2 },
                marker: { size: 8 },
              },
              {
                x: metricsData.map(d => d.beta),
                y: metricsData.map(d => d.lNorm),
                type: 'scatter',
                mode: 'lines+markers',
                name: 'L(beta) / L(0)',
                line: { color: '#e94560', width: 2 },
                marker: { size: 8 },
              },
            ]}
            layout={{
              title: 'Small-World Metrics vs Rewiring Probability',
              xaxis: {
                title: 'Rewiring Probability (beta)',
                type: 'log',
                range: [-3.5, 0.1],
                gridcolor: '#333',
              },
              yaxis: {
                title: 'Normalized Value',
                range: [0, 1.1],
                gridcolor: '#333',
              },
              paper_bgcolor: '#0f3460',
              plot_bgcolor: '#0f3460',
              font: { color: '#fff' },
              legend: { x: 0.7, y: 0.95 },
              shapes: [
                {
                  type: 'rect',
                  xref: 'x',
                  yref: 'paper',
                  x0: 0.01,
                  x1: 0.3,
                  y0: 0,
                  y1: 1,
                  fillcolor: '#667eea',
                  opacity: 0.1,
                  line: { width: 0 },
                },
              ],
              annotations: [
                {
                  x: Math.log10(0.05),
                  y: 0.5,
                  xref: 'x',
                  yref: 'y',
                  text: 'Small-World Regime',
                  showarrow: false,
                  font: { color: '#667eea' },
                },
              ],
            }}
            style={{ width: '100%', height: '450px' }}
            config={{ displayModeBar: false }}
          />
        </div>
      )}

      <div style={styles.explanation}>
        <h3>Understanding the Plot</h3>
        <p>
          The key insight of the Watts-Strogatz model is visible in this plot:
        </p>
        <ul>
          <li>
            <strong>Clustering (blue)</strong> remains high until beta [->] 0.1
          </li>
          <li>
            <strong>Path length (red)</strong> drops rapidly even at small beta values
          </li>
          <li>
            The <strong>small-world regime</strong> (shaded) has both high clustering
            AND short paths - the best of both worlds!
          </li>
        </ul>
        <p>
          This is why WS topology is ideal for neural networks: efficient information
          flow (short paths) with local structure preservation (high clustering).
        </p>
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
  },
  plotContainer: {
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
}

export default SmallWorldMetrics
