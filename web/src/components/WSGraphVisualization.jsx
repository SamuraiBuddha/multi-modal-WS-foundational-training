import React, { useEffect, useRef, useState } from 'react'
import * as d3 from 'd3'

/**
 * Interactive Watts-Strogatz graph visualization.
 * Demonstrates how rewiring probability (beta) affects network topology.
 */
function WSGraphVisualization() {
  const svgRef = useRef(null)
  const [n, setN] = useState(20)
  const [k, setK] = useState(4)
  const [beta, setBeta] = useState(0.3)
  const [graphData, setGraphData] = useState(null)

  // Generate Watts-Strogatz graph
  const generateWSGraph = (n, k, beta) => {
    const nodes = Array.from({ length: n }, (_, i) => ({
      id: i,
      x: 0,
      y: 0,
    }))

    // Create ring lattice edges
    const edges = []
    const edgeSet = new Set()

    for (let i = 0; i < n; i++) {
      for (let j = 1; j <= k / 2; j++) {
        const target = (i + j) % n
        const edgeKey = [Math.min(i, target), Math.max(i, target)].join('-')
        if (!edgeSet.has(edgeKey)) {
          edgeSet.add(edgeKey)
          edges.push({
            source: i,
            target: target,
            original: true,
          })
        }
      }
    }

    // Rewire with probability beta
    const rewiredEdges = edges.map(edge => {
      if (Math.random() < beta) {
        // Rewire to random node
        let newTarget
        do {
          newTarget = Math.floor(Math.random() * n)
        } while (newTarget === edge.source || newTarget === edge.target)

        return {
          source: edge.source,
          target: newTarget,
          original: false,
          rewired: true,
        }
      }
      return { ...edge, rewired: false }
    })

    return { nodes, edges: rewiredEdges }
  }

  // Update graph when parameters change
  useEffect(() => {
    const data = generateWSGraph(n, k, beta)
    setGraphData(data)
  }, [n, k, beta])

  // D3 visualization
  useEffect(() => {
    if (!graphData || !svgRef.current) return

    const svg = d3.select(svgRef.current)
    const width = 600
    const height = 500
    const centerX = width / 2
    const centerY = height / 2
    const radius = Math.min(width, height) / 2 - 50

    svg.selectAll('*').remove()

    // Position nodes in a circle
    const angleStep = (2 * Math.PI) / graphData.nodes.length
    graphData.nodes.forEach((node, i) => {
      node.x = centerX + radius * Math.cos(i * angleStep - Math.PI / 2)
      node.y = centerY + radius * Math.sin(i * angleStep - Math.PI / 2)
    })

    // Create SVG group
    const g = svg.append('g')

    // Draw edges
    g.selectAll('line')
      .data(graphData.edges)
      .enter()
      .append('line')
      .attr('x1', d => graphData.nodes[d.source].x)
      .attr('y1', d => graphData.nodes[d.source].y)
      .attr('x2', d => graphData.nodes[d.target].x)
      .attr('y2', d => graphData.nodes[d.target].y)
      .attr('stroke', d => d.rewired ? '#e94560' : '#4a5568')
      .attr('stroke-width', d => d.rewired ? 2 : 1)
      .attr('opacity', d => d.rewired ? 0.9 : 0.4)

    // Draw nodes
    g.selectAll('circle')
      .data(graphData.nodes)
      .enter()
      .append('circle')
      .attr('cx', d => d.x)
      .attr('cy', d => d.y)
      .attr('r', 8)
      .attr('fill', '#667eea')
      .attr('stroke', '#fff')
      .attr('stroke-width', 2)

    // Node labels
    g.selectAll('text')
      .data(graphData.nodes)
      .enter()
      .append('text')
      .attr('x', d => d.x)
      .attr('y', d => d.y + 4)
      .attr('text-anchor', 'middle')
      .attr('fill', '#fff')
      .attr('font-size', '10px')
      .text(d => d.id)

  }, [graphData])

  // Calculate metrics
  const calculateMetrics = () => {
    if (!graphData) return { clustering: 0, avgPath: 0 }

    // Simple clustering coefficient approximation
    const n = graphData.nodes.length
    const adjMatrix = Array(n).fill(null).map(() => Array(n).fill(0))

    graphData.edges.forEach(e => {
      adjMatrix[e.source][e.target] = 1
      adjMatrix[e.target][e.source] = 1
    })

    let totalTriangles = 0
    let totalTriplets = 0

    for (let i = 0; i < n; i++) {
      const neighbors = []
      for (let j = 0; j < n; j++) {
        if (adjMatrix[i][j]) neighbors.push(j)
      }

      const ki = neighbors.length
      if (ki >= 2) {
        totalTriplets += (ki * (ki - 1)) / 2
        for (let a = 0; a < neighbors.length; a++) {
          for (let b = a + 1; b < neighbors.length; b++) {
            if (adjMatrix[neighbors[a]][neighbors[b]]) {
              totalTriangles++
            }
          }
        }
      }
    }

    const clustering = totalTriplets > 0 ? totalTriangles / totalTriplets : 0
    const rewiredCount = graphData.edges.filter(e => e.rewired).length

    return {
      clustering: clustering.toFixed(3),
      rewiredEdges: rewiredCount,
      totalEdges: graphData.edges.length,
    }
  }

  const metrics = calculateMetrics()

  return (
    <div style={styles.container}>
      <div style={styles.controls}>
        <div style={styles.control}>
          <label>Nodes (n): {n}</label>
          <input
            type="range"
            min="10"
            max="40"
            value={n}
            onChange={e => setN(parseInt(e.target.value))}
            style={styles.slider}
          />
        </div>
        <div style={styles.control}>
          <label>Neighbors (k): {k}</label>
          <input
            type="range"
            min="2"
            max="8"
            step="2"
            value={k}
            onChange={e => setK(parseInt(e.target.value))}
            style={styles.slider}
          />
        </div>
        <div style={styles.control}>
          <label>Beta: {beta.toFixed(2)}</label>
          <input
            type="range"
            min="0"
            max="1"
            step="0.01"
            value={beta}
            onChange={e => setBeta(parseFloat(e.target.value))}
            style={styles.slider}
          />
        </div>
        <button
          style={styles.button}
          onClick={() => setGraphData(generateWSGraph(n, k, beta))}
        >
          Regenerate
        </button>
      </div>

      <div style={styles.visualization}>
        <svg ref={svgRef} width={600} height={500} style={styles.svg} />

        <div style={styles.metrics}>
          <h3>Network Metrics</h3>
          <p>Clustering Coefficient: <strong>{metrics.clustering}</strong></p>
          <p>Rewired Edges: <strong>{metrics.rewiredEdges}</strong> / {metrics.totalEdges}</p>
          <p style={styles.legend}>
            <span style={{...styles.legendItem, background: '#4a5568'}}>Original</span>
            <span style={{...styles.legendItem, background: '#e94560'}}>Rewired</span>
          </p>
        </div>
      </div>

      <div style={styles.explanation}>
        <h3>Watts-Strogatz Model</h3>
        <p>
          The WS model creates small-world networks by starting with a regular ring lattice
          and rewiring edges with probability <strong>beta</strong>.
        </p>
        <ul>
          <li><strong>beta = 0:</strong> Regular lattice (high clustering, long paths)</li>
          <li><strong>beta = 1:</strong> Random graph (low clustering, short paths)</li>
          <li><strong>0.01 &lt; beta &lt; 0.3:</strong> Small-world regime (high clustering AND short paths)</li>
        </ul>
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
  slider: {
    width: '150px',
  },
  button: {
    padding: '10px 20px',
    background: '#667eea',
    border: 'none',
    borderRadius: '5px',
    color: '#fff',
    cursor: 'pointer',
  },
  visualization: {
    display: 'flex',
    gap: '20px',
    alignItems: 'flex-start',
  },
  svg: {
    background: '#0f3460',
    borderRadius: '8px',
  },
  metrics: {
    padding: '15px',
    background: '#16213e',
    borderRadius: '8px',
    minWidth: '200px',
  },
  legend: {
    display: 'flex',
    gap: '10px',
    marginTop: '10px',
  },
  legendItem: {
    padding: '3px 8px',
    borderRadius: '3px',
    fontSize: '12px',
  },
  explanation: {
    marginTop: '20px',
    padding: '15px',
    background: '#16213e',
    borderRadius: '8px',
    lineHeight: 1.6,
  },
}

export default WSGraphVisualization
