# Web Visualizations

Interactive visualizations for the Multi-Modal WS Foundational Training curriculum.

## Features

- **WS Network Visualization**: Interactive Watts-Strogatz graph with adjustable parameters
- **Small-World Metrics**: Plot of clustering and path length vs rewiring probability
- **Training Dashboard**: Simulated training progress with loss/accuracy curves
- **Sparsity Evolution**: SET algorithm visualization showing topology rewiring

## Setup

```bash
cd web
npm install
npm run dev
```

Then open http://localhost:5173 in your browser.

## Build for Production

```bash
npm run build
```

The built files will be in the `dist` directory.

## Technologies

- React 18
- D3.js for network visualizations
- Plotly.js for charts
- Vite for bundling
