// zoo/options_zero_game/visualizer-ui/src/App.js
// <<< DEFINITIVE, FEATURE-COMPLETE VISUALIZER >>>

import React, { useState, useEffect } from 'react';
import './App.css';

// --- Charting Library Imports ---
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  PointElement,
  LineElement,
  Title,
  Tooltip,
  Legend,
  Filler,
  ScatterController // The critical import to fix the scatter error
} from 'chart.js';

// Register all the necessary components for Chart.js
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, ScatterController, Title, Tooltip, Legend, Filler);

// ===================================================================================
//                            CHILD COMPONENTS
// (Defined here for a single-file, copy-paste solution)
// ===================================================================================

function MetricsDashboard({ stepData }) {
  const info = stepData.info;
  if (!info) return null;

  const pnlColor = info.eval_episode_return > 0 ? '#4CAF50' : info.eval_episode_return < 0 ? '#F44336' : 'white';
  const cumulativeChange = info.price && info.start_price ? ((info.price / info.start_price) - 1) * 100 : 0;
  const cumulativeChangeColor = cumulativeChange > 0 ? '#4CAF50' : cumulativeChange < 0 ? '#F44336' : 'white';
  const lastChangeColor = info.last_price_change_pct > 0 ? '#4CAF50' : info.last_price_change_pct < 0 ? '#F44336' : 'white';

  return (
    <div className="metrics-dashboard">
      <div className="metric-item">
        <h2>Market Regime</h2>
        <p style={{ color: '#2196F3', fontWeight: 'bold' }}>{(info.market_regime || 'N/A').replace("Historical: ", "")}</p>
      </div>
      <div className="metric-item">
        <h2>Day</h2>
        <p>{stepData.day}</p>
      </div>
      <div className="metric-item">
        <h2>EOD Price</h2>
        <p>${info.price ? info.price.toFixed(2) : '0.00'}</p>
        <p style={{ fontSize: '0.8em', color: lastChangeColor }}>
          {info.last_price_change_pct ? info.last_price_change_pct.toFixed(2) : '0.00'}% vs last step
        </p>
      </div>
      <div className="metric-item">
        <h2>EOD Total PnL</h2>
        <p style={{ color: pnlColor }}>${info.eval_episode_return ? info.eval_episode_return.toFixed(2) : '0.00'}</p>
        <p style={{ fontSize: '0.8em', color: cumulativeChangeColor, fontWeight: 'bold' }}>
          {cumulativeChange.toFixed(2)}% vs Day 0
        </p>
      </div>
      <div className="metric-item">
        <h2>Action Taken</h2>
        <p style={{ fontSize: '1.1em', color: '#ddd', textTransform: 'capitalize' }}>
          {(info.executed_action_name || 'N/A').replace(/_/g, ' ')}
        </p>
      </div>
       <div className="metric-item">
        <h2>Directional Bias</h2>
        <p style={{color: '#FFC107'}}>{info.directional_bias || 'N/A'}</p>
      </div>
      <div className="metric-item">
        <h2>Volatility Bias</h2>
        <p style={{color: '#03A9F4'}}>{info.volatility_bias || 'N/A'}</p>
      </div>
    </div>
  );
}

function ActivePositions({ portfolio }) {
  if (!portfolio || portfolio.length === 0) {
    return <p className="empty-message">Portfolio is empty.</p>;
  }
  return (
    <table className="portfolio-table">
      <thead>
        <tr>
          <th>Type</th><th>Direction</th><th>Strike</th><th>Entry Prem.</th><th>Current Prem.</th><th>Live PnL</th><th>DTE</th>
        </tr>
      </thead>
      <tbody>
        {portfolio.map((pos, index) => {
          const pnlColor = pos.live_pnl > 0 ? '#4CAF50' : pos.live_pnl < 0 ? '#F44336' : 'white';
          return (
            <tr key={index} className={pos.direction.toLowerCase()}>
              <td>{pos.type.toUpperCase()}</td><td>{pos.direction.toUpperCase()}</td><td>${pos.strike_price.toFixed(2)}</td><td>${pos.entry_premium.toFixed(2)}</td>
              <td>${pos.current_premium.toFixed(2)}</td><td style={{ color: pnlColor }}>${pos.live_pnl.toFixed(2)}</td><td>{pos.days_to_expiry.toFixed(2)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function ClosedTradesLog({ closedTrades }) {
    if (!closedTrades || closedTrades.length === 0) {
    return <p className="empty-message">No trades closed yet.</p>;
  }
  return (
    <table className="info-table">
      <thead>
        <tr>
          <th>Position</th><th>Strike</th><th>Entry/Exit Day</th><th>Entry/Exit Prem.</th><th>Realized P&L</th>
        </tr>
      </thead>
      <tbody>
        {closedTrades.map((trade, index) => {
          const pnlColor = trade.realized_pnl > 0 ? '#4CAF50' : trade.realized_pnl < 0 ? '#F44336' : 'white';
          return (
            <tr key={index}>
              <td>{trade.position}</td><td>${trade.strike.toFixed(2)}</td><td>{trade.entry_day} → {trade.exit_day}</td>
              <td>${trade.entry_prem.toFixed(2)} → ${trade.exit_prem ? trade.exit_prem.toFixed(2) : 'N/A'}</td>
              <td style={{ color: pnlColor }}>${trade.realized_pnl.toFixed(2)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function PnlVerification({ verificationData }) {
  if (!verificationData) return <p className="empty-message">Awaiting data...</p>;
  const realizedColor = verificationData.realized_pnl > 0 ? '#4CAF50' : verificationData.realized_pnl < 0 ? '#F44336' : 'white';
  const unrealizedColor = verificationData.unrealized_pnl > 0 ? '#4CAF50' : verificationData.unrealized_pnl < 0 ? '#F44336' : 'white';
  const totalColor = verificationData.verified_total_pnl > 0 ? '#4CAF50' : verificationData.verified_total_pnl < 0 ? '#F44336' : 'white';

  return (
     <div className="pnl-verification">
        <div><span>Sum of Realized P&L:</span> <span style={{ color: realizedColor }}>${verificationData.realized_pnl.toFixed(2)}</span></div>
        <div><span>Sum of Unrealized P&L:</span> <span style={{ color: unrealizedColor }}>${verificationData.unrealized_pnl.toFixed(2)}</span></div>
        <div className="total"><span>Verified Total P&L:</span> <span style={{ color: totalColor }}>${verificationData.verified_total_pnl.toFixed(2)}</span></div>
     </div>
  );
}

function AgentBehaviorChart({ history }) {
  const data = {
    labels: history.map(step => step.step),
    datasets: [ { label: 'Index Price', data: history.map(step => step.info.price), borderColor: '#03A9F4', backgroundColor: '#03A9F4', yAxisID: 'y', tension: 0.1, }, ],
  };
  const options = { responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { color: 'white' } }, y: { ticks: { color: 'white' } } }, plugins: { legend: { labels: { color: 'white' } } } };
  return <div className="chart-container"><Line options={options} data={data} /></div>;
}

function PayoffDiagram({ payoffData, currentPnl, currentPrice }) {
  if (!payoffData || !payoffData.expiry_pnl || payoffData.expiry_pnl.length === 0) {
    return <p className="empty-message">Portfolio is empty.</p>;
  }

  const data = {
    labels: payoffData.expiry_pnl.map(d => d.price),
    datasets: [
      {
        label: 'Total Portfolio P&L at Expiry',
        data: payoffData.expiry_pnl.map(d => d.pnl),
        borderColor: '#FFC107',
        tension: 0.1,
        fill: true,
        backgroundColor: (context) => {
          const chart = context.chart; const {ctx, chartArea} = chart; if (!chartArea) return null;
          const zero = chart.scales.y.getPixelForValue(0);
          if (zero < chartArea.top || zero > chartArea.bottom) {
              const maxPnl = Math.max(...payoffData.expiry_pnl.map(d => d.pnl));
              return maxPnl > 0 ? 'rgba(76, 175, 80, 0.4)' : 'rgba(244, 67, 54, 0.4)';
          }
          const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
          const topPixel = chartArea.top; const bottomPixel = chartArea.bottom; const totalHeight = bottomPixel - topPixel;
          const zeroPoint = (zero - topPixel) / totalHeight;
          gradient.addColorStop(0, 'rgba(76, 175, 80, 0.6)'); gradient.addColorStop(zeroPoint, 'rgba(76, 175, 80, 0.6)');
          gradient.addColorStop(zeroPoint, 'rgba(244, 67, 54, 0.6)'); gradient.addColorStop(1, 'rgba(244, 67, 54, 0.6)');
          return gradient;
        },
      },
       {
        label: 'Current Mark-to-Market P&L', data: [{ x: currentPrice, y: currentPnl }],
        borderColor: '#E91E63', backgroundColor: '#E91E63', pointRadius: 6, type: 'scatter',
      },
    ],
  };
  
  // --- THE NEW, ROBUST Y-AXIS ZOOM LOGIC ---

  // 1. Get the absolute min/max from the theoretical payoff curve.
  const pnlValues = payoffData.expiry_pnl.map(d => d.pnl);
  const minYFromCurve = Math.min(...pnlValues);
  const maxYFromCurve = Math.max(...pnlValues);

  // 2. Calculate the desired zoom level based on your rule (current PnL * 2).
  let yLimitFromPnl = Math.abs(currentPnl) * 2;
  
  // 3. Enforce a minimum sensible zoom level to prevent a collapsed chart.
  //    Let's set a minimum of $500, or a bit more than the highest premium.
  const minSensibleLimit = 500; 
  if (yLimitFromPnl < minSensibleLimit) {
    yLimitFromPnl = minSensibleLimit;
  }

  // 4. Determine the final axis limits by taking the LARGER of the two ranges.
  //    This ensures the axis is tight, but never clips the theoretical curve.
  const finalMaxY = Math.max(maxYFromCurve, yLimitFromPnl);
  const finalMinY = Math.min(minYFromCurve, -yLimitFromPnl);

  // 5. Add a small amount of padding for visual appeal.
  const padding = (finalMaxY - finalMinY) * 0.1;
  
  const options = {
    responsive: true,
    maintainAspectRatio: false,
    scales: {
      x: {
        type: 'linear', ticks: { color: 'white' },
        min: Math.round(currentPrice / 50) * 50 - 2000,
        max: Math.round(currentPrice / 50) * 50 + 2000,
      },
      y: {
        ticks: { color: 'white' },
        min: finalMinY - padding,
        max: finalMaxY + padding,
        grid: {
            color: (context) => context.tick.value === 0 ? 'rgba(255, 255, 255, 0.8)' : 'rgba(255, 255, 255, 0.2)',
            lineWidth: (context) => context.tick.value === 0 ? 2 : 1,
        }
      },
    },
    plugins: {
      legend: { labels: { color: 'white' } },
      tooltip: {
          callbacks: {
              label: (context) => `P&L: $${context.parsed.y.toFixed(2)} at Price $${context.parsed.x.toFixed(2)}`
          }
      }
    }
  };

  return <div className="chart-container"><Line options={options} data={data} /></div>;
}

// ===================================================================================
//                                MAIN APP COMPONENT
// ===================================================================================

function App() {
  const [replayData, setReplayData] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    fetch(`/replay_log.json?t=${new Date().getTime()}`)
      .then(response => response.json())
      .then(rawHistory => {
        setReplayData(rawHistory || []);
      })
      .catch(error => console.error('Error loading replay_log.json:', error));
  }, []);

  const handleSliderChange = (event) => {
    setCurrentStep(Number(event.target.value));
  };
  
  const goToStep = (step) => {
    setCurrentStep(Math.max(0, Math.min(replayData.length - 1, step)));
  };

  const stepData = replayData[currentStep];
  const episodeHistory = replayData.slice(0, currentStep + 1);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Options-Zero-Game Replayer</h1>
        
        {replayData.length > 0 && stepData ? (
          <div className="main-content">
            <div className="top-bar">
              <h2>Step: {stepData.step} / {replayData.length - 1} (Day: {stepData.day})</h2>
              <div className="navigation-buttons">
                  <button onClick={() => goToStep(currentStep - 1)} disabled={currentStep === 0}>Prev S</button>
                  <button onClick={() => goToStep(currentStep + 1)} disabled={currentStep === replayData.length - 1}>Next S</button>
              </div>
              <input type="range" min="0" max={replayData.length - 1} value={currentStep} onChange={handleSliderChange} className="slider" />
            </div>
            
            <MetricsDashboard stepData={stepData} />

            <div className="main-layout">
              <div className="left-panel">
                <div className="card">
                  <h3>Agent Behavior</h3>
                  <AgentBehaviorChart history={episodeHistory} />
                </div>
                <div className="card">
                   <h3>Portfolio P&L Diagram</h3>
                   <PayoffDiagram 
                      payoffData={stepData.info.payoff_data}
                      currentPnl={stepData.info.pnl_verification?.verified_total_pnl}
                      currentPrice={stepData.info.price}
                   />
                </div>
              </div>

              <div className="right-panel">
                 <div className="card">
                    <h3>Active Positions</h3>
                    <ActivePositions portfolio={stepData.portfolio} />
                 </div>
                 <div className="card">
                    <h3>Closed Trades Log</h3>
                    <ClosedTradesLog closedTrades={stepData.info.closed_trades_log} />
                 </div>
                 <div className="card">
                    <h3>P&L Verification</h3>
                    <PnlVerification verificationData={stepData.info.pnl_verification} />
                 </div>
              </div>
            </div>
          </div>
        ) : (
          <p><i>Loading Replay Data... (If this persists, check the console for errors and ensure replay_log.json exists in the build folder)</i></p>
        )}
      </header>
    </div>
  );
}

export default App;
