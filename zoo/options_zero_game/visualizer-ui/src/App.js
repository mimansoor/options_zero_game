// zoo/options_zero_game/visualizer-ui/src/App.js
// <<< DEFINITIVE, PROFESSIONAL VISUALIZER VERSION >>>

import React, { useState, useEffect } from 'react';
import './App.css';

// --- Charting Library Imports ---
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler, ScatterController
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation'; // The plugin for vertical lines

// Register all the necessary components for Chart.js
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, ScatterController, Title, Tooltip, Legend, Filler, annotationPlugin);

// ===================================================================================
//                            CHILD COMPONENTS
// ===================================================================================

function MetricsDashboard({ stepData }) {
  const info = stepData.info;
  if (!info) return null;

  const pnlColor = info.eval_episode_return > 0 ? '#4CAF50' : info.eval_episode_return < 0 ? '#F44336' : 'white';
  const lastChangeColor = info.last_price_change_pct > 0 ? '#4CAF50' : info.last_price_change_pct < 0 ? '#F44336' : 'white';

  // --- THE FIX for "vs Day 0" ---
  // 1. Calculate PNL percentage change, not price percentage change.
  const pnlChangePct = info.initial_cash ? (info.eval_episode_return / info.initial_cash) * 100 : 0;
  const pnlChangeColor = pnlChangePct > 0 ? '#4CAF50' : pnlChangePct < 0 ? '#F44336' : 'white';
  // --- END OF FIX ---

  return (
    <div className="metrics-dashboard">
      <div className="metric-item"> <h2>Market Regime</h2> <p style={{ color: '#2196F3' }}>{(info.market_regime || 'N/A').replace("Historical: ", "")}</p> </div>
      <div className="metric-item"> <h2>Day</h2> <p>{stepData.day}</p> </div>
      <div className="metric-item">
        <h2>EOD Price</h2>
        <p>${info.price ? info.price.toFixed(2) : '0.00'}</p>
        <p style={{ fontSize: '0.8em', color: lastChangeColor }}>
          {/* This now correctly displays the value calculated in the backend */}
          {info.last_price_change_pct ? info.last_price_change_pct.toFixed(2) : '0.00'}% vs last step
        </p>
      </div>
      <div className="metric-item">
        <h2>EOD Total PnL</h2>
        <p style={{ color: pnlColor }}>${info.eval_episode_return ? info.eval_episode_return.toFixed(2) : '0.00'}</p>
        <p style={{ fontSize: '0.8em', color: pnlChangeColor }}>
          {/* This now correctly displays the PnL percentage change */}
          {pnlChangePct.toFixed(2)}% vs Day 0
        </p>
      </div>
      <div className="metric-item"> <h2>Action Taken</h2> <p style={{ textTransform: 'capitalize' }}> {(info.executed_action_name || 'N/A').replace(/_/g, ' ')} </p> </div>
      <div className="metric-item"> <h2>Directional Bias</h2> <p style={{color: '#FFC107'}}>{info.directional_bias || 'N/A'}</p> </div>
      <div className="metric-item"> <h2>Volatility Bias</h2> <p style={{color: '#03A9F4'}}>{info.volatility_bias || 'N/A'}</p> </div>
    </div>
  );
}

function ActivePositions({ portfolio }) {
    if (!portfolio || portfolio.length === 0) return <p className="empty-message">Portfolio is empty.</p>;
    return (
        <table className="info-table">
            <thead><tr><th>Type</th><th>Direction</th><th>Strike</th><th>Entry Prem.</th><th>Current Prem.</th><th>Live PnL</th><th>DTE</th></tr></thead>
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
    if (!closedTrades || closedTrades.length === 0) return <p className="empty-message">No trades closed yet.</p>;
    return (
        <table className="info-table">
            <thead><tr><th>Position</th><th>Strike</th><th>Entry/Exit Day</th><th>Entry/Exit Prem.</th><th>Realized P&L</th></tr></thead>
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
    const { realized_pnl, unrealized_pnl, verified_total_pnl } = verificationData;
    const realizedColor = realized_pnl > 0 ? '#4CAF50' : realized_pnl < 0 ? '#F44336' : 'white';
    const unrealizedColor = unrealized_pnl > 0 ? '#4CAF50' : unrealized_pnl < 0 ? '#F44336' : 'white';
    const totalColor = verified_total_pnl > 0 ? '#4CAF50' : verified_total_pnl < 0 ? '#F44336' : 'white';
    return (
        <div className="pnl-verification">
            <div><span>Sum of Realized P&L:</span> <span style={{ color: realizedColor }}>${realized_pnl.toFixed(2)}</span></div>
            <div><span>Sum of Unrealized P&L:</span> <span style={{ color: unrealizedColor }}>${unrealized_pnl.toFixed(2)}</span></div>
            <div className="total"><span>Verified Total P&L:</span> <span style={{ color: totalColor }}>${verified_total_pnl.toFixed(2)}</span></div>
        </div>
    );
}

function PayoffDiagram({ payoffData }) {
    if (!payoffData || !payoffData.expiry_pnl || payoffData.expiry_pnl.length === 0) {
        return <p className="empty-message">Portfolio is empty.</p>;
    }

    const { expiry_pnl, current_pnl, spot_price, sigma_levels } = payoffData;

    const data = {
        labels: expiry_pnl.map(d => d.price),
        datasets: [
            { label: 'P&L at Expiry', data: expiry_pnl.map(d => d.pnl), borderColor: '#E91E63', tension: 0.1, pointRadius: 0, fill: true,
                backgroundColor: (context) => {
                    const chart = context.chart; const {ctx, chartArea} = chart; if (!chartArea) return null;
                    const zero = chart.scales.y.getPixelForValue(0); if (zero < chartArea.top || zero > chartArea.bottom) { return Math.max(...expiry_pnl.map(d => d.pnl)) > 0 ? 'rgba(76, 175, 80, 0.4)' : 'rgba(244, 67, 54, 0.4)'; }
                    const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom); const topPixel = chartArea.top; const bottomPixel = chartArea.bottom;
                    const zeroPoint = (zero - topPixel) / (bottomPixel - topPixel);
                    gradient.addColorStop(0, 'rgba(76, 175, 80, 0.5)'); gradient.addColorStop(zeroPoint, 'rgba(76, 175, 80, 0.5)');
                    gradient.addColorStop(zeroPoint, 'rgba(244, 67, 54, 0.5)'); gradient.addColorStop(1, 'rgba(244, 67, 54, 0.5)');
                    return gradient;
                },
            },
            { label: 'Mark-to-Market P&L (T+0)', data: current_pnl.map(d => d.pnl), borderColor: '#2196F3', borderDash: [5, 5], tension: 0.4, pointRadius: 0, }
        ],
    };
    
    const allPnlValues = [...expiry_pnl.map(d => d.pnl), ...current_pnl.map(d => d.pnl)];
    const minY = Math.min(...allPnlValues); const maxY = Math.max(...allPnlValues); const padding = (maxY - minY) * 0.1;

    const options = {
        responsive: true, maintainAspectRatio: false,
        scales: { x: { type: 'linear', ticks: { color: 'white' } }, y: { ticks: { color: 'white' }, min: minY - padding, max: maxY + padding, grid: { color: (c) => c.tick.value === 0 ? 'rgba(255, 255, 255, 0.8)' : 'rgba(255, 255, 255, 0.2)', lineWidth: (c) => c.tick.value === 0 ? 2 : 1, } } },
        plugins: {
            legend: { labels: { color: 'white' }, align: 'end', position: 'bottom' },
            annotation: {
                annotations: {
                    spotLine: { type: 'line', xMin: spot_price, xMax: spot_price, borderColor: '#4CAF50', borderWidth: 2, label: { content: `Spot: ${spot_price.toFixed(2)}`, enabled: true, position: 'start', font: {size: 10} } },
                    plusOneSigma: { type: 'line', xMin: sigma_levels.plus_one, xMax: sigma_levels.plus_one, borderColor: 'rgba(255, 255, 255, 0.3)', borderWidth: 1, borderDash: [6, 6], label: { content: '+1σ', enabled: true, position: 'start', font: { size: 10 } } },
                    minusOneSigma: { type: 'line', xMin: sigma_levels.minus_one, xMax: sigma_levels.minus_one, borderColor: 'rgba(255, 255, 255, 0.3)', borderWidth: 1, borderDash: [6, 6], label: { content: '-1σ', enabled: true, position: 'start', font: { size: 10 } } },
                }
            }
        }
    };
    return <div className="chart-container"><Line options={options} data={data} /></div>;
}

// --- NEW, UPGRADED AGENT BEHAVIOR CHART ---
function AgentBehaviorChart({ episodeHistory, historicalContext }) {
  if (!episodeHistory || episodeHistory.length === 0) return null;

  // Combine the historical data with the current episode data for a continuous series
  const combinedPriceData = [...(historicalContext || []), ...episodeHistory.map(step => step.info.price)];
  
  // Create labels that show negative steps for the historical context
  const historyLength = historicalContext ? historicalContext.length : 0;
  const labels = Array.from({ length: combinedPriceData.length }, (_, i) => i - historyLength);

  const data = {
    labels: labels,
    datasets: [
      {
        label: 'Index Price',
        data: combinedPriceData,
        // --- NEW: Dynamic styling for the line ---
        // The historical part will be a lighter, thinner line.
        segment: {
            borderColor: (ctx) => ctx.p0DataIndex < historyLength ? 'rgba(3, 169, 244, 0.5)' : '#03A9F4',
            borderWidth: (ctx) => ctx.p0DataIndex < historyLength ? 1 : 2,
        },
        yAxisID: 'y',
        tension: 0.1,
        pointRadius: (context) => context.dataIndex < historyLength ? 0 : 2, // No dots on historical part
      },
    ],
  };

  const options = { responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { color: 'white' } }, y: { ticks: { color: 'white' } } }, plugins: { legend: { display: false } } };
  return <div className="chart-container"><Line options={options} data={data} /></div>;
}

function PortfolioRiskDashboard({ portfolioStats }) {
  if (!portfolioStats) return <p className="empty-message">Awaiting data...</p>;

  const { delta, gamma, theta, vega, max_profit, max_loss, rr_ratio, prob_profit } = portfolioStats;

  const deltaColor = delta > 0 ? '#4CAF50' : delta < 0 ? '#F44336' : 'white';
  const gammaColor = gamma > 0 ? '#4CAF50' : gamma < 0 ? '#F44336' : 'white';
  const thetaColor = theta > 0 ? '#4CAF50' : theta < 0 ? '#F44336' : 'white';
  const vegaColor = vega > 0 ? '#4CAF50' : vega < 0 ? '#F44336' : 'white';

  // --- THE FIX: A new helper function to format the R:R Ratio ---
  const formatRRRatio = (ratio) => {
    // Handle cases where max loss is zero (infinite reward potential)
    if (!isFinite(ratio)) {
      return '1 : ∞';
    }
    // Handle cases where there's no profit potential
    if (ratio <= 0) {
      return '1 : 0.00';
    }
    // The standard, correct formatting
    return `1 : ${ratio.toFixed(2)}`;
  };

  return (
    <div className="risk-dashboard">
      <div className="risk-item"><span>Portfolio Delta:</span> <p style={{ color: deltaColor }}>{delta.toFixed(2)}</p></div>
      <div className="risk-item"><span>Portfolio Gamma:</span> <p style={{ color: gammaColor }}>{gamma.toFixed(2)}</p></div>
      <div className="risk-item"><span>Portfolio Theta:</span> <p style={{ color: thetaColor }}>{theta.toFixed(2)}</p></div>
      <div className="risk-item"><span>Portfolio Vega:</span> <p style={{ color: vegaColor }}>{vega.toFixed(2)}</p></div>
      <div className="risk-item"><span>Max Profit:</span> <p style={{ color: '#4CAF50' }}>${max_profit.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</p></div>
      <div className="risk-item"><span>Max Loss:</span> <p style={{ color: '#F44336' }}>${max_loss.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</p></div>
      
      {/* --- Apply the new formatting here --- */}
      <div className="risk-item"><span>Risk/Reward Ratio:</span> <p>{formatRRRatio(rr_ratio)}</p></div>
      
      <div className="risk-item"><span>Prob. of Profit:</span> <p>{(prob_profit * 100).toFixed(2)}%</p></div>
    </div>
  );
}

// ===================================================================================
//                                MAIN APP COMPONENT
// ===================================================================================

function App() {
  const [replayData, setReplayData] = useState([]);
  const [historicalContext, setHistoricalContext] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    fetch(`/replay_log.json?t=${new Date().getTime()}`)
      .then(response => response.json())
      .then(logObject => { // The fetched object now has two keys
        setHistoricalContext(logObject.historical_context || []);
        setReplayData(logObject.episode_data || []);
      })
      .catch(error => console.error('Error loading replay_log.json:', error));
  }, []);

  const goToStep = (step) => setCurrentStep(Math.max(0, Math.min(replayData.length - 1, step)));
  
  const stepData = replayData[currentStep];
  // The history passed to the chart should ONLY be the episode data
  const episodeHistory = replayData.slice(0, currentStep + 1);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Options-Zero-Game Replayer</h1>
        
        {replayData.length > 0 && stepData ? (
          <div className="main-content">
            <div className="top-bar">
              {/* Display the total steps from the info dict, not replayData.length */}
              <h2>Step: {stepData.step} / {stepData.info.total_steps_in_episode} (Day: {stepData.day})</h2>
              <div className="navigation-buttons">
                  <button onClick={() => goToStep(currentStep - 1)} disabled={currentStep === 0}>Prev S</button>
                  <button onClick={() => goToStep(currentStep + 1)} disabled={currentStep === replayData.length - 1}>Next S</button>
              </div>
              <input type="range" min="0" max={replayData.length - 1} value={currentStep} onChange={(e) => setCurrentStep(Number(e.target.value))} className="slider" />
            </div>
            
            <MetricsDashboard stepData={stepData} />

            {/* --- NEW, SUPERIOR LAYOUT --- */}
            
            {/* 1. A container for the two main charts, arranged side-by-side */}
            <div className="charts-container">
              <div className="card chart-card">
                <h3>Agent Behavior</h3>
                <AgentBehaviorChart
                  episodeHistory={episodeHistory}
                  historicalContext={historicalContext}
                />
              </div>
              <div className="card chart-card">
                 <h3>Portfolio P&L Diagram</h3>
                 <PayoffDiagram payoffData={stepData.info.payoff_data} />
              </div>
            </div>

            {/* 2. A container for the info panels below the charts */}
            <div className="info-panels-container">
               <div className="card">
                  <h3>Active Positions</h3>
                  <ActivePositions portfolio={stepData.portfolio} />
               </div>
                 <div className="card">
                    <h3>Portfolio Risk Profile</h3>
                    <PortfolioRiskDashboard portfolioStats={stepData.info.portfolio_stats} />
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
        ) : (
          <p><i>Loading Replay Data... (If this persists, check console for errors and ensure replay_log.json exists in the build folder)</i></p>
        )}
      </header>
    </div>
  );
}

export default App;

