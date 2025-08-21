import React, { useState, useEffect, useMemo } from 'react';
import './App.css';

// --- Charting Library Imports ---
import { Line } from 'react-chartjs-2';
import {
  Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler
} from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler, annotationPlugin);

// ===================================================================================
//                            JAVASCRIPT RE-SIMULATOR
// ===================================================================================
const reSimulateStep = (rawStepData, deNormParams, envDefaults) => {
    if (!rawStepData) return null;
    const newStepData = JSON.parse(JSON.stringify(rawStepData));
    const priceRatio = deNormParams.startPrice / envDefaults.startPrice;
    const pnlRatio = deNormParams.lotSize / envDefaults.lotSize;
    newStepData.info.price *= priceRatio;
    if(newStepData.info.initial_cash) newStepData.info.initial_cash *= pnlRatio;
    let totalUnrealizedPnl = 0;
    if (newStepData.portfolio) {
        newStepData.portfolio.forEach(leg => {
            const originalOffset = (leg.strike_price - envDefaults.startPrice) / envDefaults.strikeDistance;
            const rawNewStrike = deNormParams.startPrice + (originalOffset * deNormParams.strikeDistance);
            leg.strike_price = Math.round(rawNewStrike / deNormParams.strikeDistance) * deNormParams.strikeDistance;
            leg.entry_premium *= priceRatio;
            leg.current_premium *= priceRatio;
            leg.live_pnl *= pnlRatio;
            totalUnrealizedPnl += leg.live_pnl;
        });
    }
    let totalRealizedPnl = 0;
    if (newStepData.info.closed_trades_log && newStepData.info.closed_trades_log.length > 0) {
        newStepData.info.closed_trades_log.forEach(trade => {
            const originalOffset = (trade.strike - envDefaults.startPrice) / envDefaults.strikeDistance;
            trade.strike = Math.round((deNormParams.startPrice + (originalOffset * deNormParams.strikeDistance)) / deNormParams.strikeDistance) * deNormParams.strikeDistance;
            trade.entry_prem *= priceRatio;
            trade.exit_prem *= priceRatio;
            trade.realized_pnl *= pnlRatio;
            totalRealizedPnl += trade.realized_pnl;
        });
    } else { totalRealizedPnl = (newStepData.info.pnl_verification.realized_pnl || 0) * pnlRatio; }
    newStepData.info.pnl_verification.realized_pnl = totalRealizedPnl;
    newStepData.info.pnl_verification.unrealized_pnl = totalUnrealizedPnl;
    newStepData.info.pnl_verification.verified_total_pnl = totalRealizedPnl + totalUnrealizedPnl;
    newStepData.info.eval_episode_return = newStepData.info.pnl_verification.verified_total_pnl;
    if(newStepData.info.portfolio_stats) {
        const stats = newStepData.info.portfolio_stats;
        stats.max_profit *= pnlRatio;
        stats.max_loss *= pnlRatio;
        stats.net_premium = (stats.net_premium || 0) * pnlRatio;
        stats.breakevens = (stats.breakevens || []).map(be => ((be - envDefaults.startPrice) / envDefaults.strikeDistance * deNormParams.strikeDistance) + deNormParams.startPrice);
    }
    if (newStepData.info.payoff_data && newStepData.info.payoff_data.expiry_pnl) {
        const payoff = newStepData.info.payoff_data;
        payoff.spot_price *= priceRatio;
        if (payoff.sigma_levels && payoff.sigma_levels.plus_one) {
            payoff.sigma_levels.plus_one *= priceRatio;
            payoff.sigma_levels.minus_one *= priceRatio;
        }
        payoff.expiry_pnl.forEach(point => { point.price *= priceRatio; point.pnl *= pnlRatio; });
        payoff.current_pnl.forEach(point => { point.price *= priceRatio; point.pnl *= pnlRatio; });
    }
    return newStepData;
};

// ===================================================================================
//                            CHILD COMPONENTS
// ===================================================================================

function MetricsDashboard({ stepData }) {
    const info = stepData?.info || {};
    const pnlColor = (info.eval_episode_return || 0) > 0 ? '#4CAF50' : (info.eval_episode_return || 0) < 0 ? '#F44336' : 'white';
    const lastChangeColor = (info.last_price_change_pct || 0) > 0 ? '#4CAF50' : (info.last_price_change_pct || 0) < 0 ? '#F44336' : 'white';
    const pnlChangePct = info.initial_cash ? ((info.eval_episode_return || 0) / info.initial_cash) * 100 : 0;
    const pnlChangeColor = pnlChangePct > 0 ? '#4CAF50' : pnlChangePct < 0 ? '#F44336' : 'white';
    return (<div className="metrics-dashboard">
        <div className="metric-item"><h2>Market Regime</h2><p style={{ color: '#2196F3' }}>{(info.market_regime || 'N/A').replace("Historical: ", "")}</p></div>
        <div className="metric-item"><h2>Day</h2><p>{stepData?.day || 0}</p></div>
        <div className="metric-item"><h2>EOD Price</h2><p>${(info.price || 0).toFixed(2)}</p><p style={{ fontSize: '0.8em', color: lastChangeColor }}>{(info.last_price_change_pct || 0).toFixed(2)}% vs last step</p></div>
        <div className="metric-item"><h2>EOD Total PnL</h2><p style={{ color: pnlColor }}>${(info.eval_episode_return || 0).toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2})}</p><p style={{ fontSize: '0.8em', color: pnlChangeColor }}>{pnlChangePct.toFixed(2)}% vs Day 0</p></div>
        <div className="metric-item"><h2>Action Taken</h2><p style={{ textTransform: 'capitalize' }}>{(info.executed_action_name || 'N/A').replace(/_/g, ' ')}</p></div>
        <div className="metric-item"><h2>Directional Bias</h2><p style={{color: '#FFC107'}}>{info.directional_bias || 'N/A'}</p></div>
        <div className="metric-item"><h2>Volatility Bias</h2><p style={{color: '#03A9F4'}}>{info.volatility_bias || 'N/A'}</p></div>
    </div>);
}

function ActivePositions({ portfolio }) {
    if (!portfolio || portfolio.length === 0) return <p className="empty-message">Portfolio is empty.</p>;
    return (<table className="info-table"><thead><tr><th>Type</th><th>Direction</th><th>Strike</th><th>Entry Prem.</th><th>Current Prem.</th><th>Live PnL</th><th>DTE</th></tr></thead><tbody>
        {portfolio.map((pos, index) => {
            const pnlColor = pos.live_pnl > 0 ? '#4CAF50' : pos.live_pnl < 0 ? '#F44336' : 'white';
            return (<tr key={index} className={pos.direction.toLowerCase()}>
                <td>{pos.type.toUpperCase()}</td><td>{pos.direction.toUpperCase()}</td><td>${(pos.strike_price || 0).toFixed(2)}</td><td>${(pos.entry_premium || 0).toFixed(2)}</td>
                <td>${(pos.current_premium || 0).toFixed(2)}</td><td style={{ color: pnlColor }}>${(pos.live_pnl || 0).toFixed(2)}</td><td>{(pos.days_to_expiry || 0).toFixed(2)}</td>
            </tr>);
        })}
    </tbody></table>);
}

function ClosedTradesLog({ closedTrades }) {
    if (!closedTrades || closedTrades.length === 0) return <p className="empty-message">No trades closed yet.</p>;
    return (<table className="info-table"><thead><tr><th>Position</th><th>Strike</th><th>Entry/Exit Day</th><th>Entry/Exit Prem.</th><th>Realized P&L</th></tr></thead><tbody>
        {closedTrades.map((trade, index) => {
            const pnlColor = trade.realized_pnl > 0 ? '#4CAF50' : trade.realized_pnl < 0 ? '#F44336' : 'white';
            return (<tr key={index}>
                <td>{trade.position}</td><td>${(trade.strike || 0).toFixed(2)}</td><td>{trade.entry_day} → {trade.exit_day}</td>
                <td>${(trade.entry_prem || 0).toFixed(2)} → ${(trade.exit_prem || 0) ? trade.exit_prem.toFixed(2) : 'N/A'}</td>
                <td style={{ color: pnlColor }}>${(trade.realized_pnl || 0).toFixed(2)}</td>
            </tr>);
        })}
    </tbody></table>);
}

function PnlVerification({ verificationData }) {
    const { realized_pnl = 0, unrealized_pnl = 0, verified_total_pnl = 0 } = verificationData || {};
    const realizedColor = realized_pnl > 0 ? '#4CAF50' : realized_pnl < 0 ? '#F44336' : 'white';
    const unrealizedColor = unrealized_pnl > 0 ? '#4CAF50' : unrealized_pnl < 0 ? '#F44336' : 'white';
    const totalColor = verified_total_pnl > 0 ? '#4CAF50' : verified_total_pnl < 0 ? '#F44336' : 'white';
    return (<div className="pnl-verification">
        <div><span>Sum of Realized P&L:</span> <span style={{ color: realizedColor }}>${realized_pnl.toFixed(2)}</span></div>
        <div><span>Sum of Unrealized P&L:</span> <span style={{ color: unrealizedColor }}>${unrealized_pnl.toFixed(2)}</span></div>
        <div className="total"><span>Verified Total P&L:</span> <span style={{ color: totalColor }}>${verified_total_pnl.toFixed(2)}</span></div>
    </div>);
}

function PayoffDiagram({ payoffData }) {
    if (!payoffData || !payoffData.expiry_pnl || !payoffData.expiry_pnl.length) { return <p className="empty-message">Portfolio is empty.</p>; }
    const { expiry_pnl, current_pnl, spot_price = 0, sigma_levels = {} } = payoffData;
    const { plus_one = spot_price, minus_one = spot_price } = sigma_levels;
    const data = { labels: expiry_pnl.map(d => d.price), datasets: [ { label: 'P&L at Expiry', data: expiry_pnl.map(d => d.pnl), borderColor: '#E91E63', tension: 0.1, pointRadius: 0, fill: true, backgroundColor: (context) => { const chart = context.chart; const { ctx, chartArea } = chart; if (!chartArea) { return null; } let zeroPoint = (chart.scales.y.getPixelForValue(0) - chartArea.top) / (chartArea.bottom - chartArea.top); zeroPoint = Math.max(0, Math.min(1, zeroPoint)); const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom); const epsilon = 0.0001; gradient.addColorStop(0, 'rgba(76, 175, 80, 0.5)'); gradient.addColorStop(Math.max(0, zeroPoint - epsilon), 'rgba(76, 175, 80, 0.5)'); gradient.addColorStop(zeroPoint, 'rgba(244, 67, 54, 0.5)'); gradient.addColorStop(1, 'rgba(244, 67, 54, 0.5)'); return gradient; }, }, { label: 'Mark-to-Market P&L (T+0)', data: current_pnl.map(d => d.pnl), borderColor: '#2196F3', borderDash: [5, 5], tension: 0.4, pointRadius: 0 } ], };
    const allPnlValues = [...expiry_pnl.map(d => d.pnl), ...current_pnl.map(d => d.pnl)];
    const minY = Math.min(...allPnlValues); const maxY = Math.max(...allPnlValues); const padding = Math.abs(maxY - minY) * 0.1;
    const options = { responsive: true, maintainAspectRatio: false, scales: { x: { type: 'linear', ticks: { color: 'white', callback: (value) => '$' + value.toLocaleString() } }, y: { ticks: { color: 'white', callback: (value) => '$' + value.toLocaleString() }, min: minY - padding, max: maxY + padding, grid: { color: (c) => c.tick.value === 0 ? 'rgba(255, 255, 255, 0.8)' : 'rgba(255, 255, 255, 0.2)', lineWidth: (c) => c.tick.value === 0 ? 2 : 1 } } }, plugins: { legend: { labels: { color: 'white' } }, annotation: { annotations: { spotLine: { type: 'line', xMin: spot_price, xMax: spot_price, borderColor: '#4CAF50', borderWidth: 2, label: { content: `Spot: ${spot_price.toFixed(2)}`, enabled: true, position: 'start' } }, plusOneSigma: { type: 'line', xMin: plus_one, xMax: plus_one, borderColor: 'rgba(255, 255, 255, 0.3)', borderWidth: 1, borderDash: [6, 6] }, minusOneSigma: { type: 'line', xMin: minus_one, xMax: minus_one, borderColor: 'rgba(255, 255, 255, 0.3)', borderWidth: 1, borderDash: [6, 6] } } } } };
    return <div className="chart-container"><Line options={options} data={data} /></div>;
}

function AgentBehaviorChart({ episodeHistory, historicalContext, deNormParams, envDefaults }) {
  if (!episodeHistory || episodeHistory.length === 0) return null;
  const combinedPriceData = [...(historicalContext || []), ...episodeHistory.map(step => step.info.price)];
  const displayPriceData = combinedPriceData.map(p => (p / envDefaults.startPrice) * deNormParams.startPrice);
  const labels = Array.from({ length: displayPriceData.length }, (_, i) => i - (historicalContext ? historicalContext.length : 0));
  const data = { labels: labels, datasets: [{ label: 'Index Price', data: displayPriceData, segment: { borderColor: (ctx) => ctx.p0DataIndex < (historicalContext ? historicalContext.length : 0) ? 'rgba(3, 169, 244, 0.5)' : '#03A9F4', borderWidth: (ctx) => ctx.p0DataIndex < (historicalContext ? historicalContext.length : 0) ? 1 : 2 }, yAxisID: 'y', tension: 0.1, pointRadius: (context) => context.dataIndex < (historicalContext ? historicalContext.length : 0) ? 0 : 2 }] };
  const options = { responsive: true, maintainAspectRatio: false, scales: { x: { ticks: { color: 'white' } }, y: { ticks: { color: 'white', callback: function(value) { return '$' + value.toLocaleString(); } } } }, plugins: { legend: { display: false } } };
  return <div className="chart-container"><Line options={options} data={data} /></div>;
}

function PortfolioRiskDashboard({ portfolioStats }) {
    const { delta = 0, gamma = 0, theta = 0, vega = 0, max_profit = 0, max_loss = 0, rr_ratio = 0, prob_profit = 0, profit_factor = 0, breakevens = [], net_premium = 0 } = portfolioStats || {};
    const deltaColor = delta > 0 ? '#4CAF50' : delta < 0 ? '#F44336' : 'white';
    const formatBreakevens = (beArray) => {
        if (!beArray || beArray.length === 0) return 'N/A';
        return beArray.map(be => `$${be.toFixed(2)}`).join(' & ');
    };
    return (<div className="risk-dashboard">
        <div className="risk-item"><span>Portfolio Delta:</span><p style={{ color: deltaColor }}>{delta.toFixed(2)}</p></div>
        <div className="risk-item"><span>Portfolio Gamma:</span><p>{gamma.toFixed(2)}</p></div>
        <div className="risk-item"><span>Portfolio Theta:</span><p style={{ color: theta > 0 ? '#4CAF50' : '#F44336' }}>{theta.toFixed(2)}</p></div>
        <div className="risk-item"><span>Portfolio Vega:</span><p style={{ color: vega > 0 ? '#4CAF50' : '#F44336' }}>{vega.toFixed(2)}</p></div>
        <div className="risk-item"><span>Max Profit:</span><p style={{ color: '#4CAF50' }}>${max_profit.toLocaleString(undefined, {minimumFractionDigits: 2})}</p></div>
        <div className="risk-item"><span>Max Loss:</span><p style={{ color: '#F44336' }}>${max_loss.toLocaleString(undefined, {minimumFractionDigits: 2})}</p></div>
        <div className="risk-item"><span>Risk/Reward Ratio:</span><p>{isFinite(rr_ratio) ? `1 : ${rr_ratio.toFixed(2)}` : '1 : ∞'}</p></div>
        <div className="risk-item"><span>Prob. of Profit:</span><p>{(prob_profit * 100).toFixed(2)}%</p></div>
        <div className="risk-item"><span>Profit Factor:</span><p>{isFinite(profit_factor) ? profit_factor.toFixed(2) : '∞'}</p></div>
        <div className="risk-item"><span>Breakeven(s):</span><p style={{ color: '#FFC107' }}>{formatBreakevens(breakevens)}</p></div>
        <div className="risk-item"><span>Net Liq. Value:</span><p style={{ color: net_premium >= 0 ? '#4CAF50' : '#F44336' }}>${net_premium.toFixed(2)}</p></div>
    </div>);
}

function StrategyReport({ reportData }) {
    const [sortConfig, setSortConfig] = useState({ key: 'Trader_Score', direction: 'descending' });
    const sortedData = useMemo(() => {
        if (!reportData || !Array.isArray(reportData)) return [];
        let sortableData = [...reportData];
        if (sortConfig !== null) { sortableData.sort((a, b) => { const valA = a[sortConfig.key] === null ? Infinity : a[sortConfig.key] || -Infinity; const valB = b[sortConfig.key] === null ? Infinity : b[sortConfig.key] || -Infinity; if (valA < valB) return sortConfig.direction === 'ascending' ? -1 : 1; if (valA > valB) return sortConfig.direction === 'ascending' ? 1 : -1; return 0; }); }
        return sortableData;
    }, [reportData, sortConfig]);

    const requestSort = (key) => { let direction = 'ascending'; if (sortConfig && sortConfig.key === key && sortConfig.direction === 'ascending') { direction = 'descending'; } setSortConfig({ key, direction }); };
    const getSortIndicatorClass = (key) => { if (!sortConfig || sortConfig.key !== key) return 'sort-indicator-hidden'; return sortConfig.direction === 'ascending' ? 'sort-indicator' : 'sort-indicator descending'; };
    
    if (!reportData || !Array.isArray(reportData) || reportData.length === 0) return <p className="empty-message">No strategy data in this report.</p>;
    
    const columns = ["Trader_Score", "Strategy", "Total_Trades", "Win_Rate_%", "Expectancy_$", "Profit_Factor", "Avg_Win_$", "Avg_Loss_$", "Max_Win_$", "Max_Loss_$", "Win_Streak", "Loss_Streak"];
    
    return (<div className="card" style={{ flex: '1 1 100%', marginTop: '20px' }}><h3>Strategy Performance Report (Click Headers to Sort)</h3><div className="table-container"><table className="info-table sortable"><thead><tr>{columns.map(col => (<th key={col} onClick={() => requestSort(col)}>{col.replace(/_/g, ' ')}<span className={getSortIndicatorClass(col)}>▲</span></th>))}</tr></thead><tbody>
        {sortedData.map((row, index) => (
            <tr key={index}>
                {columns.map(col => {
                    const value = row[col];
                    let displayValue;

                    // <<< --- THE UI FIX IS HERE --- >>>
                    if (value === null) {
                        displayValue = col === 'Profit_Factor' ? '∞' : 'N/A';
                    } else if (typeof value === 'number') {
                        displayValue = value.toLocaleString(undefined, {minimumFractionDigits: 2, maximumFractionDigits: 2});
                    } else {
                        displayValue = value;
                    }
                    return (<td key={col}>{displayValue}</td>);
                })}
            </tr>
        ))}
    </tbody></table></div></div>);
}

function ReplayerView({ deNormParams, envDefaults }) {
    const [replayData, setReplayData] = useState(null);
    const [historicalContext, setHistoricalContext] = useState([]);
    const [currentStep, setCurrentStep] = useState(0);
    useEffect(() => { fetch(`/replay_log.json?t=${new Date().getTime()}`).then(res => res.ok ? res.json() : Promise.reject(res)).then(logObject => { setHistoricalContext(logObject.historical_context || []); setReplayData(logObject.episode_data || []); }).catch(error => console.error('Could not load replay_log.json', error)); }, []);
    const goToStep = (step) => setCurrentStep(Math.max(0, Math.min(replayData ? replayData.length - 1 : 0, step)));
    const rawStepData = replayData ? replayData[currentStep] : null;
    const displayedStepData = useMemo(() => {
        const paramsMatchDefaults = deNormParams.startPrice === envDefaults.startPrice && deNormParams.strikeDistance === envDefaults.strikeDistance && deNormParams.lotSize === envDefaults.lotSize;
        if (paramsMatchDefaults || !rawStepData || !replayData) {
            return rawStepData;
        } else {
            return reSimulateStep(rawStepData, deNormParams, envDefaults);
        }
    }, [rawStepData, deNormParams, envDefaults, replayData]);
    const episodeHistory = replayData ? replayData.slice(0, currentStep + 1) : [];
    if (!replayData || !displayedStepData) { return <p><i>Loading Replay Data... (Run an evaluation to generate `replay_log.json`)</i></p>; }
    return (<div className="main-content">
        <div className="top-bar"><h2>Step: {displayedStepData.step} / {displayedStepData.info.total_steps_in_episode} (Day: {displayedStepData.day})</h2><div className="navigation-buttons"><button onClick={() => goToStep(currentStep - 1)} disabled={currentStep === 0}>Prev S</button><button onClick={() => goToStep(currentStep + 1)} disabled={currentStep >= replayData.length - 1}>Next S</button></div><input type="range" min="0" max={replayData.length - 1} value={currentStep} onChange={(e) => setCurrentStep(Number(e.target.value))} className="slider" /></div>
        <MetricsDashboard stepData={displayedStepData} />
        <div className="charts-container">
            <div className="card chart-card"><h3>Agent Behavior</h3><AgentBehaviorChart episodeHistory={episodeHistory} historicalContext={historicalContext} deNormParams={deNormParams} envDefaults={envDefaults} /></div>
            <div className="card chart-card"><h3>Portfolio P&L Diagram</h3><PayoffDiagram payoffData={displayedStepData.info.payoff_data} /></div>
        </div>
        <div className="info-panels-container">
           <div className="card"><h3>Active Positions</h3><ActivePositions portfolio={displayedStepData.portfolio} /></div>
           <div className="card"><h3>Portfolio Risk Profile</h3><PortfolioRiskDashboard portfolioStats={displayedStepData.info.portfolio_stats} /></div>
           <div className="card"><h3>Closed Trades Log</h3><ClosedTradesLog closedTrades={displayedStepData.info.closed_trades_log} /></div>
           <div className="card"><h3>P&L Verification</h3><PnlVerification verificationData={displayedStepData.info.pnl_verification} /></div>
        </div>
    </div>);
}

// ===================================================================================
//                                MAIN APP COMPONENT
// ===================================================================================

function App() {
    const [reportHistory, setReportHistory] = useState([]);
    const [selectedReportFile, setSelectedReportFile] = useState(null);
    const [selectedReportData, setSelectedReportData] = useState(null); // Default to null
    const [isLoadingReport, setIsLoadingReport] = useState(false);
    const [view, setView] = useState('replayer');

    const [startPrice, setStartPrice] = useState('20000');
    const [strikeDistance, setStrikeDistance] = useState('50');
    const [lotSize, setLotSize] = useState('75');
    const [lots, setLots] = useState('1');

    const envDefaults = useMemo(() => ({ startPrice: 20000, strikeDistance: 50, lotSize: 75 }), []);
    const deNormParams = useMemo(() => ({
        startPrice: parseFloat(startPrice) || envDefaults.startPrice,
        strikeDistance: parseFloat(strikeDistance) || envDefaults.strikeDistance,
        lotSize: parseFloat(lotSize) || envDefaults.lotSize
    }), [startPrice, strikeDistance, lotSize, envDefaults]);

    useEffect(() => { fetch('/api/history').then(res => res.json()).then(data => setReportHistory(data)).catch(error => console.error('Error fetching report history:', error)); }, []);
    useEffect(() => {
        if (!selectedReportFile) return;
        setIsLoadingReport(true);
        fetch(`/reports/${selectedReportFile}`).then(res => res.json()).then(data => {
            setSelectedReportData(data);
            setIsLoadingReport(false);
        }).catch(error => {
            console.error('Error loading report file:', error);
            setIsLoadingReport(false);
        });
    }, [selectedReportFile]);

    const getCheckpointFilename = (reportFilename) => { if (!reportFilename) return null; const timestamp = reportFilename.replace('strategy_report_', '').replace('.json', ''); return `ckpt_best_${timestamp}.pth.tar`; };

    return (
        <div className="App">
            <header className="App-header">
                <h1>Options-Zero-Game Visualizer</h1>
                
                <div className="denorm-panel card">
                    <h3>Real-World Re-Simulation Parameters</h3>
                    <div className="input-group"><label>Start Price ($):</label><input type="number" value={startPrice} onChange={e => setStartPrice(e.target.value)} /></div>
                    <div className="input-group"><label>Strike Distance ($):</label><input type="number" value={strikeDistance} onChange={e => setStrikeDistance(e.target.value)} /></div>
                    <div className="input-group"><label>Lot Size (Shares/Contract):</label><input type="number" value={lotSize} onChange={e => setLotSize(e.target.value)} /></div>
                    <div className="input-group"><label>Lots:</label><input type="number" value={lots} onChange={e => setLots(e.target.value)} min="1" step="1"/></div>
                </div>

                <div className="navigation-buttons view-toggle">
                    <button onClick={() => setView('replayer')} disabled={view === 'replayer'}>Episode Replayer</button>
                    <button onClick={() => setView('history')} disabled={view === 'history'}>Strategy History</button>
                </div>
                
                {view === 'replayer' && (
                    <ReplayerView deNormParams={deNormParams} envDefaults={envDefaults} />
                )}

                {view === 'history' && (
                    <div className="main-content">
                        <h2>Historical Strategy Reports</h2>
                        <div className="report-selector">{reportHistory.length > 0 ? reportHistory.map(report => (<div key={report.filename} className="report-item"><button className="report-button" onClick={() => setSelectedReportFile(report.filename)} disabled={selectedReportFile === report.filename}>{report.label}</button><a href={`/reports/${getCheckpointFilename(report.filename)}`} className="download-link" download>Download Model</a></div>)) : <p>No historical reports found.</p>}</div>
                        {isLoadingReport ? <p>Loading report...</p> : <StrategyReport reportData={selectedReportData} />}
                    </div>
                )}
            </header>
        </div>
    );
}

export default App;
