import { blackScholes } from 'black-scholes';
import React, { useState, useEffect, useMemo } from 'react';
import './App.css';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler } from 'chart.js';
import annotationPlugin from 'chartjs-plugin-annotation';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend, Filler, annotationPlugin);

// ===================================================================================
//                            JAVASCRIPT RE-SIMULATOR (Bulletproof Version)
// ===================================================================================

const reSimulateStep = (rawStepData, deNormParams, envDefaults) => {
    if (!rawStepData) return null;

    const newStepData = JSON.parse(JSON.stringify(rawStepData));
    const priceRatio = deNormParams.startPrice / envDefaults.startPrice;
    const pnlRatio = deNormParams.lotSize / envDefaults.lotSize;

    // --- Ensure all necessary nested objects exist ---
    newStepData.info = newStepData.info || {};
    newStepData.portfolio = newStepData.portfolio || [];
    newStepData.info.pnl_verification = newStepData.info.pnl_verification || {};
    newStepData.info.portfolio_stats = newStepData.info.portfolio_stats || {};
    newStepData.info.payoff_data = newStepData.info.payoff_data || { expiry_pnl: [], current_pnl: [], sigma_levels: {} };

    // --- De-normalize all values, now safely ---
    newStepData.info.price = (newStepData.info.price || envDefaults.startPrice) * priceRatio;
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

    let totalRealizedPnl = (newStepData.info.pnl_verification.realized_pnl || 0) * pnlRatio;
    newStepData.info.pnl_verification.realized_pnl = totalRealizedPnl;
    newStepData.info.pnl_verification.unrealized_pnl = totalUnrealizedPnl;
    newStepData.info.pnl_verification.verified_total_pnl = totalRealizedPnl + totalUnrealizedPnl;
    newStepData.info.eval_episode_return = newStepData.info.pnl_verification.verified_total_pnl;
    
    const stats = newStepData.info.portfolio_stats;
    stats.max_profit = (stats.max_profit || 0) * pnlRatio;
    stats.max_loss = (stats.max_loss || 0) * pnlRatio;
    stats.net_premium = (stats.net_premium || 0) * pnlRatio;
    stats.breakevens = (stats.breakevens || []).map(be => ((be - envDefaults.startPrice) / envDefaults.strikeDistance * deNormParams.strikeDistance) + deNormParams.startPrice);
    
    const payoff = newStepData.info.payoff_data;
    payoff.spot_price = (payoff.spot_price || newStepData.info.price) * priceRatio;
    if (payoff.sigma_levels && payoff.sigma_levels.plus_one) {
        payoff.sigma_levels.plus_one *= priceRatio;
        payoff.sigma_levels.minus_one *= priceRatio;
    }
    payoff.expiry_pnl.forEach(point => { point.price *= priceRatio; point.pnl *= pnlRatio; });
    payoff.current_pnl.forEach(point => { point.price *= priceRatio; point.pnl *= pnlRatio; });

    return newStepData;
};


// ===================================================================================
//                            CHILD COMPONENTS
// ===================================================================================

function MetricsDashboard({ stepData, lots, envDefaults }) { // <-- NEW: Added lots and envDefaults
    const info = stepData?.info || {};
    const pnlColor = (info.eval_episode_return || 0) > 0 ? '#4CAF50' : (info.eval_episode_return || 0) < 0 ? '#F44336' : 'white';
    const lastChangeColor = (info.last_price_change_pct || 0) > 0 ? '#4CAF50' : (info.last_price_change_pct || 0) < 0 ? '#F44336' : 'white';

    // <<< --- THE DEFINITIVE FIX --- >>>
    // 1. Calculate the scaled initial cash based on the environment default and the user's "Lots" input.
    const scaledInitialCash = (envDefaults.initial_cash || 500000) * (lots || 1);

    // 2. Calculate the percentage change against this stable, correct denominator.
    const pnlChangePct = scaledInitialCash ? ((info.eval_episode_return || 0) / scaledInitialCash) * 100 : 0;
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

const calculateExpiryPnl = (portfolio, priceRange, lotSize, realizedPnl) => {
    if (!portfolio || portfolio.length === 0) {
        return [];
    }

    return priceRange.map(price => {
        let unrealizedPnl = 0;
        portfolio.forEach(leg => {
            const pnlMultiplier = leg.direction === 'long' ? 1 : -1;
            let intrinsicValue = 0;
            if (leg.type === 'call') {
                intrinsicValue = Math.max(0, price - leg.strike_price);
            } else { // put
                intrinsicValue = Math.max(0, leg.strike_price - price);
            }
            const pnlPerShare = intrinsicValue - leg.entry_premium;
            unrealizedPnl += pnlPerShare * pnlMultiplier * lotSize;
        });
        return { price: price, pnl: realizedPnl + unrealizedPnl };
    });
};

const calculateT0Pnl = (portfolio, priceRange, lotSize, realizedPnl) => {
    // This function now has everything it needs to be perfectly accurate.
    if (!portfolio || portfolio.length === 0) {
        return [];
    }

    const RISK_FREE_RATE = 0.10; // Must match backend config

    return priceRange.map(price => {
        let unrealizedPnl = 0;
        portfolio.forEach(leg => {
            const pnlMultiplier = leg.direction === 'long' ? 1 : -1;

            // The blackScholes function can return NaN if inputs are invalid (e.g., DTE is 0)
            let bsPrice = 0;
            if (leg.days_to_expiry > 0) {
                 bsPrice = blackScholes(
                    price,
                    leg.strike_price,
                    leg.days_to_expiry / 365.25,
                    leg.live_iv, // <-- Use the IV provided directly from the backend log!
                    RISK_FREE_RATE,
                    leg.type
                );
            }

            const pnlPerShare = (bsPrice || 0) - leg.entry_premium;
            unrealizedPnl += pnlPerShare * pnlMultiplier * lotSize;
        });
        return { price: price, pnl: realizedPnl + unrealizedPnl };
    });
};

function PayoffDiagram({ payoffData, portfolio, pnlVerification, lotSize }) {
    // ... (the top part of the function is unchanged) ...
    if (!payoffData || !portfolio || !pnlVerification || portfolio.length === 0) {
        return <p className="empty-message">Awaiting data or portfolio is empty.</p>;
    }
    
    const { spot_price, sigma_levels = {} } = payoffData;
    const { plus_one = spot_price, minus_one = spot_price } = sigma_levels;
    const realizedPnl = pnlVerification.realized_pnl || 0;
    
    const priceRange = Array.from({length: 100}, (_, i) => spot_price * 0.85 + (spot_price * 0.3 * (i/99)));

    const expiry_pnl = calculateExpiryPnl(portfolio, priceRange, lotSize, realizedPnl);
    const current_pnl = calculateT0Pnl(portfolio, priceRange, lotSize, realizedPnl);

    const data = {
        labels: priceRange,
        datasets: [
            {
                label: 'P&L at Expiry',
                data: expiry_pnl.map(d => d.pnl),
                borderColor: '#E91E63',
                tension: 0.1,
                pointRadius: 0,
                fill: true,
                // <<< --- THE DEFINITIVE FIX IS HERE --- >>>
                backgroundColor: (context) => {
                    const chart = context.chart;
                    const { ctx, chartArea } = chart;

                    // 1. Add a robust guard clause. If the chart area isn't ready, do nothing.
                    if (!chartArea) {
                        return null;
                    }

                    // 2. Safely get the zero-line pixel.
                    const zero = chart.scales.y.getPixelForValue(0);

                    // 3. If the zero line is completely off-screen, fill with a solid color.
                    if (zero < chartArea.top || zero > chartArea.bottom) {
                        const allValues = expiry_pnl.map(d => d.pnl);
                        // If there are no PNL values yet, return a neutral color
                        if (allValues.length === 0) return 'rgba(100, 100, 100, 0.3)';
                        // Determine if the entire chart is profit or loss
                        return Math.max(...allValues) > 0 ? 'rgba(76, 175, 80, 0.4)' : 'rgba(244, 67, 54, 0.4)';
                    }
                    
                    // 4. If the zero line is on-screen, create the gradient.
                    const gradient = ctx.createLinearGradient(0, chartArea.top, 0, chartArea.bottom);
                    const zeroPoint = (zero - chartArea.top) / (chartArea.bottom - chartArea.top);
                    
                    gradient.addColorStop(0, 'rgba(76, 175, 80, 0.5)'); // Green for profit
                    gradient.addColorStop(Math.max(0, zeroPoint - 0.001), 'rgba(76, 175, 80, 0.5)');
                    gradient.addColorStop(zeroPoint, 'rgba(244, 67, 54, 0.5)'); // Red for loss
                    gradient.addColorStop(1, 'rgba(244, 67, 54, 0.5)');
                    
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
                    // The variables plus_one and minus_one are now correctly defined in scope.
                    plusOneSigma: { type: 'line', xMin: plus_one, xMax: plus_one, borderColor: 'rgba(255, 255, 255, 0.3)', borderWidth: 1, borderDash: [6, 6], label: { content: '+1σ', enabled: true, position: 'start', font: { size: 10 } } },
                    minusOneSigma: { type: 'line', xMin: minus_one, xMax: minus_one, borderColor: 'rgba(255, 255, 255, 0.3)', borderWidth: 1, borderDash: [6, 6], label: { content: '-1σ', enabled: true, position: 'start', font: { size: 10 } } },
                }
            }
        }
    };
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
    const [sortConfig, setSortConfig] = useState({ key: 'ELO_Rank', direction: 'descending' });
    const sortedData = useMemo(() => {
        if (!reportData || !Array.isArray(reportData)) return [];
        let sortableData = [...reportData];
        if (sortConfig !== null) { sortableData.sort((a, b) => { const valA = a[sortConfig.key] === null ? Infinity : a[sortConfig.key] || -Infinity; const valB = b[sortConfig.key] === null ? Infinity : b[sortConfig.key] || -Infinity; if (valA < valB) return sortConfig.direction === 'ascending' ? -1 : 1; if (valA > valB) return sortConfig.direction === 'ascending' ? 1 : -1; return 0; }); }
        return sortableData;
    }, [reportData, sortConfig]);
    const requestSort = (key) => { let direction = 'ascending'; if (sortConfig && sortConfig.key === key && sortConfig.direction === 'ascending') { direction = 'descending'; } setSortConfig({ key, direction }); };
    const getSortIndicatorClass = (key) => { if (!sortConfig || sortConfig.key !== key) return 'sort-indicator-hidden'; return sortConfig.direction === 'ascending' ? 'sort-indicator' : 'sort-indicator descending'; };
    if (!reportData || !Array.isArray(reportData) || reportData.length === 0) return <p className="empty-message">No strategy data in this report.</p>;
    
    const columns = ["ELO_Rank", "Trader_Score", "Strategy", "Total_Trades", "Win_Rate_%", "Expectancy_$", "Profit_Factor", "Avg_Win_$", "Avg_Loss_$", "Max_Win_$", "Max_Loss_$", "CVaR_95%_$", "Win_Streak", "Loss_Streak"];
    
    return (<div className="card" style={{ flex: '1 1 100%', marginTop: '20px' }}><h3>Strategy Performance Report (Click Headers to Sort)</h3><div className="table-container"><table className="info-table sortable"><thead><tr>{columns.map(col => (<th key={col} onClick={() => requestSort(col)}>{col.replace(/_/g, ' ')}<span className={getSortIndicatorClass(col)}>▲</span></th>))}</tr></thead><tbody>
        {sortedData.map((row, index) => (
            <tr key={index}>
                {columns.map(col => {
                    const value = row[col];
                    let displayValue;
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

// <<< --- NEW: A truly "dumb" ReplayerView that only renders props --- >>>
function ReplayerView({ displayedStepData, replayData, currentStep, goToStep, episodeHistory, historicalContext, deNormParams, envDefaults, lots }) { // <-- NEW: Added lots
    if (!replayData || !displayedStepData) {
        return <p><i>Loading Replay Data... (Run an evaluation to generate `replay_log.json`)</i></p>;
    }
    return (
        <div className="main-content">
            <div className="top-bar">
                <h2>Step: {displayedStepData.step} / {displayedStepData.info.total_steps_in_episode} (Day: {displayedStepData.day})</h2>
                <div className="navigation-buttons">
                    <button onClick={() => goToStep(currentStep - 1)} disabled={currentStep === 0}>Prev S</button>
                    <button onClick={() => goToStep(currentStep + 1)} disabled={currentStep >= replayData.length - 1}>Next S</button>
                </div>
                <input type="range" min="0" max={replayData.length - 1} value={currentStep} onChange={(e) => goToStep(Number(e.target.value))} className="slider" />
            </div>
            
            {/* <<< --- NEW: Pass the new props down --- >>> */}
            <MetricsDashboard stepData={displayedStepData} lots={lots} envDefaults={envDefaults} />
            <div className="charts-container">
                <div className="card chart-card"><h3>Agent Behavior</h3><AgentBehaviorChart episodeHistory={episodeHistory} historicalContext={historicalContext} deNormParams={deNormParams} envDefaults={envDefaults} /></div>
                <div className="card chart-card"><h3>Portfolio P&L Diagram</h3>
					<PayoffDiagram 
						payoffData={displayedStepData.info.payoff_data}
						portfolio={displayedStepData.portfolio}
						pnlVerification={displayedStepData.info.pnl_verification}
						lotSize={deNormParams.lotSize}
					/>
	        </div>
            </div>
            <div className="info-panels-container">
               <div className="card"><h3>Active Positions</h3><ActivePositions portfolio={displayedStepData.portfolio} /></div>
               <div className="card"><h3>Portfolio Risk Profile</h3><PortfolioRiskDashboard portfolioStats={displayedStepData.info.portfolio_stats} /></div>
               <div className="card"><h3>Closed Trades Log</h3><ClosedTradesLog closedTrades={displayedStepData.info.closed_trades_log} /></div>
               <div className="card"><h3>P&L Verification</h3><PnlVerification verificationData={displayedStepData.info.pnl_verification} /></div>
            </div>
        </div>
    );
}

// ===================================================================================
//                                MAIN APP COMPONENT
// ===================================================================================

function App() {
    const [replayData, setReplayData] = useState(null);
    const [historicalContext, setHistoricalContext] = useState([]);
    const [currentStep, setCurrentStep] = useState(0);
    const [reportHistory, setReportHistory] = useState([]);
    const [selectedReportFile, setSelectedReportFile] = useState(null);
    const [selectedReportData, setSelectedReportData] = useState(null);
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

    useEffect(() => { fetch(`/replay_log.json?t=${new Date().getTime()}`).then(res => res.ok ? res.json() : Promise.reject(res)).then(logObject => { setHistoricalContext(logObject.historical_context || []); setReplayData(logObject.episode_data || []); }).catch(error => console.error('Could not load replay_log.json', error)); }, []);
    useEffect(() => { fetch('/api/history').then(res => res.json()).then(data => setReportHistory(data)).catch(error => console.error('Error fetching report history:', error)); }, []);
    useEffect(() => { if (!selectedReportFile) return; setIsLoadingReport(true); fetch(`/reports/${selectedReportFile}`).then(res => res.json()).then(data => { setSelectedReportData(data); setIsLoadingReport(false); }).catch(error => { console.error('Error loading report file:', error); setIsLoadingReport(false); }); }, [selectedReportFile]);

    const goToStep = (step) => setCurrentStep(Math.max(0, Math.min(replayData ? replayData.length - 1 : 0, step)));
    
    const rawStepData = replayData ? replayData[currentStep] : null;

    const displayedStepData = useMemo(() => {
        const paramsMatchDefaults = deNormParams.startPrice === envDefaults.startPrice && deNormParams.strikeDistance === envDefaults.strikeDistance && deNormParams.lotSize === envDefaults.lotSize;
        let baseStepData = rawStepData;
        if (!paramsMatchDefaults && rawStepData && replayData) {
            baseStepData = reSimulateStep(rawStepData, deNormParams, envDefaults);
        }
        if (!baseStepData) return null;
        const lotsMultiplier = Math.max(1, parseInt(lots) || 1);
        if (lotsMultiplier === 1) return baseStepData;
        const finalStepData = JSON.parse(JSON.stringify(baseStepData));
        finalStepData.info.eval_episode_return *= lotsMultiplier;
        if (finalStepData.info.pnl_verification) {
            finalStepData.info.pnl_verification.realized_pnl *= lotsMultiplier;
            finalStepData.info.pnl_verification.unrealized_pnl *= lotsMultiplier;
            finalStepData.info.pnl_verification.verified_total_pnl *= lotsMultiplier;
        }
        if (finalStepData.portfolio) { finalStepData.portfolio.forEach(leg => leg.live_pnl *= lotsMultiplier); }
        if (finalStepData.info.closed_trades_log) { finalStepData.info.closed_trades_log.forEach(trade => trade.realized_pnl *= lotsMultiplier); }
        if (finalStepData.info.portfolio_stats) { finalStepData.info.portfolio_stats.max_profit *= lotsMultiplier; finalStepData.info.portfolio_stats.max_loss *= lotsMultiplier; finalStepData.info.portfolio_stats.net_premium *= lotsMultiplier; }
        if (finalStepData.info.payoff_data && finalStepData.info.payoff_data.expiry_pnl) { finalStepData.info.payoff_data.expiry_pnl.forEach(point => point.pnl *= lotsMultiplier); finalStepData.info.payoff_data.current_pnl.forEach(point => point.pnl *= lotsMultiplier); }
        return finalStepData;
    }, [rawStepData, deNormParams, envDefaults, lots, replayData]);

    const episodeHistory = replayData ? replayData.slice(0, currentStep + 1) : [];
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
                    <ReplayerView 
                        displayedStepData={displayedStepData}
                        replayData={replayData}
                        currentStep={currentStep}
                        goToStep={goToStep}
                        episodeHistory={episodeHistory}
                        historicalContext={historicalContext}
                        deNormParams={deNormParams}
                        lots={lots}
                        envDefaults={envDefaults}
                    />
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
