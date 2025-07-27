import React, { useState, useEffect } from 'react';
import './App.css';

// <<< MODIFIED: The MetricsDashboard now accepts and displays the market regime
function MetricsDashboard({ day, price, pnl, actionName, startPrice, dailyChange, marketRegime }) {
  const pnlColor = pnl > 0 ? '#4CAF50' : pnl < 0 ? '#F44336' : 'white';
  const cumulativeChange = price && startPrice ? ((price / startPrice) - 1) * 100 : 0;
  const cumulativeChangeColor = cumulativeChange > 0 ? '#4CAF50' : cumulativeChange < 0 ? '#F44336' : 'white';
  
  const dailyChangeColor = dailyChange > 0 ? '#4CAF50' : dailyChange < 0 ? '#F44336' : 'white';
  const dailyChangeString = dailyChange ? `(${dailyChange.toFixed(2)}%)` : '';

  return (
    <div className="metrics-dashboard">
      {/* <<< NEW: Market Regime Display */}
      <div className="metric-item">
        <h2>Market Regime</h2>
        <p style={{color: '#2196F3', fontWeight: 'bold'}}>{(marketRegime || 'N/A').replace(/_/g, ' ')}</p>
      </div>
      <div className="metric-item">
        <h2>Day</h2>
        <p>{day}</p>
      </div>
      <div className="metric-item">
        <h2>EOD Price</h2>
        <p>
          ${price ? price.toFixed(2) : '0.00'}
          <span style={{ fontSize: '0.7em', color: dailyChangeColor, marginLeft: '10px' }}>
            {dailyChangeString}
          </span>
        </p>
        <p style={{ fontSize: '0.8em', color: cumulativeChangeColor, fontWeight: 'bold' }}>
          {cumulativeChange.toFixed(2)}% vs Day 0
        </p>
      </div>
      <div className="metric-item">
        <h2>EOD Total PnL</h2>
        <p style={{ color: pnlColor }}>${pnl ? pnl.toFixed(2) : '0.00'}</p>
      </div>
      <div className="metric-item">
        <h2>Action Taken</h2>
        <p style={{fontSize: '1.1em', color: '#ddd', textTransform: 'capitalize'}}>
          {(actionName || 'N/A').replace(/_/g, ' ')}
        </p>
      </div>
    </div>
  );
}

function PortfolioTable({ portfolio }) {
  if (!portfolio || portfolio.length === 0) {
    return <p style={{marginTop: '20px', fontStyle: 'italic'}}>Portfolio is empty.</p>;
  }

  return (
    <table className="portfolio-table">
      <thead>
        <tr>
          <th>Type</th>
          <th>Direction</th>
          <th>Strike</th>
          <th>Entry Premium</th>
          <th>Current Premium</th>
          <th>Live PnL</th>
          <th>DTE</th>
        </tr>
      </thead>
      <tbody>
        {portfolio.map((pos, index) => {
          const pnlColor = pos.live_pnl > 0 ? '#4CAF50' : pos.live_pnl < 0 ? '#F44336' : 'white';
          return (
            <tr key={index} className={pos.direction}>
              <td>{pos.type.toUpperCase()}</td>
              <td>{pos.direction.toUpperCase()}</td>
              <td>${pos.strike_price.toFixed(2)}</td>
              <td>${pos.entry_premium.toFixed(2)}</td>
              <td>${pos.current_premium.toFixed(2)}</td>
              <td style={{ color: pnlColor, fontWeight: 'bold' }}>${pos.live_pnl.toFixed(2)}</td>
              <td>{pos.days_to_expiry.toFixed(2)}</td>
            </tr>
          );
        })}
      </tbody>
    </table>
  );
}

function processReplayData(rawHistory) {
  if (!rawHistory || rawHistory.length === 0) {
    return [];
  }
  const dailySummaries = [];
  const initialState = rawHistory[0];
  dailySummaries.push({
    day: 0,
    actionName: 'EPISODE_START',
    eodPrice: initialState.info.price,
    eodTotalPnl: initialState.info.eval_episode_return,
    eodPortfolio: initialState.portfolio,
    startPrice: initialState.info.start_price,
    dailyChange: 0,
    // <<< NEW: Capture the market regime for the whole episode
    marketRegime: initialState.info.market_regime,
  });

  for (let day = 1; day <= 30; day++) {
    const actionStepIndex = (day * 2) - 1;
    const marketCloseStepIndex = day * 2;
    if (marketCloseStepIndex >= rawHistory.length) break;
    const actionStep = rawHistory[actionStepIndex];
    const marketCloseStep = rawHistory[marketCloseStepIndex];

    dailySummaries.push({
      day: day,
      actionName: actionStep.info.action_name,
      eodPrice: marketCloseStep.info.price,
      eodTotalPnl: marketCloseStep.info.eval_episode_return,
      eodPortfolio: marketCloseStep.portfolio,
      startPrice: marketCloseStep.info.start_price,
      dailyChange: marketCloseStep.info.daily_change_pct,
      // <<< NEW: Pass the regime to each daily summary
      marketRegime: marketCloseStep.info.market_regime,
    });
  }
  return dailySummaries;
}


function App() {
  const [dailySummaries, setDailySummaries] = useState([]);
  const [currentDay, setCurrentDay] = useState(0);

  useEffect(() => {
    // Use a cache-busting query parameter to ensure we always get the latest log
    fetch(`/replay_log.json?t=${new Date().getTime()}`)
      .then(response => response.json())
      .then(rawHistory => {
        console.log('Successfully loaded raw replay data:', rawHistory);
        const processedData = processReplayData(rawHistory);
        console.log('Processed daily summaries:', processedData);
        setDailySummaries(processedData);
      })
      .catch(error => console.error('Error loading replay_log.json:', error));
  }, []);

  const handleSliderChange = (event) => {
    setCurrentDay(Number(event.target.value));
  };

  const dayData = dailySummaries[currentDay];

  return (
    <div className="App">
      <header className="App-header">
        <h1>Options-Zero-Game Replayer</h1>
        
        {dailySummaries.length === 0 ? (
          <p><i>Loading and Processing Replay Data...</i></p>
        ) : (
          <div style={{ width: '90%', maxWidth: '1200px', marginTop: '20px' }}>
            
            <h2>Day: {currentDay} / {dailySummaries.length - 1}</h2>
            <input
              type="range"
              min="0"
              max={dailySummaries.length - 1}
              value={currentDay}
              onChange={handleSliderChange}
              style={{ width: '100%' }}
            />

            {dayData && (
              <>
                <MetricsDashboard 
                  day={dayData.day} 
                  price={dayData.eodPrice} 
                  pnl={dayData.eodTotalPnl}
                  actionName={dayData.actionName}
                  startPrice={dayData.startPrice}
                  dailyChange={dayData.dailyChange}
                  // <<< NEW: Pass the market regime prop
                  marketRegime={dayData.marketRegime}
                />
                <h2 style={{marginTop: '40px'}}>End-of-Day Positions</h2>
                <PortfolioTable portfolio={dayData.eodPortfolio} />
              </>
            )}
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
