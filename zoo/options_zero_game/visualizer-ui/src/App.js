import React, { useState, useEffect } from 'react';
import './App.css';

// <<< MODIFIED: The MetricsDashboard now accepts and displays the price change
function MetricsDashboard({ day, price, pnl, actionName, startPrice, marketRegime, illegalActions, lastPriceChangePct, directionalBias, volatilityBias }) {
  const pnlColor = pnl > 0 ? '#4CAF50' : pnl < 0 ? '#F44336' : 'white';
  const cumulativeChange = price && startPrice ? ((price / startPrice) - 1) * 100 : 0;
  const cumulativeChangeColor = cumulativeChange > 0 ? '#4CAF50' : cumulativeChange < 0 ? '#F44336' : 'white';
  
  // <<< NEW: Color logic for the last price change
  const lastChangeColor = lastPriceChangePct > 0 ? '#4CAF50' : lastPriceChangePct < 0 ? '#F44336' : 'white';

  return (
    <div className="metrics-dashboard">
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
        <p>${price ? price.toFixed(2) : '0.00'}</p>
        {/* <<< NEW: Display for the last price change */}
        <p style={{ fontSize: '0.8em', color: lastChangeColor }}>
          {lastPriceChangePct ? lastPriceChangePct.toFixed(2) : '0.00'}% vs last step
        </p>
      </div>
      <div className="metric-item">
        <h2>EOD Total PnL</h2>
        <p style={{ color: pnlColor }}>${pnl ? pnl.toFixed(2) : '0.00'}</p>
        <p style={{ fontSize: '0.8em', color: cumulativeChangeColor, fontWeight: 'bold' }}>
          {cumulativeChange.toFixed(2)}% vs Day 0
        </p>
      </div>
      <div className="metric-item">
        <h2>Action Taken</h2>
        <p style={{fontSize: '1.1em', color: '#ddd', textTransform: 'capitalize'}}>
          {(actionName || 'N/A').replace(/_/g, ' ')}
        </p>
      </div>
      <div className="metric-item">
        <h2>Illegal Attempts</h2>
        <p style={{color: illegalActions > 0 ? '#FFC107' : 'white'}}>{illegalActions}</p>
      </div>
      <div className="metric-item">
        <h2>Directional Bias</h2>
        <p style={{color: '#FFC107'}}>{directionalBias || 'N/A'}</p>
      </div>
      <div className="metric-item">
        <h2>Volatility Bias</h2>
        <p style={{color: '#03A9F4'}}>{volatilityBias || 'N/A'}</p>
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
  return rawHistory;
}

function App() {
  const [replayData, setReplayData] = useState([]);
  const [currentStep, setCurrentStep] = useState(0);

  useEffect(() => {
    fetch(`/replay_log.json?t=${new Date().getTime()}`)
      .then(response => response.json())
      .then(rawHistory => {
        console.log('Successfully loaded raw replay data:', rawHistory);
        const processedData = processReplayData(rawHistory);
        console.log('Processed step data:', processedData);
        setReplayData(processedData);
      })
      .catch(error => console.error('Error loading replay_log.json:', error));
  }, []);

  const handleSliderChange = (event) => {
    setCurrentStep(Number(event.target.value));
  };

  const stepData = replayData[currentStep];

  return (
    <div className="App">
      <header className="App-header">
        <h1>Options-Zero-Game Replayer</h1>
        
        {replayData.length === 0 ? (
          <p><i>Loading and Processing Replay Data...</i></p>
        ) : (
          <div style={{ width: '90%', maxWidth: '1200px', marginTop: '20px' }}>
            
            <h2>Step: {currentStep} / {replayData.length - 1} (Day: {stepData ? stepData.day : 0})</h2>
            <input
              type="range"
              min="0"
              max={replayData.length - 1}
              value={currentStep}
              onChange={handleSliderChange}
              style={{ width: '100%' }}
            />

            {stepData && (
              <>
                <MetricsDashboard 
                  day={stepData.day} 
                  price={stepData.info.price} 
                  pnl={stepData.info.eval_episode_return}
                  actionName={stepData.info.action_name}
                  startPrice={stepData.info.start_price}
                  marketRegime={stepData.info.market_regime}
                  illegalActions={stepData.info.illegal_actions_in_episode}
                  lastPriceChangePct={stepData.info.last_price_change_pct} // <<< Pass the new prop
		  directionalBias={stepData.info.directional_bias}
		  volatilityBias={stepData.info.volatility_bias}
                />
                <h2 style={{marginTop: '40px'}}>Positions at Step {currentStep}</h2>
                <PortfolioTable portfolio={stepData.portfolio} />
              </>
            )}
          </div>
        )}
      </header>
    </div>
  );
}

export default App;
