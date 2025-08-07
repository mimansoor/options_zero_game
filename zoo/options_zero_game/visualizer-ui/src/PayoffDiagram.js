import React from 'react';
import { Line } from 'react-chartjs-2';

function PayoffDiagram({ payoffData, currentPnl, currentPrice }) {
  if (!payoffData || !payoffData.expiry_pnl || payoffData.expiry_pnl.length === 0) {
    return <p>No position to graph.</p>;
  }

  const data = {
    labels: payoffData.expiry_pnl.map(d => d.price.toFixed(2)),
    datasets: [
      {
        label: 'P&L at Expiry',
        data: payoffData.expiry_pnl.map(d => d.pnl),
        borderColor: 'rgb(255, 159, 64)',
      },
       {
        label: 'Current Mark-to-Market P&L',
        data: [{ x: currentPrice, y: currentPnl }],
        borderColor: 'rgb(153, 102, 255)',
        backgroundColor: 'rgb(153, 102, 255)',
        pointRadius: 6,
        type: 'scatter',
      },
    ],
  };

  const options = { /* ... Chart options ... */ };
  return <Line options={options} data={data} />;
}
export default PayoffDiagram;
