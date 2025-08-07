import React from 'react';
import { Line } from 'react-chartjs-2';
import { Chart as ChartJS, CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend } from 'chart.js';
ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, Title, Tooltip, Legend);

function AgentBehaviorChart({ history }) {
  const data = {
    labels: history.map(step => step.step),
    datasets: [
      {
        label: 'Index Price',
        data: history.map(step => step.info.price),
        borderColor: 'rgb(75, 192, 192)',
        backgroundColor: 'rgba(75, 192, 192, 0.5)',
        yAxisID: 'y',
      },
    ],
  };

  const options = { /* ... Chart options ... */ };
  return <Line options={options} data={data} />;
}
export default AgentBehaviorChart;
