import {
  Chart as ChartJS,
  CategoryScale,
  LinearScale,
  BarElement,
  Title,
  Tooltip,
  Legend,
} from 'chart.js';
import {Bar, Line} from 'react-chartjs-2';
import TitleCard from '../../../components/Cards/TitleCard';
import React, {useMemo} from "react";

ChartJS.register(CategoryScale, LinearScale, BarElement, Title, Tooltip, Legend);

function BarChart({ ImportanceData }){

    const options = {
        responsive: true,
        plugins: {
          legend: {
            position: 'top',
          }
        },
      };
      
      const data = useMemo(() => (
          {
              labels:  Object.keys(ImportanceData ?? {}) ?? [],
              datasets: [{
                      label: 'Importance level',
                      data: Object.values(ImportanceData ?? {}) ?? [],
                      backgroundColor: 'rgba(255, 99, 132, 1)',
                  }],
          }
      ), [ImportanceData]);

    return(
        <div style={{
            width: "350px",
        }}>
            <div style={{
                fontWeight: "600",
                fontSize: "1.25rem",
                lineHeight: "1.75rem",
                marginBottom: "10px"
            }}>
                Importance of variables
            </div>
            <Bar options={options} data={data} />
        </div>
    )
}


export default BarChart