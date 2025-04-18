import React from "react";
import LineChart from '../components/LineChart';
import BarChart from '../components/BarChart';
import Footprint from '../components/ScatterChart';

const ChartsPanel = ({ TrajectoryData, Importance, ScatterData }) => (
  <div className="grid lg:grid-cols-3 mt-4 grid-cols-1 gap-6">
    <LineChart TrajectoryData={TrajectoryData} />
    <BarChart ImportanceData={Importance} />
    <Footprint ScatterData={ScatterData} />
  </div>
);

export default ChartsPanel;
