import React, { useState } from "react";
import Run from "./components/Run"
import RunProgress from "./components/RunProgress"

const RunPage = ({run}) => {
  const [get_info, setGetInfo] = useState(false);
  const [tasks, setTasks] = useState([]);
  const [optimizer, setOptimizer] = useState({});
  const [datasets, setDatasets] = useState({});

  return (
    <div>
      <Run run={run} />
      <RunProgress />
    </div>
  );
}

export default RunPage;
