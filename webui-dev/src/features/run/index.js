import React, { useState } from "react";
import Run from "./components/Run"
import RunProgress from "./components/RunProgress"

const RunPage = ({run}) => {
  const [isRunning, setIsRunning] = useState(false);
  
  // 处理运行按钮点击
  const handleRun = () => {
    setIsRunning(true);
    
    // 如果有父组件传递的run函数，也调用它
    if (typeof run === 'function') {
      run();
    }
  };

  return (
    <div>
      <Run run={handleRun} />
      <RunProgress isRunning={isRunning} />
    </div>
  );
}

export default RunPage;
