import { useState, useEffect, useCallback } from "react";
import { Modal } from "antd";

// 用于管理 dashboard 的所有异步数据和状态
export default function useDashboardData() {
  const [tasks, setTasks] = useState([]);
  const [selectedTaskIndex, setSelectedTaskIndex] = useState(-1);
  const [scatterData, setScatterData] = useState([]);
  const [trajectoryData, setTrajectoryData] = useState([]);
  const [importance, setImportance] = useState([]);
  const [loading, setLoading] = useState(false);

  // 获取任务列表
  const fetchTasks = useCallback(() => {
    fetch('http://localhost:5001/api/Dashboard/tasks', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ action: 'ask for tasks information' })
    })
      .then(res => {
        if (!res.ok) throw new Error('任务列表获取失败');
        return res.json();
      })
      .then(data => {
        setTasks(data);
        setSelectedTaskIndex(0);
      })
      .catch(err => {
        Modal.error({ title: '任务列表获取失败', content: err.message });
      });
  }, []);

  // 获取图表数据
  const fetchCharts = useCallback((taskIndex) => {
    if (taskIndex < 0 || !tasks[taskIndex]) return;
    setLoading(true);
    fetch('http://localhost:5001/api/Dashboard/charts', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ taskname: tasks[taskIndex].problem_name })
    })
      .then(res => {
        if (!res.ok) throw new Error('图表数据获取失败');
        return res.json();
      })
      .then(data => {
        setScatterData(data.ScatterData);
        setTrajectoryData(data.TrajectoryData);
        setImportance(data.Importance);
      })
      .catch(err => {
        Modal.error({ title: '数据请求失败', content: err.message });
      })
      .finally(() => setLoading(false));
  }, [tasks]);

  // 初始化加载任务列表
  useEffect(() => {
    fetchTasks();
    // eslint-disable-next-line
  }, []);

  // 切换任务时自动加载对应图表
  useEffect(() => {
    if (selectedTaskIndex >= 0) {
      fetchCharts(selectedTaskIndex);
    }
  }, [selectedTaskIndex, fetchCharts]);

  return {
    tasks,
    selectedTaskIndex,
    setSelectedTaskIndex,
    scatterData,
    trajectoryData,
    importance,
    loading
  };
}
