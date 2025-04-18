import React, { useState, useEffect } from "react";
import SelectTask from "./components/SelectTask";
import SelectAlgorithm from "./components/SelectAlgorithm";
import RunPage from '../../features/run/index';
import { Card, Divider, Spin, message } from "antd";
import { LoadingOutlined } from '@ant-design/icons';
const Experiment = () => {
  // 简化状态变量
  const [loading, setLoading] = useState(true);
  const [tasksData, setTasksData] = useState([]);
  const [algorithmData, setAlgorithmData] = useState({
    spaceRefiner: [],
    sampler: [],
    pretrain: [],
    model: [],
    acf: [],
    normalizer: [],
    datasetSelector: []
  });
  const [optimizer, setOptimizer] = useState({});
  
  // 加载数据
  useEffect(() => {
    // 统一的数据加载函数
    const loadData = async () => {
      try {
        setLoading(true);
        const requestBody = { action: 'ask for basic information' };
        
        // 获取基本配置信息
        const basicResponse = await fetch('http://localhost:5001/api/configuration/basic_information', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
        });
        
        if (!basicResponse.ok) throw new Error('Failed to fetch basic information');
        const basicData = await basicResponse.json();
        console.log('Basic info from backend:', basicData);

        /**
         *     更新算法数据
         *     "Search Space",
         *     "Initialization",
         *     "Pretrain",
         *     "Model",
         *     "Acquisition Function",
         *     "Normalizer"
         */
        setAlgorithmData({
          spaceRefiner: basicData.SpaceRefiner || [],
          sampler: basicData.Sampler || [],
          pretrain: basicData.Pretrain || [],
          model: basicData.Model || [],
          acf: basicData.ACF || [],
          normalizer: basicData.Normalizer || [],
          datasetSelector: basicData.DataSelector || []
        });
        
        // 获取运行配置信息
        const configResponse = await fetch('http://localhost:5001/api/RunPage/get_info', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(requestBody),
        });
        
        if (!configResponse.ok) throw new Error('Failed to fetch configuration info');
        const configData = await configResponse.json();
        console.log('Config info from backend:', configData);
        
        // 更新任务数据
        if (configData.tasks && configData.tasks.length > 0) {
          const formattedTasks = configData.tasks.map((task, index) => ({
            ...task,
            index
          }));
          setTasksData(formattedTasks);
        } else if (basicData.TasksData && basicData.TasksData.length > 0) {
          // 后备: 如果第二个请求没有任务数据，使用第一个请求的数据
          setTasksData(basicData.TasksData);
        }
        
        // 更新优化器数据
        if (configData.optimizer) {
          setOptimizer(configData.optimizer);
        }
      } catch (error) {
        console.error('Error loading data:', error);
        message.error('Failed to load experiment data: ' + error.message);
      } finally {
        setLoading(false);
      }
    };
    
    loadData();
  }, []);

  // 自定义灰色系图标
  const antIcon = <LoadingOutlined style={{ fontSize: 48, color: '#9E9E9E' }} spin />;
  
  if (loading) {
    return (
      <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
        <Spin indicator={antIcon}>
          <div style={{ padding: '60px', background: 'rgba(0, 0, 0, 0.03)', borderRadius: '8px' }}>
            <p style={{ marginTop: '32px', textAlign: 'center', color: '#616161' }}>Preparing experiment interface...</p>
          </div>
        </Spin>
      </div>
    );
  }
  
  return (
    <Card>
      <div className="grid mt-4">

        <Divider orientation="left">
          <div style={{fontSize: '24px', marginBottom: '15px'}} className="text-xl font-semibold">Experimental Setup</div>
        </Divider>
        <SelectTask data={tasksData} updateTable={setTasksData} />
        
        <Divider orientation="left">
          <div style={{fontSize: '24px', marginBottom: '15px'}} className="text-xl font-semibold">Algorithm Building</div>
        </Divider>
        <SelectAlgorithm
            SearchSpaceOptions={algorithmData.spaceRefiner}
            InitializationOptions={algorithmData.sampler}
            PretrainOptions={algorithmData.pretrain}
            ModelOptions={algorithmData.model}
            AcquisitionFunctionOptions={algorithmData.acf}
            NormalizerOptions={algorithmData.normalizer}
            updateTable={setOptimizer}
        />
        
      <div style={{ marginTop: '25px' }}></div>
        <RunPage />

      </div>
    </Card>
  );
};

export default Experiment;