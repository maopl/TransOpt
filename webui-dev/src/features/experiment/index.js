import React, { useState, useEffect, useCallback } from "react";
import SelectTask from "./components/SelectTask";
import SelectAlgorithm from "./components/SelectAlgorithm";
import RunPage from '../../features/run/index';
import {Card, Divider, Spin, message, Form, Input, Select, Button, Modal} from "antd";
import { LoadingOutlined } from '@ant-design/icons';

/**
 * 任务的数据格式转换
 * @param tasks
 * @returns {*}
 */
const convertTasks = (tasks) => tasks.map(task => ({
  name: task.name,
  num_vars: parseInt(task.num_vars),
  num_objs: task.num_objs,
  fidelity: task.fidelity,
  workloads: task.workloads,
  budget_type: task.budget_type,
  budget: task.budget,
}));

/**
 * 将传统的算法值格式转换为新格式
 * @param {Object} oldFormatValue - 旧格式的算法值
 * @returns {Object} 新格式的算法值
 */
const transformToNewFormat = (oldFormatValue) => {
  const ALGORITHM_TYPES = [
    "SearchSpace",
    "Initialization",
    "Pretrain",
    "Model",
    "AcquisitionFunction",
    "Normalizer"
  ];
  
  const algorithms = ALGORITHM_TYPES.map(name => {
    // 获取算法类型值
    const type = oldFormatValue[name] || '';
    // 获取对应的数据集
    const datasetsKey = `${name}SelectedDatasets`;
    const datasets = oldFormatValue[datasetsKey] || [];
    // 提取数据集名称，转换为简单字符串数组
    const auxiliaryData = datasets.map(dataset => dataset.name || dataset.value);
    
    // 自动选择功能，目前假设所有自动选择都是false，后续可根据需求调整
    const autoSelect = false;
    
    return {
      name,
      type,
      auxiliaryData,
      autoSelect
    };
  });
  
  return { algorithms };
};

/**
 * 将新格式的算法值转换为传统格式
 * @param {Object} newFormatValue - 新格式的算法值
 * @returns {Object} 旧格式的算法值
 */
const transformToOldFormat = (newFormatValue) => {
  const result = {};
  
  if (newFormatValue.algorithms && Array.isArray(newFormatValue.algorithms)) {
    newFormatValue.algorithms.forEach(algorithm => {
      // 设置算法类型值
      result[algorithm.name] = algorithm.type;
      
      // 转换辅助数据为数据集对象数组
      const datasets = algorithm.auxiliaryData.map((dataName, index) => ({
        value: dataName,
        name: dataName,
        key: index.toString()
      }));
      
      // 设置数据集
      result[`${algorithm.name}SelectedDatasets`] = datasets;
    });
  }
  
  return result;
};

const Experiment = () => {

  // 简化状态变量
  const [loading, setLoading] = useState(true);

  const [tasksData, setTasksData] = useState([]);

  const [tasks, setTasks] = useState([]); // State to store tasks added from Drawer

  const [optimizer, setOptimizer] = useState({});

  const [algorithmData, setAlgorithmData] = useState({
    spaceRefiner: [],
    sampler: [],
    pretrain: [],
    model: [],
    acf: [],
    normalizer: [],
    datasetSelector: []
  });
  
  // 统一初始formValues结构 (保留原有格式以兼容现有组件)
  const [algorithmValue, setAlgorithmValue] = useState({
    "SearchSpace": "",
    "Initialization": "",
    "Pretrain": "",
    "Model": "",
    "AcquisitionFunction": "",
    "Normalizer": "",
    // 下面是各自的数据集等参数
    "SearchSpaceSelectedDatasets": [],
    "InitializationSelectedDatasets": [],
    "PretrainSelectedDatasets": [],
    "ModelSelectedDatasets": [],
    "AcquisitionFunctionSelectedDatasets": [],
    "NormalizerSelectedDatasets": [],
    // 你可以继续添加其它参数
  });
  
  // 新的目标结构
  const [transformedAlgorithmValue, setTransformedAlgorithmValue] = useState({
    optimizer: [
      {
        name: "SearchSpace",
        type: algorithmData.spaceRefiner?.[0]?.name || '',
        auxiliaryData: [],
        autoSelect: false
      },
      {
        name: "Initialization",
        type: algorithmData.sampler?.[0]?.name || '',
        auxiliaryData: [],
        autoSelect: false
      },
      {
        name: "Pretrain",
        type: algorithmData.pretrain?.[0]?.name || '',
        auxiliaryData: [],
        autoSelect: false
      },
      {
        name: "Model",
        type: algorithmData.model?.[0]?.name || '',
        auxiliaryData: [],
        autoSelect: false
      },
      {
        name: "AcquisitionFunction",
        type: algorithmData.acf?.[0]?.name || '',
        auxiliaryData: [],
        autoSelect: false
      },
      {
        name: "Normalizer",
        type: algorithmData.normalizer?.[0]?.name || '',
        auxiliaryData: [],
        autoSelect: false
      }
    ]
  });
  


  // 在算法值更新时，同步更新转换后的格式
  useEffect(() => {
    setTransformedAlgorithmValue(transformToNewFormat(algorithmValue));
  }, [algorithmValue]);
  
  // 创建一个能够更新新格式数据的函数
  const updateTransformedAlgorithmValue = useCallback((updatedValue) => {
    console.log('Updating transformed algorithm value:', updatedValue);
    // 使用完全新的对象引用来确保React检测到状态变化
    setTransformedAlgorithmValue(prevState => {
      // 确保这是一个新的对象引用
      const newState = JSON.parse(JSON.stringify(updatedValue));
      console.log('New state after deep copy:', newState);
      return newState;
    });
    
    // 同时更新旧格式，以保持兼容性
    const oldFormat = transformToOldFormat(updatedValue);
    console.log('Converted back to old format:', oldFormat);
    setAlgorithmValue(oldFormat);
  }, []);

  const onFinish = (values) => {
    console.log('Form values:', values);
    console.log('tasks', tasks);
    // 使用新格式的算法值提交
    console.log('Transformed algorithm value:', transformedAlgorithmValue);
    
    // 构建最终提交的数据结构
    const finalSubmitData = {
      ...values,
      tasks,
      ...transformedAlgorithmValue // 包含新格式的算法值
    };
    
    console.log('Final submit data:', finalSubmitData);
    
    // 后续处理逻辑...

    // 向后端发送请求...
    fetch('http://localhost:5001/api/configuration/run', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(finalSubmitData),
    })
        .then(response => {
          if (!response.ok) {
            throw new Error('Network response was not ok');
          }
          return response.json();
        })
        .then(isSucceed => {
          console.log('Message from back-end:', isSucceed);
          Modal.success({
            title: 'Information',
            content: 'Run Successfully!'
          })
        })
        .catch((error) => {
          console.error('Error sending message:', error);
          var errorMessage = error.error;
          Modal.error({
            title: 'Information',
            content: 'Error:' + errorMessage
          })
        });

  };

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
         *     "SearchSpace",
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
        
        // 获取运行配置信息 todo 这里要在梳理一下
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
          // 初始算法下拉框的选项
          setAlgorithmValue({
            "SearchSpace": configData.optimizer.SearchSpace,
            "Initialization": configData.optimizer.Initialization,
            "Pretrain": configData.optimizer.Pretrain,
            "Model": configData.optimizer.Model,
            "AcquisitionFunction": configData.optimizer.AcquisitionFunction,
            "Normalizer": configData.optimizer.Normalizer,
            // 下面是各自的数据集等参数
            "SearchSpaceSelectedDatasets": [],
            "InitializationSelectedDatasets": [],
            "PretrainSelectedDatasets": [],
            "ModelSelectedDatasets": [],
            "AcquisitionFunctionSelectedDatasets": [],
            "NormalizerSelectedDatasets": [],
            // 你可以继续添加其它参数
          })
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

        <Form
            name="main_form"
            onFinish={onFinish}
            style={{ width: "100%" }}
            autoComplete="off"
            initialValues={{
              Seeds: "42",
              Remote: "False"
            }}
        >
          <Divider orientation="left">
            <div style={{fontSize: '24px', marginBottom: '15px'}} className="text-xl font-semibold">Experimental Setup</div>
          </Divider>
          <Form.Item
              name="experimentName"
              style={{ marginBottom: '10px' }} // Add margin bottom
          >
            <Input
                placeholder="Experiment name"
                style={{
                  width: '300px', // Full width of the container
                  fontSize: '32px', // Font size
                  resize: 'vertical', // Allow vertical resizing only
                }}
            />
          </Form.Item>

          <Form.Item
              name="experimentDescription"
              style={{ marginBottom: '16px' }} // Add margin bottom
          >
            <Input.TextArea
                placeholder="Type the description of the experiment"
                style={{
                  width: '100%', // Full width of the container
                  height: '100px', // Height of the text area
                  fontSize: '16px', // Font size
                  resize: 'vertical', // Allow vertical resizing only
                }}
            />
          </Form.Item>
          <Form.Item>
            <SelectTask data={tasksData}
                        // updateTable={setTasksData}
                        tasks={tasks}
                        setTasks={setTasks}
            />
          </Form.Item>
          <Divider orientation="left">
            <div style={{fontSize: '24px', marginBottom: '15px'}} className="text-xl font-semibold">Algorithm Building</div>
          </Divider>

          <Form.Item>
            <SelectAlgorithm
                SearchSpaceOptions={algorithmData.spaceRefiner}
                InitializationOptions={algorithmData.sampler}
                PretrainOptions={algorithmData.pretrain}
                ModelOptions={algorithmData.model}
                AcquisitionFunctionOptions={algorithmData.acf}
                NormalizerOptions={algorithmData.normalizer}
                updateTable={setOptimizer}
                algorithmValue={algorithmValue}
                setAlgorithmValue={setAlgorithmValue}
                transformedAlgorithmValue={transformedAlgorithmValue}
                updateTransformedAlgorithmValue={updateTransformedAlgorithmValue}
            />
          </Form.Item>
          <div style={{ overflowY: 'auto', maxHeight: '150px', display: 'flex', alignItems: 'baseline', justifyContent: 'space-between', gap: 30 }}>
            <div style={{ display: 'flex', alignItems: 'baseline' }}>
              <h6 style={{ color: "black" }}>Seeds</h6>
              <Form.Item name="Seeds" style={{ marginRight: 10, marginLeft: 10 }}>
                <Input />
              </Form.Item>
              <h6 style={{ color: "black" }}>Remote</h6>
              <Form.Item name="Remote" style={{ marginRight: 10, marginLeft: 10 }}>
                <Select
                    options={[{ value: "True" },
                      { value: "False" },
                    ]}
                />
              </Form.Item>
              <h6 style={{ color: "black" }}>ServerURL</h6>
              <Form.Item name="ServerURL" style={{ marginLeft: 10 }}>
                <Input />
              </Form.Item>
            </div>
            <Form.Item>
              <Button type="primary" htmlType="submit" style={{ width: "150px", backgroundColor: 'rgb(53, 162, 235)' }}>
                Run
              </Button>
            </Form.Item>
          </div>
          {/*<Form.Item>*/}
          {/*  <div style={{ marginTop: '25px' }}></div>*/}
          {/*  <RunPage run={onFinish} />*/}
          {/*</Form.Item>*/}
        </Form>

      </div>
    </Card>
  );
};

export default Experiment;