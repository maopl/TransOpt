import React, { useState, useEffect, useCallback } from "react";
import { LineChartOutlined, CaretDownOutlined, CaretRightOutlined } from '@ant-design/icons';
import {
  Modal,
  Spin,
  Card,
  Input,
  Select,
  Form,
  Button as AntButton,
  Space,
  DatePicker,
  Row,
  Col,
  Typography,
  Button,
  Popconfirm,
  Tree
} from "antd";
import {
  LoadingOutlined,
  SearchOutlined,
  FilterOutlined,
  SortAscendingOutlined,
  InfoCircleOutlined,
  DatabaseOutlined,
  AreaChartOutlined,
  ArrowRightOutlined,
  DeleteOutlined,
  ExperimentOutlined,
  FileOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';

import LineChart from './components/LineChart';
import BarChart from './components/BarChart';
import Footprint from "./components/ScatterChart";
import StatisticalAnalysis from "./components/StatisticalAnalysis";

const { Option } = Select;
const { RangePicker } = DatePicker;
const { Text, Title } = Typography;

// 统一的卡片样式
const cardStyle = {
  borderRadius: "8px",
  boxShadow: "0 1px 2px -2px rgba(0, 0, 0, 0.16), 0 3px 6px 0 rgba(0, 0, 0, 0.12), 0 5px 12px 4px rgba(0, 0, 0, 0.09)"
};

// 统一的卡片内容样式
const cardBodyStyle = { padding: '16px' };

const Dashboard = () => {
  // 状态管理
  const [selectedExperimentIndex, setSelectedExperimentIndex] = useState(-1);
  const [selectedTaskIndex, setSelectedTaskIndex] = useState(-1);
  const [tasksInfo, setTasksInfo] = useState([]);
  const [scatterData, setScatterData] = useState([]);
  const [trajectoryData, setTrajectoryData] = useState([]);
  const [isModalVisible, setIsModalVisible] = useState(false);
  const [isMoreInfoModalVisible, setIsMoreInfoModalVisible] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const [isInitialLoading, setIsInitialLoading] = useState(true); // 首次加载状态
  const [importance, setImportance] = useState(null);
  
  // 展开/折叠实验状态
  const [expandedExperiments, setExpandedExperiments] = useState({});

  // 搜索表单状态
  const [searchForm] = Form.useForm();
  const [searchExperimentName, setSearchExperimentName] = useState('');
  const [searchProblemName, setSearchProblemName] = useState('');
  const [searchDateRange, setSearchDateRange] = useState(null);

  // 处理搜索表单提交
  const handleSearch = (values) => {
    console.log('Search form values:', values);
    const { experimentName, problemName, dateRange } = values;
    
    // 设置搜索状态 - 仅更新前端显示过滤条件，不请求后端
    setSearchExperimentName(experimentName || '');
    setSearchProblemName(problemName || '');
    setSearchDateRange(dateRange);
    
    // 计算新的过滤结果
    const newFilteredExperiments = filterExperiments(
      tasksInfo, 
      experimentName || '', 
      problemName || '', 
      dateRange
    );
    
    // 如果过滤后有结果，自动选择第一项
    if (newFilteredExperiments.length > 0) {
      const firstExp = newFilteredExperiments[0];
      if (firstExp.filteredProblems.length > 0) {
        // 寻找在过滤后数据中原始索引
        const originalExpIndex = tasksInfo.findIndex(exp => 
          exp.experimentName === firstExp.experimentName);
        
        if (originalExpIndex !== -1) {
          const originalProblemIndex = tasksInfo[originalExpIndex].problemList.findIndex(prob => 
            prob.problem_name === firstExp.filteredProblems[0].problem_name);
          
          if (originalProblemIndex !== -1) {
            setSelectedExperimentIndex(originalExpIndex);
            setSelectedTaskIndex(originalProblemIndex);
          }
        }
      }
    } else {
      // 如果没有匹配结果，清除选择
      setSelectedExperimentIndex(-1);
      setSelectedTaskIndex(-1);
    }
  };
  
  // 重置搜索
  const handleResetSearch = () => {
    searchForm.resetFields();
    setSearchExperimentName('');
    setSearchProblemName('');
    setSearchDateRange(null);
    
    // 重置后如果已有数据，选择第一项
    if (tasksInfo.length > 0) {
      setSelectedExperimentIndex(0);
      if (tasksInfo[0].problemList.length > 0) {
        setSelectedTaskIndex(0);
      }
    }
  };

  // 基于搜索条件过滤实验和问题
  const filterExperiments = useCallback((data, experimentName, problemName, dateRange) => {
    if (!data || !data.length) return [];
    
    return data.map(experiment => {
      // 实验名称过滤
      const experimentNameMatch = !experimentName || 
        experiment.experimentName.toLowerCase().includes(experimentName.toLowerCase());
      
      if (!experimentNameMatch) return { ...experiment, filteredProblems: [] };
      
      // 过滤问题列表
      const filteredProblems = experiment.problemList.filter(problem => {
        // 问题名称搜索
        const problemNameMatch = !problemName ||
          problem.displayName?.toLowerCase().includes(problemName.toLowerCase()) ||
          problem.problem_name.toLowerCase().includes(problemName.toLowerCase());
        
        // 日期范围过滤
        let dateRangeMatch = true;
        if (dateRange && dateRange.length === 2 && dateRange[0] && dateRange[1]) {
          try {
            // 提取日期范围的开始和结束时间
            const startDate = dayjs(dateRange[0]);
            const endDate = dayjs(dateRange[1]);
            
            // 从问题名称中提取时间戳
            const parts = problem.problem_name.split('_');
            if (parts.length > 0) {
              const timestamp = parts[parts.length - 1];
              
              // 检查最后一部分是否为时间戳（数字）
              if (/^\d+$/.test(timestamp)) {
                // 将时间戳（秒）转换为dayjs对象
                const timestampMs = Number(timestamp) * 1000;
                const problemDate = dayjs(timestampMs);
                
                console.log('日期比较:', {
                  problemName: problem.problem_name,
                  timestamp,
                  problemDate: problemDate.format('YYYY-MM-DD HH:mm:ss'),
                  startDate: startDate.format('YYYY-MM-DD HH:mm:ss'),
                  endDate: endDate.format('YYYY-MM-DD HH:mm:ss')
                });
                
                // 正确的日期范围比较逻辑
                // 问题日期必须：(在开始日期之后或等于开始日期) 并且 (在结束日期之前或等于结束日期)
                dateRangeMatch = (problemDate.isAfter(startDate) || problemDate.isSame(startDate, 'second')) && 
                                 (problemDate.isBefore(endDate) || problemDate.isSame(endDate, 'second'));
              } else {
                console.log('无法解析时间戳:', timestamp);
              }
            }
          } catch (error) {
            console.error("日期比较错误:", error);
            dateRangeMatch = true; // 出错时不过滤
          }
        }
        
        return problemNameMatch && dateRangeMatch
      });
      
      return {
        ...experiment,
        filteredProblems
      };
    }).filter(experiment => experiment.filteredProblems.length > 0);
  }, []);

  // 自定义灰色系图标
  const antIcon = <LoadingOutlined style={{ fontSize: 48, color: '#9E9E9E' }} spin />;

  // 获取任务列表
  useEffect(() => {
    if (selectedExperimentIndex === -1) {
      const messageToSend = {
        action: 'ask for tasks information',
      };

      fetch('http://localhost:5001/api/Dashboard/tasks', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(messageToSend),
      })
        .then(response => {
          if (!response.ok) {
            throw new Error('Network response was not ok');
          }
          return response.json();
        })
        .then(data => {
          console.log('Message from back-end:', data);

          const experimentsData = data.map(experiment => {
            // Process each problem to add displayName
            const updatedProblemList = experiment.problemList.map(problem => {
              // Parse the problem_name to extract timestamp and create formatted date
              const parts = problem.problem_name.split('_');
              let displayName = problem.problem_name;
              
              if (parts.length > 0) {
                const timestamp = parts[parts.length - 1];
                
                // Check if the last part is a timestamp (number)
                if (/^\d+$/.test(timestamp)) {
                  try {
                    // Convert timestamp to readable date format
                    // Unix timestamp is typically in seconds, but JS Date expects milliseconds
                    const date = new Date(Number(timestamp) * 1000);
                    
                    // Format the date as YYYY-MM-DD HH:MM:SS
                    const year = date.getFullYear();
                    const month = String(date.getMonth() + 1).padStart(2, '0');
                    const day = String(date.getDate()).padStart(2, '0');
                    const hours = String(date.getHours()).padStart(2, '0');
                    const minutes = String(date.getMinutes()).padStart(2, '0');
                    const seconds = String(date.getSeconds()).padStart(2, '0');
                    
                    const formattedDate = `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
                    
                    // Replace the timestamp with formatted date in problem_name
                    const newParts = [...parts];
                    newParts[newParts.length - 1] = formattedDate;
                    displayName = newParts.join('_');
                  } catch (error) {
                    console.error("Error formatting timestamp:", error);
                    // Fallback to original problem_name if date conversion fails
                  }
                }
              }
              
              return {
                ...problem,
                displayName
              };
            });
            
            return {
              ...experiment,
              problemList: updatedProblemList
            };
          });
          
          setTasksInfo(experimentsData);
          
          // 初始化展开第一个实验
          if (experimentsData.length > 0) {
            const initialExpandedState = {};
            experimentsData.forEach((exp, index) => {
              initialExpandedState[index] = index === 0; // 只展开第一个
            });
            setExpandedExperiments(initialExpandedState);
            
            // 如果第一个实验有问题列表，选中第一个问题
            if (experimentsData[0].problemList && experimentsData[0].problemList.length > 0) {
              setSelectedExperimentIndex(0);
              setSelectedTaskIndex(0);
            }
          }
          
          setIsInitialLoading(false); // 加载完成后设置为false
        })
        .catch((error) => {
          console.error('Error sending message:', error);
          setIsInitialLoading(false); // 出错时也设置为false
        });
    }
  }, [selectedExperimentIndex]);
  
  // 处理实验展开/折叠
  const toggleExperiment = (experimentIndex, e) => {
    e?.stopPropagation?.(); // 防止触发实验的点击事件
    setExpandedExperiments(prev => ({
      ...prev,
      [experimentIndex]: !prev[experimentIndex]
    }));
  };

  // 定时获取数据
  useEffect(() => {
    // 如果没有选择任务，不执行
    if (selectedExperimentIndex === -1 || selectedTaskIndex === -1 || !tasksInfo.length) return;

    const intervalId = setInterval(fetchData, 1000000);

    // 组件卸载时清除定时器
    return () => clearInterval(intervalId);
  }, [selectedExperimentIndex, selectedTaskIndex, tasksInfo]);

  // 获取轨迹数据
  const fetchData = useCallback(async () => {
    // 如果没有选择任务，不执行
    if (selectedExperimentIndex === -1 || selectedTaskIndex === -1 || !tasksInfo.length) return;

    try {
      const messageToSend = {
        taskname: tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].problem_name,
      };

      const response = await fetch('http://localhost:5001/api/Dashboard/trajectory', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(messageToSend)
      });

      if (!response.ok) {
        throw new Error('Network response was not ok');
      }

      const data = await response.json();
      console.log('Data from server:', data);

      setScatterData(data.ScatterData);
      setTrajectoryData(data.TrajectoryData);
    } catch (error) {
      console.error('Error fetching data:', error);
    }
  }, [selectedExperimentIndex, selectedTaskIndex, tasksInfo]);

  // 处理任务选择
  const handleTaskClick = useCallback((experimentIndex, taskIndex) => {
    // 如果点击的是已选中的任务，则不执行
    if (selectedExperimentIndex === experimentIndex && selectedTaskIndex === taskIndex) return;

    console.log('Selected experiment index:', experimentIndex, 'problem index:', taskIndex);
    setSelectedExperimentIndex(experimentIndex);
    setSelectedTaskIndex(taskIndex);
    setIsLoading(true);

    const selectedExperiment = tasksInfo[experimentIndex];
    const selectedProblem = selectedExperiment.problemList[taskIndex];

    const messageToSend = {
      taskname: selectedProblem.problem_name,
    };

    fetch('http://localhost:5001/api/Dashboard/charts', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(messageToSend),
    })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        setScatterData(data.ScatterData);
        setTrajectoryData(data.TrajectoryData);
        setIsLoading(false);
      })
      .catch((error) => {
        console.error('Error sending message:', error);
        setIsLoading(false);
      });
  }, [selectedExperimentIndex, selectedTaskIndex, tasksInfo]);

  // 处理实验选择
  const handleExperimentClick = useCallback((index) => {
    // 如果点击的是已选中的实验，则不执行
    if (selectedExperimentIndex === index) return;

    console.log('Selected experiment index:', index);
    setSelectedExperimentIndex(index);
    setSelectedTaskIndex(0);
    setIsLoading(true);

    const messageToSend = {
      taskname: tasksInfo[index].problemList[0].problem_name,
    };

    fetch('http://localhost:5001/api/Dashboard/charts', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(messageToSend),
    })
      .then(response => {
        if (!response.ok) {
          throw new Error('Network response was not ok');
        }
        return response.json();
      })
      .then(data => {
        setScatterData(data.ScatterData);
        setTrajectoryData(data.TrajectoryData);
        setIsLoading(false);
      })
      .catch((error) => {
        console.error('Error sending message:', error);
        setIsLoading(false);
      });
  }, [selectedExperimentIndex, tasksInfo]);

  // 更多信息弹窗相关函数
  const showMoreInfoModal = () => setIsMoreInfoModalVisible(true);
  const handleMoreInfoCancel = () => setIsMoreInfoModalVisible(false);

  const handleOk = () => {
    console.log(errorMessage);
    setIsModalVisible(false);

    const messageToSend = {
      errorMessage: errorMessage
    };

    fetch("http://localhost:5001/api/Dashboard/errorsubmit", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(messageToSend),
    })
      .then((response) => {
        if (!response.ok) {
          throw new Error("Network response was not ok");
        }
        return response.json();
      })
      .then((data) => {
        console.log("Message sent successfully:", data);
        setIsModalVisible(false);
        setErrorMessage("");
      })
      .catch((error) => {
        console.error("Error sending message:", error);
      });
  };

  const handleCancel = () => setIsModalVisible(false);

  const handleInputChange = e => setErrorMessage(e.target.value);

  // 删除任务处理函数
  const handleDelete = (taskName) => {
    const messageToSend = {
      datasets: [taskName],
    }
    console.log(messageToSend)
    fetch('http://localhost:5001/api/configuration/delete_dataset', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify(messageToSend),
    })
    .then(response => {
      if (!response.ok) {
        throw new Error('Network response was not ok');
      } 
      return response.json();
    })
    .then(succeed => {
      console.log('Message from back-end:', succeed);
      // 删除UI中的数据
      const updatedTasks = tasksInfo.map(experiment => {
        return {
          ...experiment,
          problemList: experiment.problemList.filter(task => task.problem_name !== taskName)
        }
      });
      setTasksInfo(updatedTasks);
      
      // 如果删除的是当前选中的任务，则选中第一个任务或重置
      if (updatedTasks[selectedExperimentIndex] && updatedTasks[selectedExperimentIndex].problemList.length > 0) {
        setSelectedTaskIndex(0);
      } else {
        setSelectedTaskIndex(-1);
      }
      
      Modal.success({
        title: 'Information',
        content: 'Delete successfully!'
      });
    })
    .catch((error) => {
      console.error('Error sending message:', error);
      var errorMessage = error.message || 'Unknown error';
      Modal.error({
        title: 'Information',
        content: 'Error: ' + errorMessage
      });
    });
  };

  // 首次渲染时的加载状态
  if (isInitialLoading) {
    return (
      <div style={{
        height: "100vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        flexDirection: "column"
      }}>
        <Spin indicator={antIcon} />
        <span style={{ marginTop: "16px", color: "#9E9E9E" }}>Loading tasks...</span>
      </div>
    );
  }

  // 如果还没有数据
  if (selectedExperimentIndex === -1 || !tasksInfo.length) {
    return (
      <div style={{
        height: "100vh",
        display: "flex",
        justifyContent: "center",
        alignItems: "center",
        flexDirection: "column"
      }}>
        <InfoCircleOutlined style={{ fontSize: 48, color: '#9E9E9E' }} />
        <Text style={{ marginTop: "16px", color: "#9E9E9E", fontSize: "16px" }}>
          No tasks found. Please create a task first.
        </Text>
      </div>
    );
  }

  // 过滤后的任务列表
  const filteredExperiments = filterExperiments(tasksInfo, searchExperimentName, searchProblemName, searchDateRange);

  // 自定义图标和颜色
  const titleRender = (nodeData) => {
    const isExperiment = nodeData.key.indexOf('-') === -1;
    
    // 处理实验节点
    if (isExperiment) {
      const experimentIndex = parseInt(nodeData.key);
      const isSelected = selectedExperimentIndex === experimentIndex;
      
      return (
        <div style={{ 
          display: 'flex', 
          alignItems: 'center', 
          width: '100%',
          color: isSelected ? '#1890ff' : 'rgba(0, 0, 0, 0.85)',
          fontWeight: isSelected ? '500' : 'normal'
        }}>
          <ExperimentOutlined style={{ marginRight: '8px', color: isSelected ? '#1890ff' : '#666' }} />
          <span>{nodeData.title}</span>
        </div>
      );
    } 
    // 处理问题节点
    else {
      const [experimentIndex, taskIndex] = nodeData.key.split('-').map(Number);
      const isSelected = selectedExperimentIndex === experimentIndex && selectedTaskIndex === taskIndex;
      
      return (
          <div style={{
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
            width: '300px',
            color: isSelected ? '#1890ff' : 'rgba(0, 0, 0, 0.65)',
            fontWeight: isSelected ? '500' : 'normal'
          }}>
            <div style={{ display: 'flex', alignItems: 'center' }}>
              <FileOutlined style={{ marginRight: '8px', color: isSelected ? '#1890ff' : '#999' }} />
              <span>{nodeData.title}</span>
            </div>
            <Popconfirm
              title="Delete this task"
              description="Are you sure you want to delete this task?"
              onConfirm={() => {
                const task = filteredExperiments[experimentIndex].filteredProblems[taskIndex];
                handleDelete(task.problem_name);
              }}
              okText="Yes"
              cancelText="No"
              placement="right"
            >
              <DeleteOutlined
                style={{ color: '#ff4d4f', fontSize: '14px' }}
                onClick={(e) => e.stopPropagation?.()}
              />
            </Popconfirm>
         </div>
      );
    }
  };

  // 主界面渲染
  return (
    <div style={{
      height: "86vh",
      padding: "20px",
      display: "flex",
      flexDirection: "column",
      overflow: "hidden",
      backgroundColor: "#f5f5f5"
    }}>
      {/* 顶部搜索表单 */}
      <Card
        bodyStyle={{ padding: "16px" }}
        style={{
          marginBottom: "16px",
          ...cardStyle
        }}
      >
        <Form
          form={searchForm}
          layout="horizontal"
          onFinish={handleSearch}
        >
          <Row gutter={16} align="middle">
            <Col span={6}>
              <Form.Item name="experimentName" style={{ marginBottom: 0 }}>
                <Input
                  placeholder="Expriment Name"
                  prefix={<SearchOutlined />}
                  allowClear
                  size="middle"
                />
              </Form.Item>
            </Col>
            <Col span={6}>
              <Form.Item name="problemName" style={{ marginBottom: 0 }}>
                <Input
                    placeholder="Problem Name"
                    prefix={<SearchOutlined />}
                    allowClear
                    size="middle"
                />
              </Form.Item>
            </Col>
            <Col span={7}>
              <Form.Item name="dateRange" style={{ marginBottom: 0 }}>
                <RangePicker
                    style={{ width: "100%" }} size="middle"
                    showTime={{ format: 'HH:mm:ss' }}
                    format="YYYY-MM-DD HH:mm:ss"
                />
              </Form.Item>
            </Col>
            <Col span={5}>
              <Space>
                <AntButton
                  type="primary"
                  htmlType="submit"
                  icon={<FilterOutlined />}
                  size="middle"
                >
                  Filter
                </AntButton>
                <AntButton onClick={handleResetSearch} size="middle">
                  Reset
                </AntButton>
              </Space>
            </Col>
          </Row>
        </Form>
      </Card>

      {/* 主内容区域 - 左右布局 */}
      <div style={{
        display: "flex",
        flex: "1 1 auto",
        overflow: "hidden",
        minHeight: 0, // 关键: 允许flex子项收缩到小于内容高度
        gap: "16px" // 统一间距
      }}>
        {/* 左侧数据集列表 */}
        <div style={{
          minWidth: "420px",
          width: "420px",
          display: "flex",
          flexDirection: "column",
          overflow: "hidden",
          minHeight: 0 // 关键: 允许flex子项收缩到小于内容高度
        }}>
          <Card
            bodyStyle={{
              padding: "16px",
              display: "flex",
              flexDirection: "column",
              height: "100%",
              overflow: "hidden"
            }}
            style={{
              height: "100%",
              ...cardStyle
            }}
          >
            {/* 列表头部 - 显示结果数量 */}
            <div style={{
              display: "flex",
              justifyContent: "space-between",
              alignItems: "center",
              marginBottom: "16px",
              borderBottom: "1px solid #f0f0f0",
              paddingBottom: "12px"
            }}>
              <Space>
                <DatabaseOutlined style={{ color: "#1890ff" }} />
                <Text strong>{filteredExperiments.length} Results</Text>
              </Space>

              {/*<AntButton*/}
              {/*  type="text"*/}
              {/*  size="small"*/}
              {/*  icon={<SortAscendingOutlined />}*/}
              {/*  title="Sort by name"*/}
              {/*/>*/}
            </div>

            {/* 这个div是专门用于滚动的容器 */}
            <div style={{
              overflowY: "auto",
              flex: "1 1 auto",
              paddingRight: "10px",
              minHeight: 0, // 关键: Ein mub flex子项收缩到小于内容高度
              marginBottom: "10px" // 防止内容太靠近底部
            }}>
              <Tree
                treeData={filteredExperiments.map((experiment, index) => ({
                  title: experiment.experimentName,
                  key: index.toString(),
                  icon: <ExperimentOutlined />,
                  children: experiment.filteredProblems.map((task, taskIndex) => ({
                    title: task.displayName,
                    key: `${index}-${taskIndex}`,
                    icon: <FileOutlined />,
                    isLeaf: true,
                  })),
                }))}
                onSelect={(selectedKeys, info) => {
                  if (selectedKeys.length === 0) return;
                  
                  const key = selectedKeys[0];
                  
                  // 如果是实验节点
                  if (key.indexOf('-') === -1) {
                    const experimentIndex = parseInt(key);
                    handleExperimentClick(experimentIndex);
                  } else {
                    // 如果是问题节点
                    const [experimentIndex, taskIndex] = key.split('-').map(Number);
                    handleTaskClick(experimentIndex, taskIndex);
                  }
                }}
                expandedKeys={Object.keys(expandedExperiments)
                  .filter(key => expandedExperiments[key])
                  .map(key => key.toString())}
                onExpand={(expandedKeys, info) => {
                  const key = info.node.key;
                  
                  // 只处理实验节点的展开
                  if (key.indexOf('-') === -1) {
                    toggleExperiment(parseInt(key), info.event);
                  }
                }}
                titleRender={titleRender}
                showIcon={false}
                selectedKeys={[selectedTaskIndex !== -1 
                  ? `${selectedExperimentIndex}-${selectedTaskIndex}` 
                  : selectedExperimentIndex.toString()]}
                style={{ fontSize: '14px' }}
              />
            </div>
          </Card>
        </div>

        {/* 右侧内容区域 */}
        <div style={{
          flex: "1 1 auto",
          overflow: "hidden",
          display: "flex",
          flexDirection: "column",
          minHeight: 0 // 关键: 允许flex子项收缩到小于内容高度
        }}>
          {isLoading ? (
            // 加载中状态
            <div style={{
              display: "flex",
              width: "100%",
              justifyContent: "center",
              alignItems: "center",
              height: "100%",
              flexDirection: "column",
              backgroundColor: "white",
              borderRadius: "8px",
              ...cardStyle
            }}>
              <Spin indicator={antIcon} />
              <span style={{ marginTop: "16px", color: "#9E9E9E" }}>Loading data...</span>
            </div>
          ) : (
            // 数据展示
            <div style={{
              height: "100%",
              overflowY: "auto",
              paddingRight: "5px"
            }}>
              {/* 详情卡片 */}
              <Card
                className="mb-4"
                bodyStyle={cardBodyStyle}
                style={cardStyle}
              >
                <div style={{ padding: '8px' }}>
                  {/* 标题和任务名称 */}
                  <div style={{
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "space-between",
                    marginBottom: "16px",
                    borderBottom: "1px solid #f0f0f0",
                    paddingBottom: "12px"
                  }}>
                    <Space>
                      <InfoCircleOutlined style={{ color: "#1890ff", fontSize: "18px" }} />
                      <Title level={4} style={{ margin: 0 }}>
                        {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].displayName}
                      </Title>
                    </Space>

                    <Space>
                      <AntButton
                        type="primary"
                        icon={<ArrowRightOutlined />}
                        onClick={showMoreInfoModal}
                      >
                        More Info
                      </AntButton>
                      <Popconfirm
                        title="Delete this task"
                        description="Are you sure you want to delete this task?"
                        onConfirm={() => handleDelete(tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].problem_name)}
                        okText="Yes"
                        cancelText="No"
                      >
                        <AntButton
                          type="primary"
                          danger
                          icon={<DeleteOutlined />}
                        >
                          Delete
                        </AntButton>
                      </Popconfirm>
                    </Space>
                  </div>

                  <section style={{ marginBottom: '20px', borderBottom: '1px solid #e0e0e0', paddingBottom: '15px' }}>
                    <Title level={5} style={{ color: '#333', marginBottom: '10px' }}>
                      Problem Information
                    </Title>
                    <Row gutter={[16, 8]}>
                      <Col span={24}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Problem Name:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].displayName}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Variable num:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].dim}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Objective num:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].obj}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Seeds:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].seeds}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Budget type:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].budget_type}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Budget:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].budget}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Workloads:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].workloads}
                        </Text>
                      </Col>
                    </Row>
                  </section>

                  <section style={{ marginBottom: '20px', borderBottom: '1px solid #e0e0e0', paddingBottom: '15px' }}>
                    <Title level={5} style={{ color: '#333', marginBottom: '10px' }}>
                      Algorithm Objects
                    </Title>
                    <Row gutter={[16, 8]}>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Narrow Search Space:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].SpaceRefiner}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Initialization:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].Sampler}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Pre-train:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].Pretrain}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Surrogate Model:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].Model}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Acquisition Function:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].ACF}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Normalizer:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].Normalizer}
                        </Text>
                      </Col>
                      {/* <Col span={24}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>DatasetSelector:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].DatasetSelector}
                        </Text>
                      </Col> */}
                    </Row>
                  </section>
                </div>
              </Card>

              {/* 图表区域 */}
              <Card
                bodyStyle={cardBodyStyle}
                style={cardStyle}
                title={
                  <div style={{ display: 'flex', alignItems: 'center' }}>
                    <AreaChartOutlined style={{ marginRight: '8px', color: '#1890ff' }} />
                    <Text strong>Visualization</Text>
                  </div>
                }
              >
                <div style={{ display: 'grid', gridTemplateColumns: 'repeat(3, 1fr)', gap: '16px' }}>
                  <div>
                    <Text strong style={{ display: 'block', marginBottom: '8px', textAlign: 'center' }}>
                      Convergence Trajectory
                    </Text>
                    <LineChart TrajectoryData={trajectoryData} />
                  </div>
                  <div>
                    <Text strong style={{ display: 'block', marginBottom: '8px', textAlign: 'center' }}>
                      Performance Analysis
                    </Text>
                    <BarChart ImportanceData={importance} />
                  </div>
                  <div>
                    <Text strong style={{ display: 'block', marginBottom: '8px', textAlign: 'center' }}>
                      Solution Space
                    </Text>
                    <Footprint ScatterData={scatterData} />
                  </div>
                </div>
              </Card>

              {/*Statistical Analysis*/}
            <StatisticalAnalysis />
            </div>
          )}
        </div>
      </div>

      {/* 错误提交弹窗 */}
      <Modal
        title="Submit Error"
        open={isModalVisible}
        onOk={handleOk}
        onCancel={handleCancel}
      >
        <Input.TextArea
          value={errorMessage}
          onChange={handleInputChange}
          placeholder="Please describe the error you encountered"
          style={{ width: '100%', marginBottom: '16px' }}
          rows={4}
        />
      </Modal>

      {/* 更多信息弹窗 */}
      <Modal
        title={
          <Space>
            <InfoCircleOutlined style={{ color: "#1890ff", fontSize: "18px" }} />
            <span>Detailed Information: {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].problem_name}</span>
          </Space>
        }
        open={isMoreInfoModalVisible}
        onCancel={handleMoreInfoCancel}
        width={1080}
        footer={[
          <AntButton key="close" onClick={handleMoreInfoCancel}>
            Close
          </AntButton>
        ]}
      >
        <div style={{ maxHeight: "80vh", overflowY: "auto", overflowX: "hidden" }}>
          <section style={{ marginBottom: '20px', borderBottom: '1px solid #e0e0e0', paddingBottom: '15px' }}>
            <Title level={5} style={{ color: '#333', marginBottom: '10px' }}>
              Problem Information
            </Title>
            <Row gutter={[16, 8]}>
              <Col span={24}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Problem Name:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].displayName}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Variable num:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].dim}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Objective num:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].obj}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Seeds:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].seeds}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Budget type:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].budget_type}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Budget:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].budget}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Workloads:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].workloads}
                </Text>
              </Col>
            </Row>
          </section>

          <section style={{ marginBottom: '20px', borderBottom: '1px solid #e0e0e0', paddingBottom: '15px' }}>
            <Title level={5} style={{ color: '#333', marginBottom: '10px' }}>
              Algorithm Objects
            </Title>
            <Row gutter={[16, 8]}>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Narrow Search Space:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].SpaceRefiner}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Initialization:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].Sampler}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Pre-train:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].Pretrain}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Surrogate Model:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].Model}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Acquisition Function:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].ACF}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Normalizer:</strong> {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].Normalizer}
                </Text>
              </Col>
            </Row>
          </section>

          <section>
            <Title level={5} style={{ color: '#333', marginBottom: '10px' }}>
              Auxilliary Data
            </Title>

            <Row gutter={[24, 16]}>
              <Col span={12}>
                <Card
                  size="small"
                  title={
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Narrow Search Space</span>
                      <span>
                      DatasetSelector-
                        {`${tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].AutoSelect.SearchSpace}`}
                      </span>
                    </div>
                  }
                  style={{ marginBottom: '10px' }}
                >
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].auxiliaryData?.SearchSpace.map((dataset, index) => (
                      <li key={index} style={{ fontSize: '0.9em' }}>{dataset}</li>
                    ))}
                  </ul>
                </Card>
              </Col>

              <Col span={12}>
                <Card
                  size="small"
                  title={
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Initialization</span>
                      <span>
                      DatasetSelector-
                        {`${tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].AutoSelect.Initialization}`}
                      </span>
                    </div>
                  }
                  style={{ marginBottom: '10px' }}
                >
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].auxiliaryData.Initialization.map((dataset, index) => (
                      <li key={index} style={{ fontSize: '0.9em' }}>{dataset}</li>
                    ))}
                  </ul>
                </Card>
              </Col>

              <Col span={12}>
                <Card
                  size="small"
                  title={
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Pre-train</span>
                      <span>
                        DatasetSelector-
                        {`${tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].AutoSelect.Pretrain}`}
                      </span>
                    </div>
                  }
                  style={{ marginBottom: '10px' }}
                >
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].auxiliaryData.Pretrain.map((dataset, index) => (
                      <li key={index} style={{ fontSize: '0.9em' }}>{dataset}</li>
                    ))}
                  </ul>
                </Card>
              </Col>

              <Col span={12}>
                <Card
                  size="small"
                  title={
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Surrogate Model</span>
                      <span>
                      DatasetSelector-
                        {`${tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].AutoSelect.Model}`}
                      </span>
                    </div>
                  }
                  style={{ marginBottom: '10px' }}
                >
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].auxiliaryData.Model.map((dataset, index) => (
                      <li key={index} style={{ fontSize: '0.9em' }}>{dataset}</li>
                    ))}
                  </ul>
                </Card>
              </Col>

              <Col span={12}>
                <Card
                  size="small"
                  title={
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Acquisition Function</span>
                      <span>
                      DatasetSelector-
                        {`${tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].AutoSelect.AcquisitionFunction}`}
                      </span>
                    </div>
                  }
                  style={{ marginBottom: '10px' }}
                >
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].auxiliaryData.AcquisitionFunction.map((dataset, index) => (
                      <li key={index} style={{ fontSize: '0.9em' }}>{dataset}</li>
                    ))}
                  </ul>
                </Card>
              </Col>

              <Col span={12}>
                <Card
                  size="small"
                  title={
                    <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                      <span>Normalizer</span>
                      <span>
                      DatasetSelector-
                        {`${tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].AutoSelect.Normalizer}`}
                      </span>
                    </div>
                  }
                >
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    {tasksInfo[selectedExperimentIndex].problemList[selectedTaskIndex].auxiliaryData.Normalizer.map((dataset, index) => (
                      <li key={index} style={{ fontSize: '0.9em' }}>{dataset}</li>
                    ))}
                  </ul>
                </Card>
              </Col>
            </Row>
          </section>
        </div>
      </Modal>
    </div>
  );
};

export default Dashboard;
