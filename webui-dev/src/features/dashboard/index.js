import React, { useState, useEffect, useCallback } from "react";
import { LineChartOutlined } from '@ant-design/icons';
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
} from "antd";
import {
  LoadingOutlined,
  SearchOutlined,
  FilterOutlined,
  SortAscendingOutlined,
  InfoCircleOutlined,
  DatabaseOutlined,
  AreaChartOutlined,
  ArrowRightOutlined
} from '@ant-design/icons';

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

// 模拟数据
const mockAlgorithms = [
  { value: 'all', label: 'All Algorithms' },
  { value: 'bo', label: 'Bayesian Optimization' },
  { value: 'gp', label: 'Gaussian Process' },
  { value: 'rf', label: 'Random Forest' },
  { value: 'nn', label: 'Neural Network' }
];

const mockCategories = [
  { value: 'all', label: 'All Categories' },
  { value: 'optimization', label: 'Optimization' },
  { value: 'regression', label: 'Regression' },
  { value: 'classification', label: 'Classification' }
];

const Dashboard = () => {
  // 状态管理
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

  // 搜索表单状态
  const [searchForm] = Form.useForm();
  const [searchKeyword, setSearchKeyword] = useState('');
  const [searchAlgorithm, setSearchAlgorithm] = useState('all');
  const [searchCategory, setSearchCategory] = useState('all');

  // 基于搜索条件过滤任务
  const filterTasks = useCallback((tasks) => {
    if (!tasks) return [];

    return tasks.filter(task => {
      // 关键词搜索
      const keywordMatch = !searchKeyword ||
        task.problem_name.toLowerCase().includes(searchKeyword.toLowerCase());

      // 算法筛选
      const algorithmMatch = searchAlgorithm === 'all' ||
        task.Model === searchAlgorithm ||
        task.ACF === searchAlgorithm ||
        task.SpaceRefiner === searchAlgorithm;

      // 分类筛选 (模拟，实际应用需要根据真实数据结构调整)
      const categoryMatch = searchCategory === 'all';

      return keywordMatch && algorithmMatch && categoryMatch;
    });
  }, [searchKeyword, searchAlgorithm, searchCategory]);

  // 自定义灰色系图标
  const antIcon = <LoadingOutlined style={{ fontSize: 48, color: '#9E9E9E' }} spin />;

  // 获取任务列表
  useEffect(() => {
    if (selectedTaskIndex === -1) {
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
          setTasksInfo(data);
          setSelectedTaskIndex(0);
          setIsInitialLoading(false); // 加载完成后设置为false
        })
        .catch((error) => {
          console.error('Error sending message:', error);
          setIsInitialLoading(false); // 出错时也设置为false
        });
    }
  }, [selectedTaskIndex]);

  // 定时获取数据
  useEffect(() => {
    // 如果没有选择任务，不执行
    if (selectedTaskIndex === -1 || !tasksInfo.length) return;

    const intervalId = setInterval(fetchData, 1000000);

    // 组件卸载时清除定时器
    return () => clearInterval(intervalId);
  }, [selectedTaskIndex, tasksInfo]);

  // 获取轨迹数据
  const fetchData = useCallback(async () => {
    // 如果没有选择任务，不执行
    if (selectedTaskIndex === -1 || !tasksInfo.length) return;

    try {
      const messageToSend = {
        taskname: tasksInfo[selectedTaskIndex].problem_name,
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
  }, [selectedTaskIndex, tasksInfo]);

  // 处理任务选择
  const handleTaskClick = useCallback((index) => {
    // 如果点击的是已选中的任务，则不执行
    if (selectedTaskIndex === index) return;

    console.log('Selected task index:', index);
    setSelectedTaskIndex(index);
    setIsLoading(true);

    const messageToSend = {
      taskname: tasksInfo[index].problem_name,
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
  }, [selectedTaskIndex, tasksInfo]);

  // 处理搜索表单提交
  const handleSearch = useCallback((values) => {
    setSearchKeyword(values.keyword || '');
    setSearchAlgorithm(values.algorithm || 'all');
    setSearchCategory(values.category || 'all');
    // 重置选中的任务索引
    if (tasksInfo.length > 0) {
      setSelectedTaskIndex(0);
    }
  }, [tasksInfo]);

  // 重置搜索条件
  const handleResetSearch = () => {
    searchForm.resetFields();
    setSearchKeyword('');
    setSearchAlgorithm('all');
    setSearchCategory('all');
    // 重置选中的任务索引
    if (tasksInfo.length > 0) {
      setSelectedTaskIndex(0);
    }
  };

  // 错误提交相关函数
  const showModal = () => setIsModalVisible(true);

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
  if (selectedTaskIndex === -1 || !tasksInfo.length) {
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
  const filteredTasks = filterTasks(tasksInfo);

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
            <Col span={5}>
              <Form.Item name="keyword" style={{ marginBottom: 0 }}>
                <Input
                  placeholder="Search by name"
                  prefix={<SearchOutlined />}
                  allowClear
                  size="middle"
                />
              </Form.Item>
            </Col>
            <Col span={4}>
              <Form.Item name="algorithm" style={{ marginBottom: 0 }}>
                <Select
                  placeholder="Algorithm"
                  defaultValue="all"
                  style={{ width: "100%" }}
                  size="middle"
                >
                  {mockAlgorithms.map(item => (
                    <Option key={item.value} value={item.value}>{item.label}</Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
            <Col span={4}>
              <Form.Item name="category" style={{ marginBottom: 0 }}>
                <Select
                  placeholder="Category"
                  defaultValue="all"
                  style={{ width: "100%" }}
                  size="middle"
                >
                  {mockCategories.map(item => (
                    <Option key={item.value} value={item.value}>{item.label}</Option>
                  ))}
                </Select>
              </Form.Item>
            </Col>
            <Col span={7}>
              <Form.Item name="dateRange" style={{ marginBottom: 0 }}>
                <RangePicker style={{ width: "100%" }} size="middle" />
              </Form.Item>
            </Col>
            <Col span={4}>
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
          width: "20%",
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
                <Text strong>{filteredTasks.length} Results</Text>
              </Space>
              <AntButton
                type="text"
                size="small"
                icon={<SortAscendingOutlined />}
                title="Sort by name"
              />
            </div>

            {/* 这个div是专门用于滚动的容器 */}
            <div style={{
              overflowY: "auto",
              flex: "1 1 auto",
              paddingRight: "10px",
              minHeight: 0, // 关键: Ein mub flex子项收缩到小于内容高度
              marginBottom: "10px" // 防止内容太靠近底部
            }}>
              {filteredTasks.map((task, index) => (
                <Button
                  key={index}
                  onClick={() => handleTaskClick(index)}
                  className="w-100 text-start d-flex align-items-center"
                  style={{
                    backgroundColor: selectedTaskIndex === index ? '#f0f7ff' : 'transparent',
                    color: selectedTaskIndex === index ? '#1890ff' : 'rgba(0, 0, 0, 0.65)',
                    fontWeight: selectedTaskIndex === index ? '500' : 'normal',
                    borderLeft: selectedTaskIndex === index ? '3px solid #1890ff' : '3px solid transparent',
                    padding: "12px",
                    paddingLeft: "16px",
                    marginBottom: "6px",
                    borderRadius: "2px",
                    transition: 'all 0.2s ease',
                    borderTop: 'none',
                    borderRight: 'none',
                    borderBottom: 'none',
                    boxShadow: selectedTaskIndex === index ? '0 2px 8px rgba(24, 144, 255, 0.1)' : 'none',
                    position: 'relative',
                    overflow: 'hidden',
                    width: '100%',
                    display: 'block',
                    textAlign: 'left'
                  }}
                  onMouseEnter={(e) => {
                    if (selectedTaskIndex !== index) {
                      // e.currentTarget.style.backgroundColor = '#f5f5f5';
                      e.currentTarget.style.transform = 'translateY(-2px)';
                      e.currentTarget.style.boxShadow = '0 6px 16px -2px rgba(0, 0, 0, 0.15), 0 -6px 16px -2px rgba(0, 0, 0, 0.15), 8px 0 16px -8px rgba(0, 0, 0, 0.1), -8px 0 16px -8px rgba(0, 0, 0, 0.1)';
                      e.currentTarget.style.zIndex = '1';
                    }
                  }}
                  onMouseLeave={(e) => {
                    if (selectedTaskIndex !== index) {
                      e.currentTarget.style.backgroundColor = 'transparent';
                      e.currentTarget.style.transform = 'translateY(0)';
                      e.currentTarget.style.boxShadow = 'none';
                      e.currentTarget.style.zIndex = '0';
                    }
                  }}
                >
                  <div style={{ display: 'flex', alignItems: 'center', width: '100%' }}>
                    {selectedTaskIndex === index && (
                      <span style={{ 
                        marginRight: '8px', 
                        color: '#1890ff',
                        fontSize: '10px',
                        position: 'relative',
                        top: '-1px'
                      }}>■</span>
                    )}
                    <span style={{ overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                      {task.problem_name}
                    </span>
                  </div>
                </Button>
              ))}
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
                        {tasksInfo[selectedTaskIndex].problem_name}
                      </Title>
                    </Space>

                    <AntButton
                      type="primary"
                      icon={<ArrowRightOutlined />}
                      onClick={showMoreInfoModal}
                    >
                      More Info
                    </AntButton>
                  </div>

                  <section style={{ marginBottom: '20px', borderBottom: '1px solid #e0e0e0', paddingBottom: '15px' }}>
                    <Title level={5} style={{ color: '#333', marginBottom: '10px' }}>
                      Problem Information
                    </Title>
                    <Row gutter={[16, 8]}>
                      <Col span={24}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Problem Name:</strong> {tasksInfo[selectedTaskIndex].problem_name}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Variable num:</strong> {tasksInfo[selectedTaskIndex].dim}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Objective num:</strong> {tasksInfo[selectedTaskIndex].obj}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Seeds:</strong> {tasksInfo[selectedTaskIndex].seeds}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Budget type:</strong> {tasksInfo[selectedTaskIndex].budget_type}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Budget:</strong> {tasksInfo[selectedTaskIndex].budget}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Workloads:</strong> {tasksInfo[selectedTaskIndex].workloads}
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
                          <strong>Narrow Search Space:</strong> {tasksInfo[selectedTaskIndex].SpaceRefiner}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Initialization:</strong> {tasksInfo[selectedTaskIndex].Sampler}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Pre-train:</strong> {tasksInfo[selectedTaskIndex].Pretrain}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Surrogate Model:</strong> {tasksInfo[selectedTaskIndex].Model}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Acquisition Function:</strong> {tasksInfo[selectedTaskIndex].ACF}
                        </Text>
                      </Col>
                      <Col span={8}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>Normalizer:</strong> {tasksInfo[selectedTaskIndex].Normalizer}
                        </Text>
                      </Col>
                      {/* <Col span={24}>
                        <Text style={{ fontSize: '0.95em' }}>
                          <strong>DatasetSelector:</strong> {tasksInfo[selectedTaskIndex].DatasetSelector}
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
            <span>Detailed Information: {tasksInfo[selectedTaskIndex]?.problem_name}</span>
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
                  <strong>Problem Name:</strong> {tasksInfo[selectedTaskIndex].problem_name}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Variable num:</strong> {tasksInfo[selectedTaskIndex].dim}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Objective num:</strong> {tasksInfo[selectedTaskIndex].obj}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Seeds:</strong> {tasksInfo[selectedTaskIndex].seeds}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Budget type:</strong> {tasksInfo[selectedTaskIndex].budget_type}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Budget:</strong> {tasksInfo[selectedTaskIndex].budget}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Workloads:</strong> {tasksInfo[selectedTaskIndex].workloads}
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
                  <strong>Narrow Search Space:</strong> {tasksInfo[selectedTaskIndex].SpaceRefiner}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Initialization:</strong> {tasksInfo[selectedTaskIndex].Sampler}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Pre-train:</strong> {tasksInfo[selectedTaskIndex].Pretrain}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Surrogate Model:</strong> {tasksInfo[selectedTaskIndex].Model}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Acquisition Function:</strong> {tasksInfo[selectedTaskIndex].ACF}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Normalizer:</strong> {tasksInfo[selectedTaskIndex].Normalizer}
                </Text>
              </Col>
              <Col span={24}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>DatasetSelector:</strong> {tasksInfo[selectedTaskIndex].DatasetSelector}
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
                        {tasksInfo[selectedTaskIndex].DatasetSelector
                          ? tasksInfo[selectedTaskIndex].DatasetSelector.split(',')[0].split('-')[1]
                          : ''}
                      </span>
                    </div>
                  }
                  style={{ marginBottom: '10px' }}
                >
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    {tasksInfo[selectedTaskIndex].metadata.SpaceRefiner.map((dataset, index) => (
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
                        {tasksInfo[selectedTaskIndex].DatasetSelector
                          ? tasksInfo[selectedTaskIndex].DatasetSelector.split(',')[1].split('-')[1]
                          : ''}
                      </span>
                    </div>
                  }
                  style={{ marginBottom: '10px' }}
                >
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    {tasksInfo[selectedTaskIndex].metadata.Sampler.map((dataset, index) => (
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
                        {tasksInfo[selectedTaskIndex].DatasetSelector
                          ? tasksInfo[selectedTaskIndex].DatasetSelector.split(',')[2].split('-')[1]
                          : ''}
                      </span>
                    </div>
                  }
                  style={{ marginBottom: '10px' }}
                >
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    {tasksInfo[selectedTaskIndex].metadata.Pretrain.map((dataset, index) => (
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
                        {tasksInfo[selectedTaskIndex].DatasetSelector
                          ? tasksInfo[selectedTaskIndex].DatasetSelector.split(',')[3].split('-')[1]
                          : ''}
                      </span>
                    </div>
                  }
                  style={{ marginBottom: '10px' }}
                >
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    {tasksInfo[selectedTaskIndex].metadata.Model.map((dataset, index) => (
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
                        {tasksInfo[selectedTaskIndex].DatasetSelector
                          ? tasksInfo[selectedTaskIndex].DatasetSelector.split(',')[4].split('-')[1]
                          : ''}
                      </span>
                    </div>
                  }
                  style={{ marginBottom: '10px' }}
                >
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    {tasksInfo[selectedTaskIndex].metadata.ACF.map((dataset, index) => (
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
                        {tasksInfo[selectedTaskIndex].DatasetSelector
                          ? tasksInfo[selectedTaskIndex].DatasetSelector.split(',')[5].split('-')[1]
                          : ''}
                      </span>
                    </div>
                  }
                >
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    {tasksInfo[selectedTaskIndex].metadata.Normalizer.map((dataset, index) => (
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
