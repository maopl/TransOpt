import React from "react";
import {Card, Spin, Typography, Row, Col, Space} from "antd";
import {LoadingOutlined, InfoCircleOutlined, AreaChartOutlined} from '@ant-design/icons';
import useSWR from 'swr';
import LineChart from './LineChart';
import BarChart from './BarChart';
import Footprint from "./ScatterChart";
import StatisticalAnalysis from "./StatisticalAnalysis";
import ProblemMoreInfo from "./ProblemMoreInfo";
import DeleteProblem from "./DeleteProblem";

const {Text, Title} = Typography;

// 统一的卡片样式
const cardStyle = {
    borderRadius: "8px",
    boxShadow: "0 1px 2px -2px rgba(0, 0, 0, 0.16), 0 3px 6px 0 rgba(0, 0, 0, 0.12), 0 5px 12px 4px rgba(0, 0, 0, 0.09)"
};

// 统一的卡片内容样式
const cardBodyStyle = {padding: '16px'};

// SWR fetcher 函数
const fetcher = async (url, data) => {
    const response = await fetch(url, {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(data)
    });

    if (!response.ok) {
        throw new Error('Network response was not ok');
    }

    return response.json();
};

const ProblemDetails = ({
                            currentProblem,
                            onDelete
                        }) => {


    // 自定义加载图标
    const antIcon = <LoadingOutlined style={{fontSize: 48, color: '#9E9E9E'}} spin/>;



    // 使用SWR获取数据
    const {data, error, isLoading} = useSWR(
        currentProblem ? ['http://localhost:5001/api/Dashboard/charts', {taskname: currentProblem?.problem_name}] : null,
        ([url, data]) => fetcher(url, data),
        {
            revalidateOnFocus: false,
            dedupingInterval: 5000,
        }
    );
    console.log('currentProblem111', currentProblem)
    if (!currentProblem) {
        return (
            <div style={{
                display: "flex",
                width: "100%",
                justifyContent: "center",
                alignItems: "center",
                height: "300px",
                flexDirection: "column",
                backgroundColor: "white",
                borderRadius: "8px",
                ...cardStyle
            }}>
                <Spin indicator={antIcon}/>
                <span style={{marginTop: "16px", color: "#9E9E9E"}}>Loading data...</span>
            </div>
        )
    }

    // 提取数据
    const importance = data?.ImportanceData;
    const scatterData = data?.ScatterData;
    const trajectoryData = data?.TrajectoryData;

    // 如果请求出错
    if (error) {
        return (
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
                <InfoCircleOutlined style={{fontSize: 48, color: '#ff4d4f'}}/>
                <span style={{marginTop: "16px", color: "#ff4d4f"}}>Failed to load data</span>
            </div>
        );
    }

    return (
        <div style={{
            height: "100%",
            overflowY: "auto",
            paddingRight: "5px",
            // minWidth: "1120px",
        }}>
            {/* 详情卡片 */}
            <Card
                className="mb-4"
                style={cardStyle}
            >
                <div style={{padding: '8px'}}>
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
                            <InfoCircleOutlined style={{color: "#1890ff", fontSize: "18px"}}/>
                            <Title level={4} style={{margin: 0}}>
                                {currentProblem.displayName}
                            </Title>
                        </Space>

                        <Space>
                            <ProblemMoreInfo problem={currentProblem}/>
                            <DeleteProblem
                                problemName={currentProblem.problem_name}
                                onDelete={onDelete}
                            />
                        </Space>
                    </div>

                    <section style={{marginBottom: '20px', borderBottom: '1px solid #e0e0e0', paddingBottom: '15px'}}>
                        <Title level={5} style={{color: '#333', marginBottom: '10px'}}>
                            Problem Information
                        </Title>
                        <Row gutter={[16, 8]}>
                            <Col span={24}>
                                <Text style={{fontSize: '0.95em'}}>
                                    <strong>Problem Name:</strong> {currentProblem.displayName}
                                </Text>
                            </Col>
                            <Col span={8}>
                                <Text style={{fontSize: '0.95em'}}>
                                    <strong>Variable num:</strong> {currentProblem.dim}
                                </Text>
                            </Col>
                            <Col span={8}>
                                <Text style={{fontSize: '0.95em'}}>
                                    <strong>Objective num:</strong> {currentProblem.obj}
                                </Text>
                            </Col>
                            <Col span={8}>
                                <Text style={{fontSize: '0.95em'}}>
                                    <strong>Seeds:</strong> {currentProblem.seeds}
                                </Text>
                            </Col>
                            <Col span={8}>
                                <Text style={{fontSize: '0.95em'}}>
                                    <strong>Budget type:</strong> {currentProblem.budget_type}
                                </Text>
                            </Col>
                            <Col span={8}>
                                <Text style={{fontSize: '0.95em'}}>
                                    <strong>Budget:</strong> {currentProblem.budget}
                                </Text>
                            </Col>
                            <Col span={8}>
                                <Text style={{fontSize: '0.95em'}}>
                                    <strong>Workloads:</strong> {currentProblem.workloads}
                                </Text>
                            </Col>
                        </Row>
                    </section>

                    <section style={{marginBottom: '20px', borderBottom: '1px solid #e0e0e0', paddingBottom: '15px'}}>
                        <Title level={5} style={{color: '#333', marginBottom: '10px'}}>
                            Algorithm Objects
                        </Title>
                        <Row gutter={[16, 8]}>
                            <Col span={8}>
                                <Text style={{fontSize: '0.95em'}}>
                                    <strong>Narrow Search Space:</strong> {currentProblem.SearchSpace}
                                </Text>
                            </Col>
                            <Col span={8}>
                                <Text style={{fontSize: '0.95em'}}>
                                    <strong>Initialization:</strong> {currentProblem.Initialization}
                                </Text>
                            </Col>
                            <Col span={8}>
                                <Text style={{fontSize: '0.95em'}}>
                                    <strong>Pre-train:</strong> {currentProblem.Pretrain}
                                </Text>
                            </Col>
                            <Col span={8}>
                                <Text style={{fontSize: '0.95em'}}>
                                    <strong>Surrogate Model:</strong> {currentProblem.Model}
                                </Text>
                            </Col>
                            <Col span={8}>
                                <Text style={{fontSize: '0.95em'}}>
                                    <strong>Acquisition Function:</strong> {currentProblem.AcquisitionFunction}
                                </Text>
                            </Col>
                            <Col span={8}>
                                <Text style={{fontSize: '0.95em'}}>
                                    <strong>Normalizer:</strong> {currentProblem.Normalizer}
                                </Text>
                            </Col>
                        </Row>
                    </section>
                </div>
            </Card>

            {/* 图表区域 */}
            <Card
                bodyStyle={cardBodyStyle}
                style={cardStyle}
                title={
                    <div style={{display: 'flex', alignItems: 'center'}}>
                        <AreaChartOutlined style={{marginRight: '8px', color: '#1890ff'}}/>
                        <Text strong>Visualization</Text>
                    </div>
                }
            >
                {isLoading ? <div style={{
                        display: "flex",
                        width: "100%",
                        justifyContent: "center",
                        alignItems: "center",
                        height: "300px",
                        flexDirection: "column",
                        backgroundColor: "white",
                        borderRadius: "8px",
                        ...cardStyle
                    }}>
                        <Spin indicator={antIcon}/>
                        <span style={{marginTop: "16px", color: "#9E9E9E"}}>Loading data...</span>
                    </div>
                    :
                    <div style={{
                        display: 'grid',
                        gridTemplateColumns: 'repeat(auto-fill, minmax(350px, 1fr))',
                        gap: '16px',
                        width: '100%',
                        justifyContent: 'center',
                        justifyItems: 'center'
                    }}>
                        <div>
                            <Text strong style={{display: 'block', marginBottom: '8px', textAlign: 'center'}}>
                                Convergence Trajectory
                            </Text>
                            <LineChart TrajectoryData={trajectoryData}/>
                        </div>
                        <div>
                            <Text strong style={{display: 'block', marginBottom: '8px', textAlign: 'center'}}>
                                Performance Analysis
                            </Text>
                            <BarChart ImportanceData={importance}/>
                        </div>
                        <div>
                            <Text strong style={{display: 'block', marginBottom: '8px', textAlign: 'center'}}>
                                Solution Space
                            </Text>
                            <Footprint ScatterData={scatterData}/>
                        </div>
                    </div>}
            </Card>

            {/* Statistical Analysis */}
            <StatisticalAnalysis/>
        </div>
    );
};

export default ProblemDetails;