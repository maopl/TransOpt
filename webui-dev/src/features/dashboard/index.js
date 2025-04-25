import React, { useState, useEffect, useCallback, useMemo } from "react";
import {
    Modal,
    Spin,
    Input,
    Form,
    Typography
} from "antd";
import {
    LoadingOutlined,
    InfoCircleOutlined
} from '@ant-design/icons';
import dayjs from 'dayjs';

// 导入重构后的组件
import ProblemFilter from "./components/ProblemFilter";
import ProblemList from "./components/ProblemList";
import ProblemDetails from "./components/ProblemDetails";

const { Text } = Typography;

// 基于搜索条件过滤实验和问题
const filterExperiments = (data, experimentName, problemName, dateRange) => {
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
    }).filter(experiment => experiment.filteredProblems.length > 0)
}

// 自定义灰色系图标
const antIcon = <LoadingOutlined style={{ fontSize: 48, color: '#9E9E9E' }} spin />;

// 获取任务列表（SWR实现）
const fetchTasks = async () => {
    const messageToSend = {
        action: 'ask for tasks information',
    };
    const response = await fetch('http://localhost:5001/api/Dashboard/tasks', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify(messageToSend),
    });
    if (!response.ok) {
        throw new Error('Network response was not ok');
    }
    const data = await response.json();
    return data.map(experiment => {
        const updatedProblemList = experiment.problemList.map(problem => {
            const parts = problem.problem_name.split('_');
            let displayName = problem.problem_name;
            if (parts.length > 0) {
                const timestamp = parts[parts.length - 1];
                if (/^\d+$/.test(timestamp)) {
                    try {
                        const date = new Date(Number(timestamp) * 1000);
                        const year = date.getFullYear();
                        const month = String(date.getMonth() + 1).padStart(2, '0');
                        const day = String(date.getDate()).padStart(2, '0');
                        const hours = String(date.getHours()).padStart(2, '0');
                        const minutes = String(date.getMinutes()).padStart(2, '0');
                        const seconds = String(date.getSeconds()).padStart(2, '0');
                        const formattedDate = `${year}-${month}-${day} ${hours}:${minutes}:${seconds}`;
                        const newParts = [...parts];
                        newParts[newParts.length - 1] = formattedDate;
                        displayName = newParts.join('_');
                    } catch (error) {
                        // ignore
                    }
                }
            }
            return {
                ...problem,
                displayName,
                experimentName: experiment.experimentName
            };
        });
        return {
            ...experiment,
            problemList: updatedProblemList
        };
    });
};

const Dashboard = () => {
    // 状态管理 - 优化为单一状态
    const [currentProblem, setCurrentProblem] = useState(null);
    const [isInitialLoading, setIsInitialLoading] = useState(true); // 首次加载状态
    const [deleteLoading, setDeleteLoading] = useState(false);
    // 搜索表单状态
    const [searchForm] = Form.useForm();
    const [searchExperimentName, setSearchExperimentName] = useState('');
    const [searchProblemName, setSearchProblemName] = useState('');
    const [searchDateRange, setSearchDateRange] = useState(null);

    // 全部的列表
    const [tasksInfo, setTasksInfo] = useState([]);

    // 过滤后的实验
    const [filteredExperiments, setFilteredExperiments] = useState([]);

    //获取任务列表
    const getTasksInfo = async (callback) => {
        const taskList = await fetchTasks();
        setTasksInfo(taskList);
        callback?.();
    }

    // 初始化获取任务列表
    useEffect(() => {
        getTasksInfo();
    }, []);


    // 处理搜索表单提交
    const handleSearch = (values) => {

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
        setFilteredExperiments(newFilteredExperiments);
        // 如果过滤后有结果，自动选择第一项
        if (newFilteredExperiments.length > 0) {
            const firstExp = newFilteredExperiments[0];
            if (firstExp.filteredProblems.length > 0) {
                // 找到第一个匹配的问题并设置为当前问题
                setCurrentProblem(firstExp.filteredProblems[0]);
            }
        } else {
            // 如果没有匹配结果，清除当前问题
            setCurrentProblem(null);
        }
    };

    // 重置搜索
    const handleResetSearch = () => {
        searchForm.resetFields();
        setSearchExperimentName('');
        setSearchProblemName('');
        setSearchDateRange(null);

        // 重置后如果已有数据，选择第一项
        if (tasksInfo.length > 0 && tasksInfo[0].problemList.length > 0) {
            setCurrentProblem(tasksInfo[0].problemList[0]);
        }
    };

    useEffect(() => {
        if (tasksInfo.length > 0) {
            const filteredExperiments = filterExperiments(tasksInfo, searchExperimentName, searchProblemName, searchDateRange);
            setFilteredExperiments(filteredExperiments);
            if (!currentProblem) {
                setCurrentProblem(filteredExperiments[0]?.filteredProblems[0] || null);
            }
            setIsInitialLoading(false);
        }
    }, [tasksInfo, searchExperimentName, searchProblemName, searchDateRange]);


    // 删除任务处理函数 - 使用SWR实现
    const handleDelete = async (taskName) => {
        // 设置全局 dashboard 加载状态
        setDeleteLoading(true);

        const messageToSend = {
            datasets: [taskName],
        }

        try {
            // 1. 先执行删除操作
            const response = await fetch('http://localhost:5001/api/configuration/delete_dataset', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify(messageToSend),
            });

            if (!response.ok) {
                throw new Error('Network response was not ok');
            }

            await response.json();

            fetchTasks();

            // 删除成功提示
            Modal.success({
                title: 'Information',
                content: 'Delete successfully!'
            });

            // 重新获取任务列表
            getTasksInfo(() => {
                // 只有在删除的是当前选中的问题时，才需要重新选择
                if (currentProblem && currentProblem.problem_name === taskName) {
                    // 当前选中的问题被删除了，需要重新选择
                    if (filteredExperiments.length > 0 && filteredExperiments[0].filteredProblems.length > 0) {
                        console.log('filteredExperiments', filteredExperiments)
                        setCurrentProblem(filteredExperiments[0].filteredProblems[0]);
                    } else {
                        setCurrentProblem(null);
                    }
                }
                // 如果删除的不是当前选中的问题，则保持当前选择不变
            })

        } catch (error) {
            console.error('Error sending message:', error);
            const errorMessage = error.message || 'Unknown error';
            Modal.error({
                title: 'Information',
                content: 'Error: ' + errorMessage
            });
        } finally {
            // 无论成功或失败，都关闭加载状态
            setDeleteLoading(false);
        }
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
    if (!tasksInfo.length) {
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

    return (

        <Spin icon={antIcon} spinning={deleteLoading}>
            <div style={{
                height: "calc(100vh - 120px)",
                padding: "20px",
                display: "flex",
                flexDirection: "column",
                overflow: "hidden",
                backgroundColor: "#f5f5f5"
            }}>
                {/* 顶部搜索表单 */}
                <ProblemFilter
                    form={searchForm}
                    onFilter={handleSearch}
                    onReset={handleResetSearch}
                />

                {/* 主内容区域 - 左右布局 */}
                <div style={{
                    display: "flex",
                    flex: "1",
                    overflow: "hidden",
                    height: "100%",
                    minHeight: "500px",
                    gap: "16px"
                }}>
                    {/* 左侧问题列表 */}
                    <ProblemList
                        filteredExperiments={filteredExperiments}
                        setCurrentProblem={setCurrentProblem}
                        currentProblem={currentProblem}
                        onDelete={handleDelete}
                    />

                    {/* 右侧内容区域 */}
                    <div style={{
                        flex: "1",
                        overflowX: "scroll",
                    }}>
                        {currentProblem ? (
                            <ProblemDetails
                                currentProblem={currentProblem}
                                onDelete={handleDelete}
                            />
                        ) : (
                            <div style={{
                                display: "flex",
                                justifyContent: "center",
                                alignItems: "center",
                                height: "100%",
                                flexDirection: "column"
                            }}>
                                <InfoCircleOutlined style={{ fontSize: 48, color: '#9E9E9E' }} />
                                <Text style={{ marginTop: "16px", color: "#9E9E9E", fontSize: "16px" }}>
                                    No problem selected. Please select a problem from the list.
                                </Text>
                            </div>
                        )}
                    </div>
                </div>
            </div>
        </Spin>
    );
};

export default Dashboard;
