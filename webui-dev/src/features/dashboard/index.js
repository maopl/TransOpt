import React, {useState, useEffect, useCallback, useMemo} from "react";
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

const {Text} = Typography;

// 基于搜索条件过滤实验和问题
const filterExperiments = (data, experimentName, problemName, dateRange) => {
    if (!data || !data.length) return [];

    return data.map(experiment => {
        // 实验名称过滤
        const experimentNameMatch = !experimentName ||
            experiment.experimentName.toLowerCase().includes(experimentName.toLowerCase());

        if (!experimentNameMatch) return {...experiment, filteredProblems: []};

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
const antIcon = <LoadingOutlined style={{fontSize: 48, color: '#9E9E9E'}} spin/>;
const Dashboard = () => {
    // 状态管理 - 优化为单一状态
    const [tasksInfo, setTasksInfo] = useState([]);
    const [currentProblem, setCurrentProblem] = useState(null);
    const [isModalVisible, setIsModalVisible] = useState(false);
    const [errorMessage, setErrorMessage] = useState("");
    const [isInitialLoading, setIsInitialLoading] = useState(true); // 首次加载状态

    // 搜索表单状态
    const [searchForm] = Form.useForm();
    const [searchExperimentName, setSearchExperimentName] = useState('');
    const [searchProblemName, setSearchProblemName] = useState('');
    const [searchDateRange, setSearchDateRange] = useState(null);
    // 过滤后的任务列表
    const filteredExperiments = useMemo(() => filterExperiments(tasksInfo, searchExperimentName, searchProblemName, searchDateRange),
        [tasksInfo, searchExperimentName, searchProblemName, searchDateRange]);


    // 处理搜索表单提交
    const handleSearch = (values) => {

        const {experimentName, problemName, dateRange} = values;

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

    // 获取任务列表
    useEffect(() => {
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
                            displayName,
                            experimentName: experiment.experimentName // 添加实验名称到问题对象中，便于后续处理
                        };
                    });

                    return {
                        ...experiment,
                        problemList: updatedProblemList
                    };
                });

                setTasksInfo(experimentsData);

                setIsInitialLoading(false);
            })
            .catch((error) => {
                console.error('Error sending message:', error);
                setIsInitialLoading(false);
            });
    }, []);

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

    // 删除任务处理函数 - 优化选择逻辑
    const handleDelete = (taskName) => {

        const messageToSend = {
            datasets: [taskName],
        }
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

                if (taskName === currentProblem.problem_name) {
                    // todo 另外选一个
                }

                Modal.success({
                    title: 'Information',
                    content: 'Delete successfully!'
                });
            })
            .catch((error) => {
                console.error('Error sending message:', error);
                const errorMessage = error.message || 'Unknown error';
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
                <Spin indicator={antIcon}/>
                <span style={{marginTop: "16px", color: "#9E9E9E"}}>Loading tasks...</span>
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
                <InfoCircleOutlined style={{fontSize: 48, color: '#9E9E9E'}}/>
                <Text style={{marginTop: "16px", color: "#9E9E9E", fontSize: "16px"}}>
                    No tasks found. Please create a task first.
                </Text>
            </div>
        );
    }



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
            <ProblemFilter
                form={searchForm}
                onFilter={handleSearch}
                onReset={handleResetSearch}
            />

            {/* 主内容区域 - 左右布局 */}
            <div style={{
                display: "flex",
                overflow: "hidden",
                minHeight: 0,
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
                            <InfoCircleOutlined style={{fontSize: 48, color: '#9E9E9E'}}/>
                            <Text style={{marginTop: "16px", color: "#9E9E9E", fontSize: "16px"}}>
                                No problem selected. Please select a problem from the list.
                            </Text>
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
                    style={{width: '100%', marginBottom: '16px'}}
                    rows={4}
                />
            </Modal>
        </div>
    );
};

export default Dashboard;
