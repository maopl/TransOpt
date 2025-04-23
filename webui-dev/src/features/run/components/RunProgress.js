import React, { useState, useEffect, useRef } from "react";
import { MinusCircleOutlined } from '@ant-design/icons'
import {
    Progress,
    ConfigProvider
} from "antd";

const RunProgress = ({ isRunning = false }) => {
    // 使用useState代替类组件中的state
    const [twoColors] = useState({
        '0%': '#108ee9',
        '100%': '#87d068',
    });
    const [data, setData] = useState([]);
    
    // 使用useRef存储intervalId，以便在清理函数中访问最新值
    const intervalIdRef = useRef(null);

    // 处理停止任务的请求
    const handleClick = (task_name) => {
        const messageToSend = {
            name: task_name
        }
        fetch('http://localhost:5001/api/configuration/stop_progress', {
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
        })
        .catch((error) => {
            console.error('Error sending message:', error);
        });
    };

    // 获取进度数据的函数
    const fetchProgressData = () => {
        fetch('http://localhost:5001/api/RunPage/get_progress', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({}),
        })
        .then(response => {
            if (!response.ok) {
                throw new Error('Network response was not ok');
            }
            return response.json();
        })
        .then(progressData => {
            setData(progressData || []);
        })
        .catch(error => {
            console.error('Error fetching progress data:', error);
        });
    };

    // 监听isRunning状态的变化，控制定时器的启动和停止
    useEffect(() => {
        // 只有当isRunning为true时才启动定时器
        if (isRunning) {
            console.log('Starting progress polling...');
            
            // 立即获取一次初始数据
            fetchProgressData();
            
            // 设置定时器，定期获取进度数据
            intervalIdRef.current = setInterval(fetchProgressData, 1000);
            
            // 清理函数，在组件卸载或isRunning变为false时执行
            return () => {
                console.log('Stopping progress polling...');
                if (intervalIdRef.current) {
                    clearInterval(intervalIdRef.current);
                    intervalIdRef.current = null;
                }
            };
        }
    }, [isRunning]); // 依赖项包括isRunning，确保其变化时重新执行effect

    return (
        <ConfigProvider
            theme={{
                token:{
                    colorText: "#696969"
                },
                components: {
                    Progress: {
                        remainingColor: "#696969"
                    },
                },
            }}
        >
            <div style={{ overflowY: 'auto', maxHeight: '200px', maxWidth: '100%' }}>
                {data.map((task, index) => (
                    <div key={index} style={{ marginBottom: 10 }}>
                        <h6>{task.name}</h6>
                        <Progress 
                            percent={task.progress} 
                            status="active" 
                            type="line" 
                            strokeColor={twoColors} 
                            style={{ width:"93%", marginRight:25}} 
                        />
                        <MinusCircleOutlined 
                            style={{color: 'red'}} 
                            onClick={() => handleClick(task.name)} 
                        />
                    </div>
                ))}
            </div>
        </ConfigProvider>
    );
};

export default RunProgress;