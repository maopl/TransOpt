import React, { useState } from "react";

import {
    Progress,
    ConfigProvider
} from "antd";


class RunProgress extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            twoColors: {
                '0%': '#108ee9',
                '100%': '#87d068',
            },
            data: []
        }
    }
    // 与后端交互，获取任务进度
    componentDidMount() {
        // 开始定时调用 fetchData 函数
        this.intervalId = setInterval(this.fetchData, 1000);
      }
    
      componentWillUnmount() {
        // 清除定时器，以防止内存泄漏
        clearInterval(this.intervalId);
      }
    
      fetchData = async () => {
        try {
          const messageToSend = {
            message:"ask for progress"
          }
          const response = await fetch('http://localhost:5000/api/configuration/run_progress', {
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
          // 在这里处理从服务器获取的数据
          this.setState({
            data: data
          })
          // console.log('State:', this.state.BarData)
        } catch (error) {
          console.error('Error fetching data:', error);
        }
      };

      render() {
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
                <div style={{ overflowY: 'auto', maxHeight: '200px' }}>
                    {this.state.data.map((task, index) => (
                        <div key={index} style={{ marginBottom: 10 }}>
                            <h6>{task.name}</h6>
                            <Progress percent={task.progress} status="active" type="line" strokeColor={this.state.twoColors} />
                        </div>
                    ))}
                </div>
            </ConfigProvider>
        )
      }
}



export default RunProgress;