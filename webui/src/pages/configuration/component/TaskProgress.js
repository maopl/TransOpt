import React, { useState } from "react";

import {
    Progress
} from "antd";
import Trajectory from "../../report/charts/Trajectory";


class TaskProgress extends React.Component {
    constructor(props) {
        super(props);
        this.state = {
            twoColors: {
                '0%': '#108ee9',
                '100%': '#87d068',
            },
            problem_name: "",
            task_progress: 0,
            TrajectoryData: []
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
          const response = await fetch('http://localhost:5000/api/configuration/progress', {
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
            problem_name: data.problem_name,
            progress: data.progress,
            TrajectoryData: data.TrajectoryData
          })
          // console.log('State:', this.state.BarData)
        } catch (error) {
          console.error('Error fetching data:', error);
        }
      };

      render() {
        return (
            <div>
                <div style={{ marginBottom: 10 }}>
                    <h6>{this.state.problem_name}</h6>
                    <Progress percent={this.state.progress} type="line" strokeColor={this.state.twoColors}/>
                </div>
                <Trajectory TrajectoryData={this.state.TrajectoryData}/>
            </div>
        )
      }
}



export default TaskProgress;