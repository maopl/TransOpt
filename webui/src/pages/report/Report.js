import React from "react";

import { Row, Col, Button } from "reactstrap";

import s from "./Report.module.scss";
import Widget from "../../components/Widget/Widget";

import Trajectory from "./charts/Trajectory";
import Radar from "./charts/Radar";
import Scatter from "./charts/Scatter";
import Bar from "./charts/Bar";


class Report extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      selectedTaskIndex: -1,
      tasksInfo: [],
      BarData: [],
      RadarData: [],
      ScatterData: [],
      TrajectoryData: []
    };
  }

  // Select the corresponding task to display
  handleTaskClick = (index) => {
    console.log(index)
    this.setState({ selectedTaskIndex: index });
  }

  componentDidMount() {
    // 开始定时调用 fetchData 函数
    this.intervalId = setInterval(this.fetchData, 5000);
  }

  componentWillUnmount() {
    // 清除定时器，以防止内存泄漏
    clearInterval(this.intervalId);
  }

  fetchData = async () => {
    try {
      const messageToSend = {
        taskname:this.state.tasksInfo[this.state.selectedTaskIndex].name,
      }
      const response = await fetch('http://localhost:5000/api/report/charts', {
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
        BarData: data.BarData,
        RadarData: data.RadarData,
        ScatterData: data.ScatterData,
        TrajectoryData: data.TrajectoryData
      })
      // console.log('State:', this.state.BarData)
    } catch (error) {
      console.error('Error fetching data:', error);
      // 在这里处理错误
    }
  };

  render() { 
    // If first time rendering, then render the default task
    // If not, then render the task that was clicked
    if (this.state.selectedTaskIndex === -1) {
      // TODO: ask for task list from back-end
      const messageToSend = {
        action: 'ask for tasks information',
      }
      fetch('http://localhost:5000/api/report/tasks', {
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
        this.setState({ selectedTaskIndex: 0,  tasksInfo: data });
      })
      .catch((error) => {
        console.error('Error sending message:', error);
      });
      
      // Set the default task as the first task in the list
      return (
        <div className={s.root}>
          <h1 className="page-title">
            Report - <span className="fw-semi-bold">Tasks</span>
          </h1>
        </div>
      )
    } else {
      return (
        <div className={s.root}>
          <h1 className="page-title">
            Report - <span className="fw-semi-bold">Tasks</span>
          </h1>
          <div>
            <Row>
              <Col lg={2} xs={4}>
                <Row>
                  <Col lg={12} xs={12}>
                    <Widget
                      title={
                        <h5>
                          Choose <span className="fw-semi-bold">Task</span>
                        </h5>
                      }
                      collapse
                    >
                    {this.state.tasksInfo.map((task, index) => (
                      <Button
                        key={index}
                        className={s.btn}
                        onClick={() => this.handleTaskClick(index)}
                        >
                          {task.name}
                        </Button>
                    ))}
                    </Widget>
                  </Col>
                  <Col lg={12} xs={12}>
                    <Widget
                      title={
                        <h5>
                          <span className="fw-semi-bold">Information</span>
                        </h5>
                      }
                      collapse
                    >
                      <h4><strong>Task Name</strong></h4>
                      <h5>{this.state.tasksInfo[this.state.selectedTaskIndex].name}</h5>
                      <h4 className="mt-5"><strong>Auxiliary Data List</strong></h4>
                      <ul>
                        {this.state.tasksInfo[this.state.selectedTaskIndex].data.map((dataset, index) => (
                          <li key={index}><h5>{dataset}</h5></li>
                        ))}
                      </ul>
                      <h4 className="mt-5"><strong>Algorithm</strong></h4>
                      <ul>
                        {this.state.tasksInfo[this.state.selectedTaskIndex].algorithm.map((algo, index) => (
                          <li key={index}><h5>{algo}</h5></li>
                        ))}
                      </ul>
                    </Widget>
                  </Col>
                </Row>
              </Col>
              <Col lg={10} xs={8}>
                <Row>
                  <Col lg={12} xs={12}>
                    <Widget
                      title={
                        <h5>
                          <span className="fw-semi-bold">Optimization Trajectory</span>
                        </h5>
                      }
                      collapse
                    >
                      <Trajectory TrajectoryData={this.state.TrajectoryData}/>
                      {/* <Trajectory /> */}
                    </Widget>
                  </Col>
                  <Col lg={6} xs={12}>
                    <Widget
                      title={
                        <h5>
                            <span className="fw-semi-bold">Performance Metric</span>
                        </h5>
                      }
                      collapse
                    >
                      <Radar RadarData={this.state.RadarData}/>
                    </Widget>
                  </Col>
                  <Col lg={6} xs={12}>
                    <Widget
                      title={
                        <h5>
                            <span className="fw-semi-bold">Configuration Footprint</span>
                        </h5>
                      }
                      collapse
                    >
                      <Scatter ScatterData={this.state.ScatterData} />
                    </Widget>
                  </Col>
                  <Col lg={6} xs={12}>
                    <Widget
                      title={
                        <h5>
                          <span className="fw-semi-bold">Feature Importance</span>
                        </h5>
                      }
                      collapse
                    >
                      <Bar BarData={this.state.BarData} />
                    </Widget>
                  </Col>
                </Row>
              </Col>
            </Row>
          </div>
        </div>
      );
    }
  }

}

export default Report;