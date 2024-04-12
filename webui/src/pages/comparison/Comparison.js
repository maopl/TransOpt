import React from "react";

import {
  Row,
  Col,
  Button
} from "reactstrap";

import Widget from "../../components/Widget/Widget";

import s from "./Comparison.module.scss";

import Box from "./charts/Box";
import Trajectory from "./charts/Trajectory";
import SelectAlgorithm from "./component/SelectAlgorithm"

class Report extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      selectedTaskIndex: -1,
      TasksList: [],
      AlgorithmsList: [],
      selectedAlgorithms: [],
      BoxData: {},
      TrajectoryData: [],
    };

    this.checkAll = this.checkAll.bind(this);
  }

  checkAll(ev, checkbox) {
    const checkboxArr = new Array(this.state[checkbox].length).fill(
      ev.target.checked
    );
    this.setState({
      [checkbox]: checkboxArr,
    });
  }

  changeCheck(ev, checkbox, id) {
    //eslint-disable-next-line
    this.state[checkbox][id] = ev.target.checked;
    if (!ev.target.checked) {
      //eslint-disable-next-line
      this.state[checkbox][0] = false;
    }
    this.setState({
      [checkbox]: this.state[checkbox],
    });
  }

  handleTaskClick = (index) => {
    this.setState({ selectedTaskIndex: index });

    const messageToSend = {
      task: this.state.TasksList[index],
    }
    fetch('http://localhost:5000/api/comparison/choose_task', {
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
      this.setState({ AlgorithmsList: data });
    })
    .catch((error) => {
      console.error('Error sending message:', error);
    });
  }

  handleAlgorithmClick = (checkedList) => {
    // console.log(checkedList)
    const data = checkedList.map(item => {
      return item;
    });
    this.setState({selectedAlgorithms: data})
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
        taskname:this.state.TasksList[this.state.selectedTaskIndex],
        selectedAlgorithms: this.state.selectedAlgorithms
      }
      const response = await fetch('http://localhost:5000/api/comparison/charts', {
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
        BoxData: data.BoxData,  
        TrajectoryData: data.TrajectoryData
      })
      console.log("state:",Object.keys(this.state.BoxData).length)
    } catch (error) {
      console.error('Error fetching data:', error);
      // 在这里处理错误
    }
  };

  render() {
    if (this.state.TasksList.length === 0) {
      const messageToSend = {
        action: 'ask for tasks information',
      }
      fetch('http://localhost:5000/api/comparison/tasks', {
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
        this.setState({ selectedTaskIndex: 0,  TasksList: data });
      })
      .catch((error) => {
        console.error('Error sending message:', error);
      });

      return (
        <div className={s.root}>
          <h1 className="page-title">
            Report - <span className="fw-semi-bold">Comparison</span>
          </h1>
        </div>
      )
    } else {
      return (
        <div className={s.root}>
          <h1 className="page-title">
            Report - <span className="fw-semi-bold">Comparison</span>
          </h1>
          <div>
            <Row>
              <Col lg={12} xs={12}>
                <Widget
                  title={
                    <h5>
                      1. Choose <span className="fw-semi-bold">Task</span>
                    </h5>
                  }
                  collapse
                >
                  <div className="tasklist">
                    {this.state.TasksList.map((task, index) => (
                      <Button key={index} className={s.btn} onClick={() => this.handleTaskClick(index)}>
                        {task}
                      </Button>
                    ))}
                  </div>
                </Widget>
              </Col>
              <Col lg={12} xs={12}>
                <Row>
                  <Col lg={2} xs={4}>
                    <Widget
                      title={
                        <h5>
                          2. Choose <span className="fw-semi-bold">Algorithm</span>
                        </h5>
                      }
                      collapse
                    >
                    <SelectAlgorithm data={this.state.AlgorithmsList} handelClick={this.handleAlgorithmClick} />
                    </Widget>
                  </Col>
                  <Col lg={10} xs={8}>
                    <Row>
                      <Col lg={6} xs={12}>
                        <Widget
                          title={
                            <h5>
                              <span className="fw-semi-bold">Box</span>
                            </h5>
                          }
                          collapse
                        > 
                          <Box BoxData={this.state.BoxData}/>
                        </Widget>
                      </Col>
                      <Col lg={6} xs={12}>
                        <Widget
                          title={
                            <h5>
                              <span className="fw-semi-bold">Trajectory</span>
                            </h5>
                          }
                          collapse
                        >
                          <Trajectory TrajectoryData={this.state.TrajectoryData}/>
                        </Widget>
                      </Col>
                    </Row>
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
