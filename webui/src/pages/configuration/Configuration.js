import React from "react";
import {
  Row,
  Col,
} from "reactstrap";
import { Button, Modal } from "antd";

import s from "./Configuration.module.scss"

import Widget from "../../components/Widget/Widget";

import SelectTask from "./component/SelectTask";
import SelectAlgorithm from "./component/SelectAlgorithm";
import ChatUI from "./component/ChatUI";
import SelectData from "./component/SelectData";
import SearchData from "./component/SearchData"

import TasksData from './data/TasksData.json'
import AlgorithmsData from './data/AlgorithmsData.json'


class Configuration extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      TasksData: [],
      AlgorithmsData: [],
      DatasetData: []
    };
  }

  set_dataset = (datasets) => {
    console.log(datasets)
    this.setState({ DatasetData: datasets })
  }

  handelRunClick = () => {
    const messageToSend = {
      message: 'run the experiment',
    };
    console.log(messageToSend)
    fetch('http://localhost:5000/api/configuration/run', {
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
      Modal.success({
        title: 'Infor',
        content: 'Run successfully!'
      })
    })
    .catch((error) => {
      console.error('Error sending message:', error);
    });
  }

  render() {
    if (this.state.TasksData.length === 0 || this.state.AlgorithmsData.length === 0) {
      const messageToSend = {
        action: 'ask for basic information',
      }
      fetch('http://localhost:5000/api/configuration/basic_information', {
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
        this.setState({ TasksData: data.TasksData,  AlgorithmsData: data.AlgorithmsData });
      })
      .catch((error) => {
        console.error('Error sending message:', error);
      });

      return (
        <div className={s.root}>
          <h1 className="page-title">
            Experiment - <span className="fw-semi-bold">Configuration</span>
          </h1>
        </div>
      )
    } else {
      return (
        <div className={s.root}>
          <h1 className="page-title">
            Experiment - <span className="fw-semi-bold">Configuration</span>
          </h1>
          <Row>
            <Col lg={6} sm={8}>
            <Row>
              <Col lg={12} sm={12}>
                <Widget
                  title={
                    <h5>
                      1.Choose <span className="fw-semi-bold">Task</span>
                    </h5>
                  }
                  collapse
                >
                  <h3>
                    List-<span className="fw-semi-bold">Task</span>
                  </h3>
                  <p>
                  Click "Add Task" to select an optimization task, and then choose the name, dimensions, number of objectives, fidelity, and budget for the task. 
                  The budget can be specified in two forms: (1) evaluation count, for example, "1000", or (2) evaluation time, for example, "1d2h3m4s". 
                  Click "-" to delete the added task. Once the selection is complete, click the "Submit" button.
                  </p>
                  <SelectTask data={TasksData}/>
                </Widget>
              </Col>
              <Col lg={12} sm={12}>
                <Widget
                  title={
                    <h5>
                      2. Choose <span className="fw-semi-bold">Algorithms</span>
                    </h5>
                  }
                  collapse
                >
                  <h3>
                    List-<span className="fw-semi-bold">Algorithms</span>
                  </h3>
                  <p>
                    There are some discription.There are some discription.There are some discription.There are some discription.
                  </p>
                  <SelectAlgorithm data={AlgorithmsData}/>
                </Widget>
              </Col>
              <Col lg={12} sm={12}> 
                <Widget
                  title={
                    <h5>
                      3. Choose <span className="fw-semi-bold">Datasets</span>
                    </h5>
                  }
                  collapse
                >
                  <h3>
                    <span className="fw-semi-bold">Search</span>
                  </h3>
                  <p>
                    There are some discription.There are some discription.There are some discription.There are some discription.
                  </p>
                  <SearchData set_dataset={this.set_dataset}/>
                  <h3 className="mt-5">
                    <span className="fw-semi-bold">Choose</span>
                  </h3>
                  <p>
                    There are some discription.There are some discription.There are some discription.There are some discription.
                  </p>
                  <SelectData data={this.state.DatasetData}/>
                </Widget>
              </Col>
              <Col lg={12} sm={12}>
                <Widget
                  title={
                    <h5>
                      4. <span className="fw-semi-bold">Run</span>
                    </h5>
                  }
                  collapse
                >
                  <h3>
                    <span className="fw-semi-bold">Run the Experiment</span>
                  </h3>
                  <p>
                    Click the "run" button to start the experiment.
                  </p>
                  <Button type="primary" htmlType="submit" style={{width:"120px"}} onClick={this.handelRunClick}>
                    Run
                  </Button>
                </Widget>
              </Col>
            </Row>
            </Col>
            <Col lg={6} sm={4}>
              <Widget
                title={
                  <h5>
                    Chat<span className="fw-semi-bold">TOS</span>
                  </h5>
                }
              >
                <div className={s.chatui}>
                  <ChatUI />
                </div>
              </Widget>
            </Col>
          </Row>
        </div>
      );
    }
  }
}

export default Configuration;