import React from "react";

import { Row, Col, Button } from "reactstrap";

import s from "./Report.module.scss";
import Widget from "../../components/Widget/Widget";

import Trajectory from "./charts/Trajectory";
import Radar from "./charts/Radar";
import Scatter from "./charts/Scatter";
import Bar from "./charts/Bar";

const handleTaskClick = () => {
  console.log('Task 1 button clicked');

  const messageToSend = {
    paremeter: 'task_button',
  };

  fetch('http://localhost:5000/api/task_button', {
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
    // TODO
    return response.json();
  })
  .then(data => {
    console.log('Message from back-end:', data);
    // TODO
  })
  .catch((error) => {
    console.error('Error sending message:', error);
    // TODO:
  });


};


class Report extends React.Component {
  state = {
    initEchartsOptions: {
      renderer: "canvas",
    },
  };

  render() { 
    // If first time rendering, then render the default task
    // If not, then render the task that was clicked
    if (this.state.task === undefined) {
      // TODO: ask for task list from back-end


      // Set the default task as the first task in the list
    }

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
                  <Button className={s.btn} onClick={handleTaskClick}> 
                      Task 1
                  </Button>
                  <Button className={s.btn} onClick={handleTaskClick}>
                      Task 2
                  </Button>
                  <Button className={s.btn} onClick={handleTaskClick}>
                      Task 3
                  </Button>
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
                    <h5>Task 1</h5>
                    <h4 className="mt-5"><strong>Auxiliary Data List</strong></h4>
                    <ul>
                      <li><h5>Dataset 1</h5></li>
                      <li><h5>Dataset 2</h5></li>
                    </ul>
                    <h4 className="mt-5"><strong>Algorithm</strong></h4>
                    <ul>
                      <li><h5>BO</h5></li>
                      <li><h5>MTBO</h5></li>
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
                    <Trajectory />
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
                    <Radar />
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
                    <Scatter />
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
                    <Bar />
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

export default Report;
