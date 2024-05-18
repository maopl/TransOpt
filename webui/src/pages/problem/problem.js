import React from "react";
import {
  Row,
  Col,
} from "reactstrap";
import { Button, Modal } from "antd";

import s from "./problem.module.scss"

import Widget from "../../components/Widget/Widget";

import SelectTask from "./component/SelectTask";




class Problem extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      TasksData: [],
      SpaceRefiner: [],
      Sampler: [],
      Pretrain: [],
      Model: [],
      ACF: [],
      DataSelector: [],
      Normalizer: [],
      DatasetData: {"isExact": false, "datasets": []}
    };
  }

  set_dataset = (datasets) => {
    console.log(datasets)
    this.setState({ DatasetData: datasets })
  }

  render() {
    if (this.state.TasksData.length === 0) {
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
        this.setState({ TasksData: data.TasksData,
                        SpaceRefiner: data.SpaceRefiner,
                        Sampler: data.Sampler,
                        Pretrain: data.Pretrain,
                        Model: data.Model,
                        ACF: data.ACF,
                        DataSelector: data.DataSelector,
                        Normalizer: data.Normalizer,
                      });
      })
      .catch((error) => {
        console.error('Error sending message:', error);
      });

      return (
        <div className={s.root}>
          <h1 className="page-title">
            Experiment - <span className="fw-semi-bold">Problem specification</span>
          </h1>
        </div>
      )
    } else {
      return (
        <div className={s.root}>
          <h1 className="page-title">
            Experiment - <span className="fw-semi-bold">Problem specification</span>
          </h1>
            <Row>
              <Col lg={12} sm={12}>
                <Widget
                  title={
                    <h5>
                      1.<span className="fw-semi-bold">Choose Tasks</span>
                    </h5>
                  }
                  collapse
                >
                  <SelectTask data={this.state.TasksData}/>
                </Widget>
              </Col>
            </Row>
        </div>
      );
    }
  }
}

export default Problem;