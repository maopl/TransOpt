import React from "react";
import {
  Row,
  Col,
} from "reactstrap";
import { Button, Modal } from "antd";

import s from "./Configuration.module.scss"

import Widget from "../../components/Widget/Widget";

import SelectTask from "./component/SelectTask";
import SelectPlugins from "./component/SelectPlugin";
import ChatUI from "./component/ChatUI";
import SelectData from "./component/SelectData";
import SearchData from "./component/SearchData"
import Run from "./component/Run"
import TaskProgress from "./component/TaskProgress"



class Configuration extends React.Component {
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
                      1.<span className="fw-semi-bold">Choose Tasks</span>
                    </h5>
                  }
                  collapse
                >
                  <SelectTask data={this.state.TasksData}/>
                </Widget>
              </Col>
              <Col lg={12} sm={12}>
                <Widget
                  title={
                    <h5>
                      2. <span className="fw-semi-bold">Choose Optimization Plugins</span>
                    </h5>
                  }
                  collapse
                >
                  <SelectPlugins SpaceRefiner={this.state.SpaceRefiner}
                                    Sampler={this.state.Sampler}
                                    Pretrain={this.state.Pretrain}
                                    Model={this.state.Model}
                                    ACF={this.state.ACF}
                                    DataSelector={this.state.DataSelector}
                                    Normalizer={this.state.Normalizer}
                  />
                </Widget>
              </Col>
              <Col lg={12} sm={12}> 
                <Widget
                  title={
                    <h5>
                      3. <span className="fw-semi-bold">Choose Datasets</span>
                    </h5>
                  }
                  collapse
                >
                  <SearchData set_dataset={this.set_dataset}/>
                  <p>
                    Choose the datasets you want to use in the experiment.
                  </p>
                  <SelectData DatasetData={this.state.DatasetData} set_dataset={this.set_dataset}/>
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
                  <Run />
                  {/* <TaskProgress /> */}
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