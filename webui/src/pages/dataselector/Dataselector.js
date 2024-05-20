import React from "react";
import {
  Row,
  Col,
} from "reactstrap";

import s from "./Dataselector.module.scss"

import Widget from "../../components/Widget/Widget";

import SelectData from "./component/SelectData";
import SearchData from "./component/SearchData"


class Dataselector extends React.Component {
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
            Experiment - <span className="fw-semi-bold">Dataselector</span>
          </h1>
        </div>
      )
    } else {
      return (
        <div className={s.root}>
          <h1 className="page-title">
            Experiment - <span className="fw-semi-bold">Dataselector</span>
          </h1>
            <Row>
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
            </Row>
        </div>
      );
    }
  }
}

export default Dataselector;