import React from "react";
import {
  Row,
  Col,
} from "reactstrap";

import s from "./Dataselector.module.scss"

import Widget from "../../components/Widget/Widget";

import SelectData from "./component/SelectData";
import SearchData from "./component/SearchData"
import DataTable from "./component/DataTable";


class Dataselector extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      get_info: false,
      datasets: {},
      DatasetData: {"isExact": false, "datasets": []}
    };
  }

  set_dataset = (datasets) => {
    console.log(datasets)
    this.setState({ DatasetData: datasets })
  }

  render() {
    if (this.state.get_info === false) {
      // TODO: ask for task list from back-end
      const messageToSend = {
        action: 'ask for information',
      }
      fetch('http://localhost:5000/api/RunPage/get_info', {
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
        console.log('Configuration infomation from back-end:', data);
        this.setState({ get_info: true,  
                        datasets: data.datasets});
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
              <Col lg={12} xs={12}>
                <Widget
                  title={
                    <h5>
                      <span className="fw-semi-bold">Selected Datasets</span>
                    </h5>
                  }
                  collapse
                >
                  <DataTable datasets={this.state.datasets} />
                </Widget>
              </Col>
            </Row>
        </div>
      );
    }
  }
}

export default Dataselector;