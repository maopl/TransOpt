import React from "react";
import {
  Row,
  Col,
} from "reactstrap";

import s from "./Configuration.module.scss"

import Widget from "../../components/Widget/Widget";

import SelectTask from "./component/SelectTask";
import SelectAlgorithm from "./component/SelectAlgorithm";
import ChatUI from "./component/ChatUI";
import SearchData from "./component/SearchData";
import SelectData from "./component/SelectData";

class Configuration extends React.Component {
  render() {
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
                  There are some discription.There are some discription.There are some discription.There are some discription.
                </p>
                <SelectTask />
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
                <SelectAlgorithm />
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
                <SearchData />
                <h3 className="mt-5">
                  <span className="fw-semi-bold">Choose</span>
                </h3>
                <p>
                  There are some discription.There are some discription.There are some discription.There are some discription.
                </p>
                <SelectData />
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

export default Configuration;
