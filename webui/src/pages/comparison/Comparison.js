import React from "react";

import {
  Row,
  Col,
  Table,
  Input,
  Label,
  Button
} from "reactstrap";

import Widget from "../../components/Widget/Widget";

import s from "./Comparison.module.scss";

import Box from "./charts/Box";
import Trajectory from "./charts/Trajectory";

class Report extends React.Component {
  constructor(props) {
    super(props);

    this.state = {
      checkboxes_algorithm: [false, false, false, false, false],
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

  render() {
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
                  <Button className={s.btn}>
                      Task 1
                  </Button>
                  <Button className={s.btn}>
                      Task 2
                  </Button>
                  <Button className={s.btn}>
                      Task 3
                  </Button>
                  <Button className={s.btn}>
                      Task 4
                  </Button>
                  <Button className={s.btn}>
                      Task 5
                  </Button>
                  <Button className={s.btn}>
                      Task 6
                  </Button>
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
                  <div className={`widget-table-overflow ${s.overFlow}`}>
                    <Table className="table-striped table-lg mt-lg mb-0">
                      <thead>
                        <tr>
                          <th>
                            <div className="abc-checkbox">
                              <Input
                                id="checkbox-algorithm-comparison"
                                type="checkbox"
                                checked= {this.state.checkboxes_algorithm[0]}
                                onChange={(event) =>
                                  this.checkAll(event, "checkboxes_algorithm")
                                }
                              />
                              <Label for="checkbox-algorithm-comparison" />
                            </div>
                          </th>
                          <th>Algorithms</th>
                        </tr>
                      </thead>
                      <tbody>
                        <tr>
                          <td>
                            <div className="abc-checkbox">
                              <Input
                                id="checkbox-BO"
                                type="checkbox"
                                checked={this.state.checkboxes_algorithm[1]}
                                onChange={(event) =>
                                  this.changeCheck(event, "checkboxes_algorithm", 1)
                                }
                              />
                              <Label for="checkbox-BO" />
                            </div>
                          </td>
                          <td>BO</td>
                        </tr>
                        <tr>
                          <td>
                            <div className="abc-checkbox">
                              <Input
                                id="checkbox-MTBO"
                                type="checkbox"
                                checked={this.state.checkboxes_algorithm[2]}
                                onChange={(event) =>
                                  this.changeCheck(event, "checkboxes_algorithm", 2)
                                }
                              />
                              <Label for="checkbox-MTBO" />
                            </div>
                          </td>
                          <td>MTBO</td>
                        </tr>
                        <tr>
                          <td>
                            <div className="abc-checkbox">
                              <Input
                                id="checkbox-a1"
                                type="checkbox"
                                checked={this.state.checkboxes_algorithm[3]}
                                onChange={(event) =>
                                  this.changeCheck(event, "checkboxes_algorithm", 3)
                                }
                              />
                              <Label for="checkbox-a1" />
                            </div>
                          </td>
                          <td>Algorithm 1</td>
                        </tr>
                        <tr>
                          <td>
                            <div className="abc-checkbox">
                              <Input
                                id="checkbox-a2"
                                type="checkbox"
                                checked={this.state.checkboxes_algorithm[4]}
                                onChange={(event) =>
                                  this.changeCheck(event, "checkboxes_algorithm", 4)
                                }
                              />
                              <Label for="checkbox-a2" />
                            </div>
                          </td>
                          <td>Algorithm 2</td>
                        </tr>
                      </tbody>
                    </Table>
                  </div>
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
                        <Box />
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
                        <Trajectory />
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

export default Report;
