import React from "react";

import { Row, Col } from "reactstrap";

import Run from "./components/Run"
import RunProgress from "./components/RunProgress"


class RunPage extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      get_info: false,
      tasks: [],
      optimizer: {},
      datasets: {},
    };
  }

  render() {
    // If first time rendering, then render the default task
    // If not, then render the task that was clicked
    if (this.state.get_info === false) {
      // TODO: ask for task list from back-end
      const messageToSend = {
        action: 'ask for information',
      }
      fetch('http://localhost:5001/api/RunPage/get_info', {
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
          this.setState({
            get_info: true,
            tasks: data.tasks,
            optimizer: data.optimizer,
            datasets: data.datasets
          });
        })
        .catch((error) => {
          console.error('Error sending message:', error);
        });


      // Set the default task as the first task in the list
      return (
        <div>
          <h1 className="page-title">
            <span className="fw-semi-bold">Run</span>
          </h1>
        </div>
      )
    } else {

      return (
        <div>
          <Run />
          <RunProgress />
        </div>
      );
    }
  }

}

export default RunPage;
