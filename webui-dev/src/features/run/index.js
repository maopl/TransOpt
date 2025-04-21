import React from "react";
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

      return (
          <div>
            <Run />
            <RunProgress />
          </div>
      );
  }
}

export default RunPage;
