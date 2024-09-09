import React from "react";
import {
  Row,
  Col,
} from "reactstrap";

import TitleCard from "../../components/Cards/TitleCard"


import SelectTask from "./components/SelectTask";
import SetInstance from "./components/SetInstance";

import TaskTable from "./components/TaskTable";


class Problem extends React.Component {
  constructor(props) {
    super(props);
    this.state = {
      TasksData: [],
      tasks: [],
    };
  }

  updateTable = (newTasks) => {
    this.setState({ tasks: newTasks });
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
        this.setState({ TasksData: data.TasksData });
      })
      .catch((error) => {
        console.error('Error sending message:', error);
      });

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
        this.setState({ tasks: data.tasks });
      })
      .catch((error) => {
        console.error('Error sending message:', error);
      });


    } else {
      return (
        <div className="grid mt-4 grid-cols-1 lg:grid-cols-[50%_50%] gap-6">

            <TitleCard title={'Choose a problem generator'}>
                      <SelectTask data={this.state.TasksData} updateTable={this.updateTable}/>
            </TitleCard>

            <TitleCard title={'Generate instances'}>
                      <SetInstance data={this.state.TasksData} updateTable={this.updateTable}/>
            </TitleCard>



          <div className="grid mt-4 w-[1600px] grid-cols-1 gap-6">
            <TitleCard
              title={
                <h5>
                  <span className="fw-semi-bold">Problem Information</span>
                </h5>
              }
              collapse
            >
              <TaskTable tasks={this.state.tasks} />
            </TitleCard>
          </div>
        


        </div>
      );
    }
  }
}

export default Problem;