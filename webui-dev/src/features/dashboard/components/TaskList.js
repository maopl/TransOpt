import React from "react";
import { Button } from "antd";

const TaskList = ({ tasks, selectedTaskIndex, onTaskClick }) => (
  <div style={{ overflowY: 'auto', maxHeight: "400px" }}>
    {tasks.map((task, index) => (
      <Button
        key={index}
        type={selectedTaskIndex === index ? "primary" : "default"}
        onClick={() => onTaskClick(index)}
        style={{ marginBottom: 8, width: '100%' }}
      >
        {task.problem_name}
      </Button>
    ))}
  </div>
);

export default TaskList;
