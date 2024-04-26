import React, { useState } from "react";

import {
    Progress
} from "antd";

function TaskProgress() {
    const [tasksProgress, setTasksProgress] = useState([
        {name: "Task 1", progress: 80},
        {name: "Task 2", progress: 60},
    ])

    const task1 = tasksProgress[0]
    const task2 = tasksProgress[1]

    return (
        <div style={{ overflowY: 'auto', maxHeight: '250px' }}>
            <h6>{task1.name}</h6>
            <Progress percent={task1.progress} type="line" />
        </div>
    )
}

export default TaskProgress;