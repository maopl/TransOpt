import React, { useState } from "react";

import { MinusCircleOutlined, PlusOutlined } from '@ant-design/icons';
import {
    Button,
    Form,
    Input,
    Space, 
    Select,
    Modal,
} from "antd";

const filterOption = (input, option) =>
  (option?.value ?? '').toLowerCase().includes(input.toLowerCase());

function SelectDim({anyDim, dim, name, restField}) {
  if (anyDim) {
      return (
        <Form.Item
           {...restField}
           name={[name, 'dim']}
         >
        <Input placeholder="dim" style={{width:"70px"}} />
        </Form.Item>
      )
  } else {
      return (
        <Form.Item
           {...restField}
           name={[name, 'dim']}
         >
           <Select
             placeholder="dim"
             options={dim}
             style={{width:"70px"}}
           />
         </Form.Item>
      )
  }
}

function ATask({key, name, restField, remove, data}) {
    const [selectedTask, setSelectedTask] = useState(data[0])
    
    function handleNameChange(value) {
        console.log(`selected ${value}`);
        const task = data.find(task => task.name === value)
        setSelectedTask(task)
    }

    return (
        <Space className="space" key={key} style={{ display: 'flex', marginBottom: 1 }} align="baseline">
           <Form.Item
             {...restField}
             name={[name, 'name']}
           >
             <Select
               showSearch
               placeholder="name"
               optionFilterProp="value"
               filterOption={filterOption}
               options={data.map(item => ({ value: item.name}))}
               onChange={handleNameChange}
             />
           </Form.Item>
           <SelectDim anyDim={selectedTask.anyDim} dim={selectedTask.dim.map(item => ({value: item}))} name={name} restField={restField} />
           <Form.Item
             {...restField}
             name={[name, 'obj']}
             rules={[{ required: true, message: 'Missing obj' }]}
           >
             <Select
               placeholder="obj"
               options={selectedTask.obj.map(item => ({value: item}))}
             />
           </Form.Item>
           <Form.Item
             {...restField}
             name={[name, 'fidelity']}
           >
             <Select
               placeholder="fidelity"
               options={selectedTask.fidelity.map(item => ({value: item}))}
             />
           </Form.Item>
           <Form.Item
             {...restField}
             name={[name, 'budget_type']}
           >
             <Select
               placeholder="budget_type"
               options={[ {value: "evaluation count"},
                          {value: "time(day)"},
                          {value: "time(hour)"},
                          {value: "time(minute)"},
                        ]}
             />
           </Form.Item>
           <Form.Item
             {...restField}
             name={[name, 'budget']}
           >
             <Input placeholder="budget" style={{width:"100px"}} />
           </Form.Item>
           <MinusCircleOutlined style={{color: 'white'}} onClick={() => remove(name)} />
        </Space>
    )
}

function SelectTask({data}) {
  const onFinish = (values) => {
    // 构造要发送到后端的数据
    const messageToSend = values.Tasks.map(task => ({
      name: task.name,
      dim: parseInt(task.dim),
      obj: task.obj,
      fidelity: task.fidelity,
      budget_type: task.budget_type,
      budget: task.budget,
    }));
    console.log('Request data:', messageToSend);
    // 向后端发送请求...
    fetch('http://localhost:5000/api/configuration/select_task', {
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
      .then(succeed => {
        console.log('Message from back-end:', succeed);
        Modal.success({
          title: 'Tasks List',
          content: 'Submit successfully!'
        })
      })
      .catch((error) => {
        console.error('Error sending message:', error);
      });
  };

  return (
    <Form
      name="dynamic_form_nest_item"
      onFinish={onFinish}
      style={{ width:"100%" }}
      autoComplete="off"
    >
      <Form.List name="Tasks">
        {(fields, { add, remove }) => (
          <>
            {fields.map(({ key, name, ...restField }) => (
              <ATask key={key} name={name} restField={restField} remove={remove} data={data} />
            ))}
            <Form.Item>
              <Button type="dashed" onClick={() => add()} icon={<PlusOutlined />} style={{width:"120px"}}>
                Add Task
              </Button>
            </Form.Item>
            <Form.Item>
              <Button type="primary" htmlType="submit" style={{width:"120px"}}>
                Submit
              </Button>
            </Form.Item>
          </>
        )}
      </Form.List>
    </Form>
  )
}

export default SelectTask;