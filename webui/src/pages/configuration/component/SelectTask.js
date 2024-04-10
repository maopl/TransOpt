import React, { useState } from "react";
// import './Select.css';

import data from '../data/TasksData.json'

import { MinusCircleOutlined, PlusOutlined } from '@ant-design/icons';
import {
    Button,
    Form,
    Input,
    Space, 
    Select
} from "antd";

const onFinish = (values) => {
    console.log('Received values of form:', values);
  };

const filterOption = (input, option) =>
  (option?.value ?? '').toLowerCase().includes(input.toLowerCase());

function SelectDim({anyDim, dim, name, restField}) {
  if (anyDim) {
      return (
        <Form.Item
           {...restField}
           name={[name, 'dim']}
           rules={[{ required: true, message: 'Missing dim' }]}
         >
        <Input placeholder="dim" style={{width:"70px"}} />
        </Form.Item>
      )
  } else {
      return (
        <Form.Item
           {...restField}
           name={[name, 'dim']}
           rules={[{ required: true, message: 'Missing dim' }]}
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

function ATask({key, name, restField, remove}) {
    const [selectedTask, setSelectedTask] = useState(data[0])
    
    function handleNameChange(value) {
        console.log(`selected ${value}`);
        const task = data.find(task => task.name === value)
        setSelectedTask(task)
    }

    return (
        <Space className="space" key={key} style={{ display: 'flex', marginBottom: 8 }} align="baseline">
           <Form.Item
             {...restField}
             name={[name, 'name']}
             rules={[{ required: true, message: 'Missing name' }]}
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
             rules={[{ required: true, message: 'Missing fidelity' }]}
           >
             <Select
               placeholder="fidelity"
               options={selectedTask.fidelity.map(item => ({value: item}))}
             />
           </Form.Item>
           <MinusCircleOutlined style={{color: 'white'}} onClick={() => remove(name)} />
        </Space>
    )
}

function SelectTask() {
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
                  <ATask key={key} name={name} restField={restField} remove={remove} />
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