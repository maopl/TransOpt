import React, {useState} from "react";

import {
    Row,
    Col,
    Button,
    InputNumber,
    Slider,
    Space,
    Input,
    Form,
    ConfigProvider,
} from "antd";

function DecimalStep({ name, inputValue, onChange }) {
  return (
    <Row>
      <Col span={6}>
          <h5 style={{ height: '100%', lineHeight: '100%', color:'white' }}>{name}</h5>
      </Col>
      <Col span={10}>
        <Slider
          min={0}
          max={1}
          onChange={onChange}
          value={typeof inputValue === 'number' ? inputValue : 0}
          step={0.01}
        />
      </Col>
      <Col span={4}>
        <InputNumber
          min={0}
          max={1}
          style={{ margin: '0 16px' }}
          step={0.01}
          value={inputValue}
          onChange={onChange}
        />
      </Col>
    </Row>
  );
};
function SearchData({set_dataset}) {
  const [values, setValues] = useState({
    task_name: "",
    num_variables: 0,
    variables_name: "",
    num_objectives: 0,
  });

  const [form] = Form.useForm()


  const onFinish = (values) => {
    const messageToSend = values;
    console.log('Request data:', messageToSend);
    // 向后端发送请求...
    fetch('http://localhost:5000/api/configuration/search_dataset', {
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
      // 返回dataset
      set_dataset(data)
    })
    .catch((error) => {
      console.error('Error sending message:', error);
    });
  }

  return(
    <ConfigProvider
      theme={{
        components: {
          Input: {
            addonBg:"white"
          },
        },
      }}  
    >
    <Form
      name="SearchData"
      form={form}
      onFinish={onFinish}
      style={{width:"100%"}}
      autoComplete="off"
    >
      <Space className="space" style={{ display: 'flex'}} align="baseline">
        <Form.Item
          name="task_name"
          style={{flexGrow: 1}}
        >
          <Input addonBefore={"Task Name"}/>
        </Form.Item>
        <Form.Item
          name="num_variables"
          style={{flexGrow: 1}}
        >
          <Input addonBefore={"Num of Variables"}/>
        </Form.Item>
      </Space>
      <Space className="space" style={{ display: 'flex'}} align="baseline">
        <Form.Item
          name="variables_name"
          style={{flexGrow: 1}}
        >
          <Input addonBefore={"Variables Name"}/>
        </Form.Item>
        <Form.Item
          name="num_objectives"
          style={{flexGrow: 1}}
        >
          <Input addonBefore={"Num of Objectives"}/>
        </Form.Item>
      </Space>
      <Form.Item>
        <Button type="primary" htmlType="submit" style={{width:"120px"}}>
          Search
        </Button>
      </Form.Item>
    </Form>
    </ConfigProvider>
  )
}

export default SearchData