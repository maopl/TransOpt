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
    Select,
} from "antd";
import { Modal } from "reactstrap";


function SearchData({set_dataset}) {
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
    .then(message => {
      console.log('Message from back-end:', message);
      if (message.isSucceed === false) {
        var errorMessage = message.info;
        console.log("Error:", errorMessage);
        Modal.error({
          title: 'Information',
          content: 'error:' + errorMessage
        })
      } else {
        // 返回dataset
        set_dataset(message)
      }
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
      <h6 style={{color:"white"}}>Search method:</h6>
      <Space className="space" style={{ display: 'flex'}} align="baseline">
      <Form.Item
        name="search_method"
      >
        <Select style={{minWidth: 150}}
          options={[ {value: "Hash"},
                      {value: "Fuzzy"},
                      {value: "LSH"},
                  ]}
        />
      </Form.Item>
      <Form.Item>
        <Button type="primary" htmlType="submit" style={{width:"120px"}}>
          Search
        </Button>
      </Form.Item>
      </Space>
    </Form>
    </ConfigProvider>
  )
}

export default SearchData