import React, { useState } from "react";

import {
    Button,
    Form,
    Input,
    Space, 
    Select,
    Modal,
    ConfigProvider
} from "antd";

function Run() {
    const [form] = Form.useForm()

    const onFinish = (values) => {
        // 构造要发送到后端的数据
        const messageToSend = values
        console.log('Request data:', messageToSend);
        // 向后端发送请求...
        fetch('http://localhost:5000/api/configuration/run', {
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
              title: 'Infor',
              content: 'Run Successfully!'
            })
          })
          .catch((error) => {
            console.error('Error sending message:', error);
          });
      };

    return (
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
            form={form}
            name="dynamic_form_nest_item"
            onFinish={onFinish}
            style={{ width:"100%" }}
            autoComplete="off"
        >
            <div style={{ overflowY: 'auto', maxHeight: '150px' }}>
                <div style={{ display: 'flex', alignItems: 'baseline' }}>
                    <h6 style={{color:"white"}}>Seeds</h6>
                    <Form.Item name="Seeds" style={{marginRight:10, marginLeft:10}}>
                        <Input />
                    </Form.Item>
                    <h6 style={{color:"white"}}>Remote</h6>
                    <Form.Item name="Remote" style={{marginRight:10, marginLeft:10}}>
                      <Select
                      options={[ {value: "True"},
                                  {value: "False"},
                              ]}
                      />
                    </Form.Item>
                    <h6 style={{color:"white"}}>ServerURL</h6>
                    <Form.Item name="ServerURL" style={{marginLeft:10}}>
                        <Input />
                    </Form.Item>
                </div>
            </div>
            <Form.Item>
            <Button type="primary" htmlType="submit" style={{width:"120px"}}>
                Run
            </Button>
            </Form.Item>
        </Form>
      </ConfigProvider>
    );
}

export default Run;