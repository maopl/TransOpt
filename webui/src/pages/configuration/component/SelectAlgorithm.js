import React, { useState } from "react";

import {
    Button,
    Form,
    Input,
    Select,
    ConfigProvider,
    Modal,
    Space,
} from "antd";

function SelectAlgorithm({data}) {
    const [selectedAlgorithm, setSelectedAlgorithm] = useState(data[0])
    const [form] = Form.useForm()

    function handleNameChange(value) {
        console.log(`selected ${value}`);
        const algorithm = data.find(algorithm => algorithm.name === value)
        // console.log(algorithm);
        setSelectedAlgorithm(algorithm)
        form.setFieldValue('parameters', algorithm.default)
    }

    const onFinish = (values) => {
        // 构造要发送到后端的数据
        const messageToSend = values;
        console.log('Request data:', messageToSend);
        // 向后端发送请求...
        fetch('http://localhost:5000/api/configuration/select_algorithm', {
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
              title: 'Algorithm',
              content: 'Submit successfully!'
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
            name="Algorithm"
            onFinish={onFinish}
            style={{width:"100%"}}
            autoComplete="off"
        >
          <div>
              <h5 style={{color:"#f4f4f599"}}>
                <span className="fw-semi-bold">Search space</span>
              </h5>
          </div>
          <div style={{ display: 'flex', marginBottom: 8, alignItems: 'baseline' }}>
            <Form.Item
              name={'search_space'}
              style={{ marginRight: 8 , width: 150}}
            >
              <Select
                placeholder="name"
                value={selectedAlgorithm.name}
                options={data.map(item => ({ value: item.name }))}
                onChange={handleNameChange}
              />
            </Form.Item>
            <Form.Item
              name={'ss_parameters'}
              style={{ flex: 1 }}
            >
              <Input placeholder="Parameters" value={selectedAlgorithm.default} />
            </Form.Item>
          </div>
          <div>
              <h5 style={{color:"#f4f4f599"}}>
                <span className="fw-semi-bold">Sample initial set</span>
              </h5>
          </div>
          <div style={{ display: 'flex', marginBottom: 8, alignItems: 'baseline' }}>
            <Form.Item
              name={'sample'}
              style={{ marginRight: 8 , width: 150}}
            >
              <Select
                placeholder="name"
                value={selectedAlgorithm.name}
                options={data.map(item => ({ value: item.name }))}
                onChange={handleNameChange}
              />
            </Form.Item>
            <Form.Item
              name={'sample_parameters'}
              style={{ flex: 1 }}
            >
              <Input placeholder="Parameters" value={selectedAlgorithm.default} />
            </Form.Item>
          </div>
          <div>
              <h5 style={{color:"#f4f4f599"}}>
                <span className="fw-semi-bold">Pre-train model</span>
              </h5>
          </div>
          <div style={{ display: 'flex', marginBottom: 8, alignItems: 'baseline' }}>
            <Form.Item
              name={'pre_train_model'}
              style={{ marginRight: 8 , width: 150}}
            >
              <Select
                placeholder="name"
                value={selectedAlgorithm.name}
                options={data.map(item => ({ value: item.name }))}
                onChange={handleNameChange}
              />
            </Form.Item>
            <Form.Item
              name={'pre_train_parameters'}
              style={{ flex: 1 }}
            >
              <Input placeholder="Parameters" value={selectedAlgorithm.default} />
            </Form.Item>
          </div>
          <Form.Item>
            <Button type="primary" htmlType="submit" style={{width:"120px"}}>
              Submit
            </Button>
          </Form.Item>
        </Form>
        </ConfigProvider>
    )
}

export default SelectAlgorithm;