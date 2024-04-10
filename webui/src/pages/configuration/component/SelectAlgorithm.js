import React, { useState } from "react";

import data from '../data/AlgorithmsData.json'

import {
    Button,
    Form,
    Input,
    Select,
    ConfigProvider
} from "antd";

const onFinish = (values) => {
    console.log('Received values of form:', values);
  };

function SelectAlgorithm() {
    const [selectedAlgorithm, setSelectedAlgorithm] = useState(data[0])
    const [form] = Form.useForm()

    function handleNameChange(value) {
        console.log(`selected ${value}`);
        const algorithm = data.find(algorithm => algorithm.name === value)
        // console.log(algorithm);
        setSelectedAlgorithm(algorithm)
        form.setFieldValue('parameter', algorithm.default)
    }

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
            <Form.Item
                name={'name'}
                rules={[{ required: true, message: 'Missing name'}]}
            >
                <Select
                placeholder="name"
                value={selectedAlgorithm.name}
                options={data.map(item => ({ value: item.name }))}
                onChange={handleNameChange}
                />
            </Form.Item>
            <Form.Item
                name={'parameter'}
            >
                <Input placeholder="Parameters" value={selectedAlgorithm.default} addonBefore={"Parameters"} style={{addonBg:'#ffffff'}}/>
            </Form.Item>
        </Form>
        </ConfigProvider>
    )
}

export default SelectAlgorithm;