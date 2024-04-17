import React, {useState} from "react";

import {
    Row,
    Col,
    Button,
    InputNumber,
    Slider,
    Space,
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
    name: 0,
    dim: 0,
    obj: 0,
    fidelityName: 0,
    fidelity: 0,
  });

  const handleInputChange = (name, value) => {
    setValues({ ...values, [name]: value });
  };

  const handleSearch = () => {
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
    <Space style={{ width: '100%' }} direction="vertical">
      <DecimalStep name="name :" inputValue={values.name} onChange={(value) => handleInputChange('name', value)} />
      <DecimalStep name="dim :" inputValue={values.dim} onChange={(value) => handleInputChange('dim', value)} />
      <DecimalStep name="obj :" inputValue={values.obj} onChange={(value) => handleInputChange('obj', value)} />
      <DecimalStep name="fidelity name :" inputValue={values.fidelityName} onChange={(value) => handleInputChange('fidelityName', value)} />
      <DecimalStep name="fidelity :" inputValue={values.fidelity} onChange={(value) => handleInputChange('fidelity', value)} />
    <Button onClick={handleSearch}>Search</Button>
    </Space>
  )
}

export default SearchData