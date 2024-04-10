import React, {useState} from "react";

import {
    Row,
    Col,
    Button,
    InputNumber,
    Slider,
    Space,
} from "antd";

function DecimalStep({name}) {
    const [inputValue, setInputValue] = useState(0);
  
    const onChange = (value) => {
      if (isNaN(value)) {
        return;
      }
      setInputValue(value);
    };
  
    return (
      <Row>
        <Col span={6}>
            <h5 style={{height: '100%', lineHeight: '100%', color:'white'}}>{name}</h5>
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

  function SearchData() {
    return(
        <Space style={{ width: '100%' }} direction="vertical">
            <DecimalStep name={"name :"} />
            <DecimalStep name={"dim :"} />
            <DecimalStep name={"obj :"} />
            <DecimalStep name={"fidelity name :"} />
            <DecimalStep name={"fidelity :"} />
            <Button>Search</Button>
        </Space>
    )
  }

export default SearchData