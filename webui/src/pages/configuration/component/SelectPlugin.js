import React, { useState } from "react";

import {
    Button,
    Form,
    Input,
    Select,
    ConfigProvider,
    Modal,
} from "antd";

function SelectAlgorithm({SearchSpace, Sample, PreTrain, Train, Acf, DataSelector}) {
    const [form] = Form.useForm()
    
    const [selectedSearchSpace, setSearchSpace] = useState(SearchSpace[0])
    function handleSSNameChange(value) {
        // console.log(`selected ${value}`);
        const searchspace = SearchSpace.find(searchspace => searchspace.name === value)
        setSearchSpace(searchspace)
        form.setFieldValue('SearchSpaceParameters', searchspace.default)
    }

    const [selectedSample, setSample] = useState(Sample[0])
    function handleSampleNameChange(value) {
        // console.log(`selected ${value}`);
        const sample = Sample.find(sample => sample.name === value)
        setSample(sample)
        form.setFieldValue('SampleParameters', sample.default)
    }

    const [selectedPreTrain, setPreTrain] = useState(PreTrain[0])
    function handlePreTrainNameChange(value) {
        // console.log(`selected ${value}`);
        const pretrain = PreTrain.find(pretrain => pretrain.name === value)
        setPreTrain(pretrain)
        form.setFieldValue('PreTrainParameters', pretrain.default)
    }

    const [selectedTrain, setTrain] = useState(Train[0])
    function handleTrainNameChange(value) {
        // console.log(`selected ${value}`);
        const train = Train.find(train => train.name === value)
        setTrain(train)
        form.setFieldValue('TrainParameters', train.default)
    }

    const [selectedAcf, setAcf] = useState(Acf[0])
    function handleAcfNameChange(value) {
        // console.log(`selected ${value}`);
        const acf = Acf.find(acf => acf.name === value)
        setAcf(acf)
        form.setFieldValue('AcfParameters', acf.default)
    }

    const [selectedDataSelector, setDataSelector] = useState(DataSelector[0])
    function handleDataSelectorNameChange(value) {
        // console.log(`selected ${value}`);
        const dataselector = DataSelector.find(dataselector => dataselector.name === value)
        setDataSelector(dataselector)
        form.setFieldValue('DataSelectorParameters', dataselector.default)
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
            initialValues={{
              SearchSpace: SearchSpace[0].name,
              SearchSpaceParameters: SearchSpace[0].default,
              Sample: Sample[0].name,
              SampleParameters: Sample[0].default,
              PreTrain: PreTrain[0].name,
              PreTrainParameters: PreTrain[0].default,
              Train: Train[0].name,
              TrainParameters: Train[0].default,
              Acf: Acf[0].name,
              AcfParameters: Acf[0].default,
              DataSelector: DataSelector[0].name,
              DataSelectorParameters: DataSelector[0].default,
            }}
        >
          <div style={{ overflowY: 'auto', maxHeight: '200px' }}>
            <div>
                <h5 style={{color:"#f4f4f599"}}>
                  <span className="fw-semi-bold">Space refiner</span>
                </h5>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline' }}>
              <Form.Item
                name={'SearchSpace'}
                style={{ marginRight: 8 , width: 150}}
              >
                <Select 
                  placeholder="name"
                  defaultValue="default"
                  options={SearchSpace.map(item => ({ value: item.name }))}
                  onChange={handleSSNameChange}
                />
              </Form.Item>
              <Form.Item
                name={'SearchSpaceParameters'}
                style={{ flex: 1 }}
              >
                <Input placeholder="Parameters" value={selectedSearchSpace.default} />
              </Form.Item>
            </div>

            <div>
                <h5 style={{color:"#f4f4f599"}}>
                  <span className="fw-semi-bold">Sampler</span>
                </h5>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline' }}>
              <Form.Item
                name={'Sample'}
                style={{ marginRight: 8 , width: 150}}
              >
                <Select
                  placeholder="name"
                  defaultValue="default"
                  options={Sample.map(item => ({ value: item.name }))}
                  onChange={handleSampleNameChange}
                />
              </Form.Item>
              <Form.Item
                name={'SampleParameters'}
                style={{ flex: 1 }}
              >
                <Input placeholder="Parameters" value={selectedSample.default} />
              </Form.Item>
            </div>

            <div>
                <h5 style={{color:"#f4f4f599"}}>
                  <span className="fw-semi-bold">Pre-train</span>
                </h5>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline' }}>
              <Form.Item
                name={'PreTrain'}
                style={{ marginRight: 8 , width: 150}}
              >
                <Select
                  placeholder="name"
                  defaultValue="default"
                  options={PreTrain.map(item => ({ value: item.name }))}
                  onChange={handlePreTrainNameChange}
                />
              </Form.Item>
              <Form.Item
                name={'PreTrainParameters'}
                style={{ flex: 1 }}
              >
                <Input placeholder="Parameters" value={selectedPreTrain.default} />
              </Form.Item>
            </div>

            <div>
                <h5 style={{color:"#f4f4f599"}}>
                  <span className="fw-semi-bold">Model</span>
                </h5>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline' }}>
              <Form.Item
                name={'Train'}
                style={{ marginRight: 8 , width: 150}}
              >
                <Select
                  placeholder="name"
                  defaultValue="default"
                  options={Train.map(item => ({ value: item.name }))}
                  onChange={handleTrainNameChange}
                />
              </Form.Item>
              <Form.Item
                name={'TrainParameters'}
                style={{ flex: 1 }}
              >
                <Input placeholder="Parameters" value={selectedTrain.default} />
              </Form.Item>
            </div>

            <div>
                <h5 style={{color:"#f4f4f599"}}>
                  <span className="fw-semi-bold">Acquisition function</span>
                </h5>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline' }}>
              <Form.Item
                name={'Acf'}
                style={{ marginRight: 8 , width: 150}}
              >
                <Select
                  placeholder="name"
                  defaultValue="default"
                  options={Acf.map(item => ({ value: item.name }))}
                  onChange={handleAcfNameChange}
                />
              </Form.Item>
              <Form.Item
                name={'AcfParameters'}
                style={{ flex: 1 }}
              >
                <Input placeholder="Parameters" value={selectedAcf.default} />
              </Form.Item>
            </div>

            <div>
                <h5 style={{color:"#f4f4f599"}}>
                  <span className="fw-semi-bold">Dataset selector</span>
                </h5>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline' }}>
              <Form.Item
                name={'DataSelector'}
                style={{ marginRight: 8 , width: 150}}
              >
                <Select
                  placeholder="name"
                  defaultValue={DataSelector[0].name}
                  options={DataSelector.map(item => ({ value: item.name }))}
                  onChange={handleDataSelectorNameChange}
                />
              </Form.Item>
              <Form.Item
                name={'DataSelectorParameters'}
                style={{ flex: 1 }}
              >
                <Input defaultValue={''} placeholder="Parameters" value={selectedDataSelector.default} />
              </Form.Item>
            </div>
          </div>

          <Form.Item style={{marginTop:10}}>
            <Button type="primary" htmlType="submit" style={{width:"120px"}}>
              Submit
            </Button>
          </Form.Item>
        </Form>
        </ConfigProvider>
    )
}

export default SelectAlgorithm;