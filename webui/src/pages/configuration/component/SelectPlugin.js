import React, { useState } from "react";

import {
    Button,
    Form,
    Input,
    Select,
    ConfigProvider,
    Modal,
} from "antd";

function SelectAlgorithm({SpaceRefiner, Sampler, Pretrain, Model, ACF, DataSelector, Normalizer}) {
    const [form] = Form.useForm()

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
              title: 'Information',
              content: 'Submit successfully!'
            })
          })
          .catch((error) => {
            console.error('Error sending message:', error);
            var errorMessage = error.error;
            Modal.error({
              title: 'Information',
              content: 'error:' + errorMessage
            })
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
              SpaceRefiner: SpaceRefiner[0].name,
              SpaceRefinerParameters: '',
              SpaceRefinerDataSelector: 'default',
              SpaceRefinerDataSelectorParameters: '',
              Sampler: Sampler[0].name,
              SamplerParameters: '',
              SamplerDataSelector: 'default',
              SamplerDataSelectorParameters: '',
              Pretrain: Pretrain[0].name,
              PretrainParameters: '',
              PretrainDataSelector: 'default',
              PretrainDataSelectorParameters: '',
              Model: Model[0].name,
              ModelParameters: '',
              ModelDataSelector: 'default',
              ModelDataSelectorParameters: '',
              ACF: ACF[0].name,
              ACFParameters: '',
              ACFDataSelector: 'default',
              ACFDataSelectorParameters: '',
              Normalizer: Normalizer[0].name,
              NormalizerParameters: '',
              NormalizerDataSelector: 'default',
              NormalizerDataSelectorParameters: '',
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
                name={'SpaceRefiner'}
                style={{ marginRight: 8 , width: 150}}
              >
                <Select 
                  placeholder="name"
                  defaultValue={SpaceRefiner[0].name}
                  options={SpaceRefiner.map(item => ({ value: item.name }))}
                />
              </Form.Item>
              <Form.Item
                name={'SpaceRefinerParameters'}
                style={{ flex: 1 , marginRight: 8}}
              >
                <Input placeholder="Parameters"/>
              </Form.Item>
              <h7 style={{color:"white", marginRight:8}}>DataSelector: </h7>
              <Form.Item
                name={'SpaceRefinerDataSelector'}
                style={{ marginRight: 8 , width: 150}}
              >
                <Select
                  placeholder="DataSelector"
                  options={DataSelector.map(item => ({ value: item.name }))}
                />
              </Form.Item>
              <Form.Item
                name={'SpaceRefinerDataSelectorParameters'}
                style={{ flex: 1 }}
              >
                <Input placeholder="Parameters"/>
              </Form.Item>
            </div>

            <div>
                <h5 style={{color:"#f4f4f599"}}>
                  <span className="fw-semi-bold">Sampler</span>
                </h5>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline' }}>
              <Form.Item
                name={'Sampler'}
                style={{ marginRight: 8 , width: 150}}
              >
                <Select
                  placeholder="name"
                  defaultValue={Sampler[0].name}
                  options={Sampler.map(item => ({ value: item.name }))}
                />
              </Form.Item>
              <Form.Item
                name={'SamplerParameters'}
                style={{ flex: 1 }}
              >
                <Input placeholder="Parameters"/>
              </Form.Item>
              <Form.Item
                name={'SamplerInitNum'}
                style={{ flex: 1, marginLeft: 8, marginRight: 8}}
              >
                <Input placeholder="Initial Sample Size"/>
              </Form.Item>
              <h7 style={{color:"white", marginRight:8}}>DataSelector: </h7>
              <Form.Item
                name={'SamplerDataSelector'}
                style={{ marginRight: 8 , width: 150}}
              >
                <Select
                  placeholder="DataSelector"
                  options={DataSelector.map(item => ({ value: item.name }))}
                />
              </Form.Item>
              <Form.Item
                name={'SamplerDataSelectorParameters'}
                style={{ flex: 1 }}
              >
                <Input placeholder="Parameters"/>
              </Form.Item>
            </div>

            <div>
                <h5 style={{color:"#f4f4f599"}}>
                  <span className="fw-semi-bold">Pre-train</span>
                </h5>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline' }}>
              <Form.Item
                name={'Pretrain'}
                style={{ marginRight: 8 , width: 150}}
              >
                <Select
                  placeholder="name"
                  defaultValue={Pretrain[0].name}
                  options={Pretrain.map(item => ({ value: item.name }))}
                />
              </Form.Item>
              <Form.Item
                name={'PretrainParameters'}
                style={{ flex: 1 , marginRight: 8 }}
              >
                <Input placeholder="Parameters"/>
              </Form.Item>
              <h7 style={{color:"white", marginRight:8}}>DataSelector: </h7>
              <Form.Item
                name={'PretrainDataSelector'}
                style={{ marginRight: 8 , width: 150}}
              >
                <Select
                  placeholder="DataSelector"
                  options={DataSelector.map(item => ({ value: item.name }))}
                />
              </Form.Item>
              <Form.Item
                name={'PretrainDataSelectorParameters'}
                style={{ flex: 1 }}
              >
                <Input placeholder="Parameters"/>
              </Form.Item>
            </div>

            <div>
                <h5 style={{color:"#f4f4f599"}}>
                  <span className="fw-semi-bold">Model</span>
                </h5>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline' }}>
              <Form.Item
                name={'Model'}
                style={{ marginRight: 8 , width: 150}}
              >
                <Select
                  placeholder="name"
                  defaultValue={Model[0].name}
                  options={Model.map(item => ({ value: item.name }))}
                />
              </Form.Item>
              <Form.Item
                name={'ModelParameters'}
                style={{ flex: 1, marginRight: 8}}
              >
                <Input placeholder="Parameters" />
              </Form.Item>
              <h7 style={{color:"white", marginRight:8}}>DataSelector: </h7>
              <Form.Item
                name={'ModelDataSelector'}
                style={{ marginRight: 8 , width: 150}}
              >
                <Select
                  placeholder="DataSelector"
                  options={DataSelector.map(item => ({ value: item.name }))}
                />
              </Form.Item>
              <Form.Item
                name={'ModelDataSelectorParameters'}
                style={{ flex: 1 }}
              >
                <Input placeholder="Parameters" />
              </Form.Item>
            </div>

            <div>
                <h5 style={{color:"#f4f4f599"}}>
                  <span className="fw-semi-bold">Acquisition function</span>
                </h5>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline' }}>
              <Form.Item
                name={'ACF'}
                style={{ marginRight: 8 , width: 150}}
              >
                <Select
                  placeholder="name"
                  defaultValue={ACF[0].name}
                  options={ACF.map(item => ({ value: item.name }))}
                />
              </Form.Item>
              <Form.Item
                name={'ACFParameters'}
                style={{ flex: 1, marginRight: 8}}
              >
                <Input placeholder="Parameters" />
              </Form.Item>
              <h7 style={{color:"white", marginRight:8}}>DataSelector: </h7>
              <Form.Item
                name={'ACFDataSelector'}
                style={{ marginRight: 8 , width: 150}}
              >
                <Select
                  placeholder="DataSelector"
                  options={DataSelector.map(item => ({ value: item.name }))}
                />
              </Form.Item>
              <Form.Item
                name={'ACFDataSelectorParameters'}
                style={{ flex: 1 }}
              >
                <Input placeholder="Parameters" />
              </Form.Item>
            </div>

            <div>
                <h5 style={{color:"#f4f4f599"}}>
                  <span className="fw-semi-bold">Normalizer</span>
                </h5>
            </div>
            <div style={{ display: 'flex', alignItems: 'baseline' }}>
              <Form.Item
                name={'Normalizer'}
                style={{ marginRight: 8 , width: 150}}
              >
                <Select
                  placeholder="name"
                  defaultValue={Normalizer[0].name}
                  options={Normalizer.map(item => ({ value: item.name }))}
                />
              </Form.Item>
              <Form.Item
                name={'NormalizerParameters'}
                style={{ flex: 1, marginRight: 8}}
              >
                <Input placeholder="Parameters"/>
              </Form.Item>
              <h7 style={{color:"white", marginRight:8}}>DataSelector: </h7>
              <Form.Item
                name={'NormalizerDataSelector'}
                style={{ marginRight: 8 , width: 150}}
              >
                <Select
                  placeholder="DataSelector"
                  options={DataSelector.map(item => ({ value: item.name }))}
                />
              </Form.Item>
              <Form.Item
                name={'NormalizerDataSelectorParameters'}
                style={{ flex: 1 }}
              >
                <Input placeholder="Parameters"/>
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