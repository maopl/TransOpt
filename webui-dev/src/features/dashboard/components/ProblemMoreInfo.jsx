import React, { useState } from "react";
import { Button, Modal, Typography, Row, Col, Space, Card } from "antd";
import { ArrowRightOutlined, InfoCircleOutlined } from '@ant-design/icons';

const { Text, Title } = Typography;

const ProblemMoreInfo = ({ problem }) => {
  const [isModalVisible, setIsModalVisible] = useState(false);

  const showModal = () => setIsModalVisible(true);
  const handleCancel = () => setIsModalVisible(false);

  return (
    <>
      <Button
        type="primary"
        icon={<ArrowRightOutlined />}
        onClick={showModal}
      >
        More Info
      </Button>

      {/*<Modal*/}
      {/*  title={*/}
      {/*    <Space>*/}
      {/*      <InfoCircleOutlined style={{ color: "#1890ff", fontSize: "18px" }} />*/}
      {/*      <span>Detailed Information: {problem.problem_name}</span>*/}
      {/*    </Space>*/}
      {/*  }*/}
      {/*  open={isModalVisible}*/}
      {/*  onCancel={handleCancel}*/}
      {/*  width={1080}*/}
      {/*  footer={[*/}
      {/*    <Button key="close" onClick={handleCancel}>*/}
      {/*      Close*/}
      {/*    </Button>*/}
      {/*  ]}*/}
      {/*>*/}
      {/*  <div style={{ maxHeight: "80vh", overflowY: "auto", overflowX: "hidden" }}>*/}
      {/*    <section style={{ marginBottom: '20px', borderBottom: '1px solid #e0e0e0', paddingBottom: '15px' }}>*/}
      {/*      <Title level={5} style={{ color: '#333', marginBottom: '10px' }}>*/}
      {/*        Problem Information*/}
      {/*      </Title>*/}
      {/*      <Row gutter={[16, 8]}>*/}
      {/*        <Col span={24}>*/}
      {/*          <Text style={{ fontSize: '0.95em' }}>*/}
      {/*            <strong>Problem Name:</strong> {problem.displayName}*/}
      {/*          </Text>*/}
      {/*        </Col>*/}
      {/*        <Col span={8}>*/}
      {/*          <Text style={{ fontSize: '0.95em' }}>*/}
      {/*            <strong>Variable num:</strong> {problem.dim}*/}
      {/*          </Text>*/}
      {/*        </Col>*/}
      {/*        <Col span={8}>*/}
      {/*          <Text style={{ fontSize: '0.95em' }}>*/}
      {/*            <strong>Objective num:</strong> {problem.obj}*/}
      {/*          </Text>*/}
      {/*        </Col>*/}
      {/*        <Col span={8}>*/}
      {/*          <Text style={{ fontSize: '0.95em' }}>*/}
      {/*            <strong>Seeds:</strong> {problem.seeds}*/}
      {/*          </Text>*/}
      {/*        </Col>*/}
      {/*        <Col span={12}>*/}
      {/*          <Text style={{ fontSize: '0.95em' }}>*/}
      {/*            <strong>Budget type:</strong> {problem.budget_type}*/}
      {/*          </Text>*/}
      {/*        </Col>*/}
      {/*        <Col span={12}>*/}
      {/*          <Text style={{ fontSize: '0.95em' }}>*/}
      {/*            <strong>Budget:</strong> {problem.budget}*/}
      {/*          </Text>*/}
      {/*        </Col>*/}
      {/*        <Col span={24}>*/}
      {/*          <Text style={{ fontSize: '0.95em' }}>*/}
      {/*            <strong>Workloads:</strong> {problem.workloads}*/}
      {/*          </Text>*/}
      {/*        </Col>*/}
      {/*      </Row>*/}
      {/*    </section>*/}

      {/*    <section style={{ marginBottom: '20px', borderBottom: '1px solid #e0e0e0', paddingBottom: '15px' }}>*/}
      {/*      <Title level={5} style={{ color: '#333', marginBottom: '10px' }}>*/}
      {/*        Algorithm Objects*/}
      {/*      </Title>*/}
      {/*      <Row gutter={[16, 8]}>*/}
      {/*        <Col span={12}>*/}
      {/*          <Text style={{ fontSize: '0.95em' }}>*/}
      {/*            <strong>Narrow Search Space:</strong> {problem.SpaceRefiner}*/}
      {/*          </Text>*/}
      {/*        </Col>*/}
      {/*        <Col span={12}>*/}
      {/*          <Text style={{ fontSize: '0.95em' }}>*/}
      {/*            <strong>Initialization:</strong> {problem.Sampler}*/}
      {/*          </Text>*/}
      {/*        </Col>*/}
      {/*        <Col span={12}>*/}
      {/*          <Text style={{ fontSize: '0.95em' }}>*/}
      {/*            <strong>Pre-train:</strong> {problem.Pretrain}*/}
      {/*          </Text>*/}
      {/*        </Col>*/}
      {/*        <Col span={12}>*/}
      {/*          <Text style={{ fontSize: '0.95em' }}>*/}
      {/*            <strong>Surrogate Model:</strong> {problem.Model}*/}
      {/*          </Text>*/}
      {/*        </Col>*/}
      {/*        <Col span={12}>*/}
      {/*          <Text style={{ fontSize: '0.95em' }}>*/}
      {/*            <strong>Acquisition Function:</strong> {problem.ACF}*/}
      {/*          </Text>*/}
      {/*        </Col>*/}
      {/*        <Col span={12}>*/}
      {/*          <Text style={{ fontSize: '0.95em' }}>*/}
      {/*            <strong>Normalizer:</strong> {problem.Normalizer}*/}
      {/*          </Text>*/}
      {/*        </Col>*/}
      {/*      </Row>*/}
      {/*    </section>*/}

      {/*    {problem.description && (*/}
      {/*      <section style={{ marginBottom: '20px', borderBottom: '1px solid #e0e0e0', paddingBottom: '15px' }}>*/}
      {/*        <Title level={5} style={{ color: '#333', marginBottom: '10px' }}>*/}
      {/*          Problem Description*/}
      {/*        </Title>*/}
      {/*        <Text style={{ fontSize: '0.95em', display: 'block' }}>*/}
      {/*          {problem.description}*/}
      {/*        </Text>*/}
      {/*      </section>*/}
      {/*    )}*/}

      {/*    {problem.configuration && (*/}
      {/*      <section style={{ marginBottom: '20px', borderBottom: '1px solid #e0e0e0', paddingBottom: '15px' }}>*/}
      {/*        <Title level={5} style={{ color: '#333', marginBottom: '10px' }}>*/}
      {/*          Configuration*/}
      {/*        </Title>*/}
      {/*        <pre style={{*/}
      {/*          backgroundColor: '#f5f5f5',*/}
      {/*          padding: '16px',*/}
      {/*          borderRadius: '4px',*/}
      {/*          fontSize: '0.9em',*/}
      {/*          maxHeight: '300px',*/}
      {/*          overflowY: 'auto'*/}
      {/*        }}>*/}
      {/*          {JSON.stringify(problem.configuration, null, 2)}*/}
      {/*        </pre>*/}
      {/*      </section>*/}
      {/*    )}*/}
      {/*  </div>*/}
      {/*</Modal>*/}

      <Modal
          title={
            <Space>
              <InfoCircleOutlined style={{ color: "#1890ff", fontSize: "18px" }} />
              <span>Detailed Information: {problem.problem_name}</span>
            </Space>
          }
          open={isModalVisible}
          onCancel={handleCancel}
          width={1080}
          footer={[
            <Button key="close" onClick={handleCancel}>
              Close
            </Button>
          ]}
      >
        <div style={{ maxHeight: "80vh", overflowY: "auto", overflowX: "hidden" }}>
          <section style={{ marginBottom: '20px', borderBottom: '1px solid #e0e0e0', paddingBottom: '15px' }}>
            <Title level={5} style={{ color: '#333', marginBottom: '10px' }}>
              Problem Information
            </Title>
            <Row gutter={[16, 8]}>
              <Col span={24}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Problem Name:</strong> {problem.displayName}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Variable num:</strong> {problem.dim}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Objective num:</strong> {problem.obj}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Seeds:</strong> {problem.seeds}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Budget type:</strong> {problem.budget_type}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Budget:</strong> {problem.budget}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Workloads:</strong> {problem.workloads}
                </Text>
              </Col>
            </Row>
          </section>

          <section style={{ marginBottom: '20px', borderBottom: '1px solid #e0e0e0', paddingBottom: '15px' }}>
            <Title level={5} style={{ color: '#333', marginBottom: '10px' }}>
              Algorithm Objects
            </Title>
            <Row gutter={[16, 8]}>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Narrow Search Space:</strong> {problem.SearchSpace}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Initialization:</strong> {problem.Initialization}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Pre-train:</strong> {problem.Pretrain}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Surrogate Model:</strong> {problem.Model}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Acquisition Function:</strong> {problem.AcquisitionFunction}
                </Text>
              </Col>
              <Col span={8}>
                <Text style={{ fontSize: '0.95em' }}>
                  <strong>Normalizer:</strong> {problem.Normalizer}
                </Text>
              </Col>
            </Row>
          </section>

          <section>
            <Title level={5} style={{ color: '#333', marginBottom: '10px' }}>
              Auxilliary Data
            </Title>

            <Row gutter={[24, 16]}>
              <Col span={12}>
                <Card
                    size="small"
                    title={
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>Narrow Search Space</span>
                        <span>
                      DatasetSelector-
                          {`${problem.AutoSelect.SearchSpace}`}
                      </span>
                      </div>
                    }
                    style={{ marginBottom: '10px' }}
                >
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    {problem.auxiliaryData?.SearchSpace.map((dataset, index) => (
                        <li key={index} style={{ fontSize: '0.9em' }}>{dataset}</li>
                    ))}
                  </ul>
                </Card>
              </Col>

              <Col span={12}>
                <Card
                    size="small"
                    title={
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>Initialization</span>
                        <span>
                      DatasetSelector-
                          {`${problem.AutoSelect.Initialization}`}
                      </span>
                      </div>
                    }
                    style={{ marginBottom: '10px' }}
                >
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    {problem.auxiliaryData.Initialization.map((dataset, index) => (
                        <li key={index} style={{ fontSize: '0.9em' }}>{dataset}</li>
                    ))}
                  </ul>
                </Card>
              </Col>

              <Col span={12}>
                <Card
                    size="small"
                    title={
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>Pre-train</span>
                        <span>
                        DatasetSelector-
                          {`${problem.AutoSelect.Pretrain}`}
                      </span>
                      </div>
                    }
                    style={{ marginBottom: '10px' }}
                >
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    {problem.auxiliaryData.Pretrain.map((dataset, index) => (
                        <li key={index} style={{ fontSize: '0.9em' }}>{dataset}</li>
                    ))}
                  </ul>
                </Card>
              </Col>

              <Col span={12}>
                <Card
                    size="small"
                    title={
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>Surrogate Model</span>
                        <span>
                      DatasetSelector-
                          {`${problem.AutoSelect.Model}`}
                      </span>
                      </div>
                    }
                    style={{ marginBottom: '10px' }}
                >
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    {problem.auxiliaryData.Model.map((dataset, index) => (
                        <li key={index} style={{ fontSize: '0.9em' }}>{dataset}</li>
                    ))}
                  </ul>
                </Card>
              </Col>

              <Col span={12}>
                <Card
                    size="small"
                    title={
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>Acquisition Function</span>
                        <span>
                      DatasetSelector-
                          {`${problem.AutoSelect.AcquisitionFunction}`}
                      </span>
                      </div>
                    }
                    style={{ marginBottom: '10px' }}
                >
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    {problem.auxiliaryData.AcquisitionFunction.map((dataset, index) => (
                        <li key={index} style={{ fontSize: '0.9em' }}>{dataset}</li>
                    ))}
                  </ul>
                </Card>
              </Col>

              <Col span={12}>
                <Card
                    size="small"
                    title={
                      <div style={{ display: 'flex', justifyContent: 'space-between' }}>
                        <span>Normalizer</span>
                        <span>
                      DatasetSelector-
                          {`${problem.AutoSelect.Normalizer}`}
                      </span>
                      </div>
                    }
                >
                  <ul style={{ paddingLeft: '20px', marginBottom: 0 }}>
                    {problem.auxiliaryData.Normalizer.map((dataset, index) => (
                        <li key={index} style={{ fontSize: '0.9em' }}>{dataset}</li>
                    ))}
                  </ul>
                </Card>
              </Col>
            </Row>
          </section>
        </div>
      </Modal>

    </>
  );
};

export default ProblemMoreInfo;