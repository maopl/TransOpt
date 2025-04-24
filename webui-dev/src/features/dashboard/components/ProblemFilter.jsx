import React from "react";
import { Card, Input, DatePicker, Form, Button, Space, Row, Col } from "antd";
import { SearchOutlined, FilterOutlined } from '@ant-design/icons';

// 统一的卡片样式
const cardStyle = {
  borderRadius: "8px",
  boxShadow: "0 1px 2px -2px rgba(0, 0, 0, 0.16), 0 3px 6px 0 rgba(0, 0, 0, 0.12), 0 5px 12px 4px rgba(0, 0, 0, 0.09)"
};

const { RangePicker } = DatePicker;

const ProblemFilter = ({ onFilter, onReset, form }) => {
  return (
    <Card
      bodyStyle={{ padding: "16px" }}
      style={{
        marginBottom: "16px",
        ...cardStyle
      }}
    >
      <Form
        form={form}
        layout="horizontal"
        onFinish={onFilter}
      >
        <Row gutter={16} align="middle">
          <Col span={6}>
            <Form.Item name="experimentName" style={{ marginBottom: 0 }}>
              <Input
                placeholder="Expriment Name"
                prefix={<SearchOutlined />}
                allowClear
                size="middle"
              />
            </Form.Item>
          </Col>
          <Col span={6}>
            <Form.Item name="problemName" style={{ marginBottom: 0 }}>
              <Input
                placeholder="Problem Name"
                prefix={<SearchOutlined />}
                allowClear
                size="middle"
              />
            </Form.Item>
          </Col>
          <Col span={7}>
            <Form.Item name="dateRange" style={{ marginBottom: 0 }}>
              <RangePicker
                style={{ width: "100%" }} size="middle"
                showTime={{ format: 'HH:mm:ss' }}
                format="YYYY-MM-DD HH:mm:ss"
              />
            </Form.Item>
          </Col>
          <Col span={5}>
            <div style={{display: "flex", justifyContent: "flex-end"}}>
              <Space>
                <Button
                    type="primary"
                    htmlType="submit"
                    icon={<FilterOutlined />}
                    size="middle"
                >
                  Filter
                </Button>
                <Button onClick={onReset} size="middle">
                  Reset
                </Button>
              </Space>
            </div>
          </Col>
        </Row>
      </Form>
    </Card>
  );
};

export default ProblemFilter;