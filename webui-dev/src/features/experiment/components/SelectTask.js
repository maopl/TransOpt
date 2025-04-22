import React, { useState } from "react";
import { PlusOutlined } from '@ant-design/icons';
import { Button, Form, Input, Select, Modal, Table } from "antd";

const filterOption = (input, option) =>
  (option?.value ?? '').toLowerCase().includes(input.toLowerCase());

function TaskTable({ tasks, handleDelete, setDrawerVisible }) {
  return (
    <>
    <div>
    <Table
      dataSource={tasks}
      pagination={false}
      rowKey="index"
      columns={[
        { title: '#', dataIndex: 'index', key: 'index' },
        { title: 'Task Name', dataIndex: 'name', key: 'name' },
        { title: 'Variables', dataIndex: 'num_vars', key: 'num_vars' },
        { title: 'Objectives', dataIndex: 'num_objs', key: 'num_objs' },
        { title: 'Fidelity', dataIndex: 'fidelity', key: 'fidelity' },
        { title: 'Workloads', dataIndex: 'workloads', key: 'workloads' },
        { title: 'Budget Type', dataIndex: 'budget_type', key: 'budget_type' },
        { title: 'Budget', dataIndex: 'budget', key: 'budget' },
        {
          title: "Action",
          key: "action",
          render: (_, record, index) => (
            <Button
              type="link"
              danger
              onClick={() => handleDelete(index)}
            >
              Delete
            </Button>
          ),
        },
      ]}
      locale={{
        emptyText: 'No task'
      }}
    />
        <Button onClick={() => setDrawerVisible(true)} icon={<PlusOutlined />} style={{
              marginTop: "5px",
              width: "100%",
              borderColor: 'gray',
              border: "1px dashed"
            }}>
              Add new task
            </Button>
     </div>
     <div style={{ textAlign: 'right',marginTop:'10px' }}>
    </div>
    </>
  );
}


function SelectTask({ data,tasks, setTasks, updateTable }) {
  const [drawerVisible, setDrawerVisible] = useState(false);
  const [form] = Form.useForm(); // Form instance to manage form submission in the drawer

  const handleDrawerSubmit = () => {
    form
      .validateFields()
      .then(values => {
        console.log('Drawer form values:', values);

        setTasks(prevTasks => [...prevTasks, values]);

        form.resetFields(); // Reset the form fields after submission
        setDrawerVisible(false); // Close the drawer

      })
      .catch(info => {
        console.log('Validate Failed:', info);
      });
  };

  const handleDelete = (index) => {
    // 使用数组索引来过滤任务，而不是依赖task.index属性
    const updatedTasks = tasks.filter((_, i) => i !== index);
    setTasks(updatedTasks);
  };
  return (
    <>
        <TaskTable tasks={tasks} handleDelete={handleDelete} setDrawerVisible={setDrawerVisible}/>
      <Modal
        title="Add new task"
        placement="center"
        onCancel={() => setDrawerVisible(false)}
        open={drawerVisible}
        width={720}
        footer={(_, { CancelBtn }) => (
          <>
            <CancelBtn />
            <Button
                onClick={handleDrawerSubmit}
                type="primary" htmlType="submit" style={{ width: "73px", backgroundColor: 'rgb(53, 162, 235)' }}>
              Add
            </Button>
          </>
        )}
      >
        <Form
          form={form}
          labelCol={{ span: 8 }}
          wrapperCol={{ span: 16 }}
          name="drawer_form"

          style={{ width: "100%" }}
          autoComplete="off"
        >
          <Form.Item
            name="name"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>Problem Name</span>}
            rules={[{ required: true, message: 'Please select a problem name!' }]}
          >
            <Select
              showSearch
              placeholder="problem name"
              optionFilterProp="value"
              filterOption={filterOption}
              style={{ fontSize: '14px', width: '300px' }}
              options={data.map(item => ({ value: item.name }))}
            />
          </Form.Item>
          <Form.Item
            name="num_vars"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>Number of Variables</span>}
            rules={[{ required: true, message: 'Please enter the number of variables!' }]}
          >
            <Input placeholder="number of variables" style={{ fontSize: '14px', width: '300px' }}/>
          </Form.Item>
          <Form.Item
            name="num_objs"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>Number of Objectives</span>}
            rules={[{ required: true, message: 'Please select the number of objectives!' }]}
          >
            <Input placeholder="number of objectives" style={{ fontSize: '14px', width: '300px' }}/>
          </Form.Item>
          <Form.Item
            name="fidelity"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>Fidelity</span>}
            rules={[{ required: false, message: 'Please select fidelity!' }]}
          >
            <Select
              placeholder="fidelity"
              options={[]}
              style={{ fontSize: '14px', width: '300px' }}
            />
          </Form.Item>
          <Form.Item
            name="workloads"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>Workloads</span>}
            rules={[{ required: true, message: 'Please specify workloads!' }]}
          >
            <Input placeholder="specify workloads" style={{ fontSize: '14px', width: '300px' }}/>
          </Form.Item>
          <Form.Item
            name="budget_type"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>Budget Type</span>}
            rules={[{ required: true, message: 'Please select budget type!' }]}
          >
            <Select
              placeholder="budget type"
              style={{ fontSize: '14px', width: '200px' }}
              options={[
                { value: "function evaluations" },
                { value: "hours" },
                { value: "minutes" },
                { value: "seconds" },
              ]}
            />
          </Form.Item>
          <Form.Item
            name="budget"
            label={<span style={{ fontSize: '18px', fontWeight: 'bold' }}>Budget</span>}

            rules={[{ required: true, message: 'Please enter the budget!' }]}
          >
            <Input placeholder="budget" style={{ fontSize: '14px', width: '200px' }} />
          </Form.Item>
        </Form>
      </Modal >
    </>
  );
}


export default SelectTask;
