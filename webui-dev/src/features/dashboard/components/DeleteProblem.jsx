import React from "react";
import { Button, Popconfirm } from "antd";
import { DeleteOutlined } from '@ant-design/icons';

const DeleteProblem = ({ problemName, onDelete }) => {
  return (
    <Popconfirm
      title="Delete this task"
      description="Are you sure you want to delete this task?"
      onConfirm={(e) => {
        e.preventDefault();
        onDelete(problemName)
      }}
      okText="Yes"
      cancelText="No"
    >
      <Button
        type="primary"
        danger
        icon={<DeleteOutlined />}
      >
        Delete
      </Button>
    </Popconfirm>
  );
};

export default DeleteProblem;