import React, {useState} from "react";

import {
    Button,
    Checkbox,
    ConfigProvider,
    Modal,
} from "antd";

const CheckboxGroup = Checkbox.Group;

function SelectData({data}) {
    const [checkedList, setCheckedList] = useState([]);
    const checkAll = data.length === checkedList.length;
    const indeterminate = checkedList.length > 0 && checkedList.length < data.length;
    const onChange = (list) => {
        setCheckedList(list);
    };
    const onCheckAllChange = (e) => {
        setCheckedList(e.target.checked ? data : []);
    };
    const handelClick = () => {
      const messageToSend = checkedList.map(item => {
        return item;
      });
      console.log(messageToSend)
      fetch('http://localhost:5000/api/configuration/dataset', {
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
          title: 'Infor',
          content: 'Submit successfully!'
        })
      })
      .catch((error) => {
        console.error('Error sending message:', error);
      });
    }

    return(
        <ConfigProvider
          theme={{
            components: {
              Checkbox: {
                colorText:"white"
              },
            },
          }}        
        >
          <div>
            <Checkbox indeterminate={indeterminate} onChange={onCheckAllChange} checked={checkAll}>
                Check all
            </Checkbox>
            <CheckboxGroup options={data} value={checkedList} onChange={onChange}/>
          </div>
          <div style={{marginTop:"20px"}}>
            <Button type="primary" htmlType="submit" style={{width:"120px"}} onClick={handelClick}>
              Submit
            </Button>
          </div>
        </ConfigProvider>
    )
}

export default SelectData