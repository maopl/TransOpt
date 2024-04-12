import React, {useState} from "react";

import {
    Button,
    Checkbox,
    ConfigProvider,
    Modal,
} from "antd";

const CheckboxGroup = Checkbox.Group;

function SelectData({data, handelClick}) {
    const [checkedList, setCheckedList] = useState([]);
    const checkAll = data.length === checkedList.length;
    const indeterminate = checkedList.length > 0 && checkedList.length < data.length;
    const onChange = (list) => {
        setCheckedList(list);
    };
    const onCheckAllChange = (e) => {
        setCheckedList(e.target.checked ? data : []);
    };

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
            <Checkbox indeterminate={indeterminate} onChange={onCheckAllChange} checked={checkAll}>
                Check all
            </Checkbox>
            <CheckboxGroup options={data} value={checkedList} onChange={onChange}/>
            <Button onClick={() => handelClick(checkedList)}>choose</Button>
        </ConfigProvider>
    )
}

export default SelectData