import React, {useState} from "react";

import {
    Button,
    Checkbox,
    ConfigProvider
} from "antd";

import data from "../data/DatasetData.json"

const CheckboxGroup = Checkbox.Group;

function SelectData() {
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
            <Button>Begin</Button>
        </ConfigProvider>
    )
}

export default SelectData