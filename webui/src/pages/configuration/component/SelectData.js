import React, {useState} from "react";

import {
    Button,
    Checkbox,
    ConfigProvider,
    Modal,
    Select,
} from "antd";

const CheckboxGroup = Checkbox.Group;

function SelectData({DatasetData, set_dataset}) {
    var data = []
    if (DatasetData.isExact) {
      data = [DatasetData.datasets.name]
    } else {
      data = DatasetData.datasets
    }
    const [checkedList, setCheckedList] = useState([]);
    const [selectedOption, setSelectedOption] = useState();
    const checkAll = data.length === checkedList.length;
    const indeterminate = checkedList.length > 0 && checkedList.length < data.length;
    const onChange = (list) => {
        setCheckedList(list);
    };
    const onCheckAllChange = (e) => {
        setCheckedList(e.target.checked ? data : []);
    };
    const handleSelectChange = (value) => {
      setSelectedOption(value); // 当选择发生变化时更新选项
    };
    const handleClick = () => {
      const datasetList = checkedList.map(item => {
        return item;
      });
      const messageToSend = {
        object: selectedOption,
        datasets: datasetList,
      }
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
          title: 'Information',
          content: 'Submit successfully!'
        })
      })
      .catch((error) => {
        console.error('Error sending message:', error);
        var errorMessage = error.error;
        Modal.error({
          title: 'Information',
          content: 'Error:' + errorMessage
        })
      });
    }

    const handleDelete = () => {
      const datasetList = checkedList.map(item => {
        return item;
      });
      const messageToSend = {
        datasets: datasetList,
      }
      console.log(messageToSend)
      fetch('http://localhost:5000/api/configuration/delete_dataset', {
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
        datasetList.forEach(item => {
          let index = data.indexOf(item);
          if (index !== -1) {
            data.splice(index, 1);
          }
        });
        var newDataset = {"isExact": false, "datasets": data}
        console.log("new dataset:", newDataset)
        // set_dataset(newDataset)
        Modal.success({
          title: 'Information',
          content: 'Delete successfully!'
        })
      })
      .catch((error) => {
        console.error('Error sending message:', error);
        var errorMessage = error.error;
        Modal.error({
          title: 'Information',
          content: 'Error:' + errorMessage
        })
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
          <div style={{ overflowY: 'auto', maxHeight: '150px' }}>
            <Checkbox indeterminate={indeterminate} onChange={onCheckAllChange} checked={checkAll}>
                Check all
            </Checkbox>
            <CheckboxGroup options={data} value={checkedList} onChange={onChange}/>
          </div>
          <Info  isExact={DatasetData.isExact} data={DatasetData.datasets}/>
          <div style={{marginTop:"20px"}}>
            <Select
            style={{minWidth: 170}}
            options={[ {value: "Space refiner"},
                        {value: "Sampler"},
                        {value: "Pre-train"},
                        {value: "Model"},
                        {value: "Acquisition function"},
                        {value: "Normalizer"}
                    ]}
            onChange={handleSelectChange}
            />
            <Button type="primary" htmlType="submit" style={{width:"120px", marginLeft:10}} onClick={handleClick}>
              Submit
            </Button>
            <Button danger style={{width:"120px", marginLeft:10}} onClick={handleDelete}>
              Delete
            </Button>
          </div>
        </ConfigProvider>
    )
}


function Info({isExact, data}) {
  if (isExact) {
    return (
      <div style={{ overflowY: 'auto', maxHeight: '250px' }}>
        <h4><strong>Information</strong></h4>
          <ul>
            <li><h6><span className="fw-semi-bold">Name</span>: {data.name}</h6></li>
            <li><h6><span className="fw-semi-bold">Dim</span>: {data.dim}</h6></li>
            <li><h6><span className="fw-semi-bold">Obj</span>: {data.obj}</h6></li>
            <li><h6><span className="fw-semi-bold">Fidelity</span>: {data.fidelity}</h6></li>
            <li><h6><span className="fw-semi-bold">Workloads</span>: {data.workloads}</h6></li>
            <li><h6><span className="fw-semi-bold">Budget type</span>: {data.budget_type}</h6></li>
            <li><h6><span className="fw-semi-bold">Budget</span>: {data.budget}</h6></li>
            <li><h6><span className="fw-semi-bold">Seeds</span>: {data.seeds}</h6></li>
          </ul>
          <h4 className="mt-5"><strong>Algorithm</strong></h4>
          <ul>
            <li><h6><span className="fw-semi-bold">Space refiner</span>: {data.SpaceRefiner}</h6></li>
            <li><h6><span className="fw-semi-bold">Sampler</span>: {data.Sampler}</h6></li>
            <li><h6><span className="fw-semi-bold">Pre-train</span>: {data.Pretrain}</h6></li>
            <li><h6><span className="fw-semi-bold">Model</span>: {data.Model}</h6></li>
            <li><h6><span className="fw-semi-bold">ACF</span>: {data.ACF}</h6></li>
            <li><h6><span className="fw-semi-bold">DatasetSelector</span>: {data.DatasetSelector}</h6></li>
            <li><h6><span className="fw-semi-bold">Normalizer</span>: {data.Normalizer}</h6></li>
          </ul>
          <h4 className="mt-5"><strong>Auxiliary Data List</strong></h4>
          <ul>
            {data.metadata.map((dataset, index) => (
              <li key={index}><h6>{dataset}</h6></li>
            ))}
          </ul>
      </div>
    )
  } else {
    return (
      <></>
    )
  }
}

export default SelectData