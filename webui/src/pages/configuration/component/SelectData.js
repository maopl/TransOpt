import React, {useState} from "react";

import {
    Button,
    Checkbox,
    ConfigProvider,
    Modal,
} from "antd";

const CheckboxGroup = Checkbox.Group;

function SelectData({Datasetdata}) {
    const [exact, setExact] = useState(false);
    const [data, setData] = useState([]);
    if (Array.isArray(Datasetdata)) {
      setData(Datasetdata);
    } else {
      setData([Datasetdata.name])
      setExact(true);
    }
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
          title: 'Information',
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
          <div style={{ overflowY: 'auto', maxHeight: '150px' }}>
            <Checkbox indeterminate={indeterminate} onChange={onCheckAllChange} checked={checkAll}>
                Check all
            </Checkbox>
            <CheckboxGroup options={data} value={checkedList} onChange={onChange}/>
          </div>
          {exact ? (
            <div style={{ overflowY: 'auto', maxHeight: '250px' }}>
            <h4><strong>Detail information</strong></h4>
            <ul>
              <li><h5><span className="fw-semi-bold">Name</span>: {Datasetdata.name}</h5></li>
              <li><h5><span className="fw-semi-bold">Dim</span>: {Datasetdata.dim}</h5></li>
              <li><h5><span className="fw-semi-bold">Obj</span>: {Datasetdata.obj}</h5></li>
              <li><h5><span className="fw-semi-bold">Fidelity</span>: {Datasetdata.fidelity}</h5></li>
              <li><h5><span className="fw-semi-bold">Workloads</span>: {Datasetdata.workloads}</h5></li>
              <li><h5><span className="fw-semi-bold">Budget type</span>: {Datasetdata.budget_type}</h5></li>
              <li><h5><span className="fw-semi-bold">Budget</span>: {Datasetdata.budget}</h5></li>
              <li><h5><span className="fw-semi-bold">Seeds</span>: {Datasetdata.seeds}</h5></li>
            </ul>
            <h4 className="mt-5"><strong>Algorithm</strong></h4>
            <ul>
              <li><h5><span className="fw-semi-bold">Space refiner</span>: {Datasetdata.SpaceRefiner}</h5></li>
              <li><h5><span className="fw-semi-bold">Sampler</span>: {Datasetdata.Sampler}</h5></li>
              <li><h5><span className="fw-semi-bold">Pre-train</span>: {Datasetdata.Pretrain}</h5></li>
              <li><h5><span className="fw-semi-bold">Model</span>: {Datasetdata.Model}</h5></li>
              <li><h5><span className="fw-semi-bold">ACF</span>: {Datasetdata.ACF}</h5></li>
              <li><h5><span className="fw-semi-bold">DatasetSelector</span>: {Datasetdata.DatasetSelector}</h5></li>
            </ul>
            <h4 className="mt-5"><strong>Auxiliary Data List</strong></h4>
            <ul>
              {Datasetdata.datasets.map((dataset, index) => (
                <li key={index}><h5>{dataset}</h5></li>
              ))}
            </ul>
            </div>
          ):(null)}
          <div style={{marginTop:"20px"}}>
            <Button type="primary" htmlType="submit" style={{width:"120px"}} onClick={handelClick}>
              Submit
            </Button>
          </div>
        </ConfigProvider>
    )
}

export default SelectData