import React, { useState, useEffect } from "react";
import TitleCard from "../../components/Cards/TitleCard"
import SelectTask from "./components/SelectTask";
import SelectAlgorithm from "./components/SelectAlgorithm";
import SearchData from "./components/SearchData";
import SelectData from "./components/SelectData";
import RunPage from '../../features/run/index'
const Experiment = (props) => {
  // Convert class state to useState hooks
  const [tasksData, setTasksData] = useState([]);
  const [tasks, setTasks] = useState([]);
  const [spaceRefiner, setSpaceRefiner] = useState([]);
  const [sampler, setSampler] = useState([]);
  const [pretrain, setPretrain] = useState([]);
  const [model, setModel] = useState([]);
  const [acf, setACF] = useState([]);
  const [dataSelector, setDataSelector] = useState([]);
  const [normalizer, setNormalizer] = useState([]);
  const [optimizer, setOptimizer] = useState({});
  const [getInfo, setGetInfo] = useState(false);
  const [datasetData, setDatasetData] = useState({"isExact": false, "datasets": []});
  const [spaceRefinerDataSelector, setSpaceRefinerDataSelector] = useState("");
  const [spaceRefinerDataSelectorParameters, setSpaceRefinerDataSelectorParameters] = useState("");
  const [samplerDataSelector, setSamplerDataSelector] = useState("");
  const [samplerDataSelectorParameters, setSamplerDataSelectorParameters] = useState("");
  const [pretrainDataSelector, setPretrainDataSelector] = useState("");
  const [pretrainDataSelectorParameters, setPretrainDataSelectorParameters] = useState("");
  const [modelDataSelector, setModelDataSelector] = useState("");
  const [modelDataSelectorParameters, setModelDataSelectorParameters] = useState("");
  const [acfDataSelector, setACFDataSelector] = useState("");
  const [acfDataSelectorParameters, setACFDataSelectorParameters] = useState("");
  const [normalizerDataSelector, setNormalizerDataSelector] = useState("");
  const [normalizerDataSelectorParameters, setNormalizerDataSelectorParameters] = useState("");
  const [datasetSelector, setDatasetSelector] = useState([]);

  // Convert class methods to regular functions
  const updateTaskTable = (newTasks) => {
    setTasks(newTasks);
  };

  const updateOptTable = (newOptimizer) => {
    setOptimizer(newOptimizer);
  };

  const updateDataTable = (newDatasets) => {
    console.log("newDatasets", newDatasets);
    const { object, DatasetSelector, parameter, datasets } = newDatasets;
    if (object === "Narrow Search Space") {
      setSpaceRefiner(datasets);
      setSpaceRefinerDataSelector(DatasetSelector);
      setSpaceRefinerDataSelectorParameters(parameter);
    } else if (object === "Initialization") {
      setSampler(datasets);
      setSamplerDataSelector(DatasetSelector);
      setSamplerDataSelectorParameters(parameter);
    } else if (object === "Pre-train") {
      setPretrain(datasets);
      setPretrainDataSelector(DatasetSelector);
      setPretrainDataSelectorParameters(parameter);
    } else if (object === "Surrogate Model") {
      setModel(datasets);
      setModelDataSelector(DatasetSelector);
      setModelDataSelectorParameters(parameter);
    } else if (object === "Acquisition Function") {
      setACF(datasets);
      setACFDataSelector(DatasetSelector);
      setACFDataSelectorParameters(parameter);
    } else if (object === "Normalizer") {
      setNormalizer(datasets);
      setNormalizerDataSelector(DatasetSelector);
      setNormalizerDataSelectorParameters(parameter);
    }
  };

  const set_dataset = (datasets) => {
    console.log(datasets);
    setDatasetData(datasets);
  };

  // Convert class lifecycle method to useEffect
  useEffect(() => {
    if (tasksData.length === 0) {
      const messageToSend = {
        action: 'ask for basic information',
      };

      // First fetch for basic information
      fetch('http://localhost:5001/api/configuration/basic_information', {
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
      .then(data => {
        console.log('Message from back-end:', data);
        setTasksData(data.TasksData);
        setTasks(data.tasks);
        setSpaceRefiner(data.SpaceRefiner);
        setSampler(data.Sampler);
        setPretrain(data.Pretrain);
        setModel(data.Model);
        setACF(data.ACF);
        setDataSelector(data.DataSelector);
        setNormalizer(data.Normalizer);
        setGetInfo(true);
        setSpaceRefinerDataSelector(data.SpaceRefinerDataSelector);
        setSpaceRefinerDataSelectorParameters(data.SpaceRefinerDataSelectorParameters);
        setSamplerDataSelector(data.SamplerDataSelector);
        setSamplerDataSelectorParameters(data.SamplerDataSelectorParameters);
        setPretrainDataSelector(data.PretrainDataSelector);
        setPretrainDataSelectorParameters(data.PretrainDataSelectorParameters);
        setModelDataSelector(data.ModelDataSelector);
        setModelDataSelectorParameters(data.ModelDataSelectorParameters);
        setACFDataSelector(data.ACFDataSelector);
        setACFDataSelectorParameters(data.ACFDataSelectorParameters);
        setNormalizerDataSelector(data.NormalizerDataSelector);
        setNormalizerDataSelectorParameters(data.NormalizerDataSelectorParameters);
      })
      .catch((error) => {
        console.error('Error sending message:', error);
      });

      // Second fetch for get_info
      fetch('http://localhost:5001/api/RunPage/get_info', {
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
      .then(data => {
        console.log('Configuration infomation from back-end:', data);
        setTasks(data.tasks);
        setOptimizer(data.optimizer);
      })
      .catch((error) => {
        console.error('Error sending message:', error);
      });
    }
  }, []); // Empty dependency array means this runs once on component mount

  // Render component
  if (tasksData.length === 0) {
    return null; // Return null while data is loading
  }

  return (
    <div className="grid mt-4">
        <TitleCard>
            <SelectTask data={tasksData} updateTable={updateTaskTable}/>
        </TitleCard>

        <TitleCard
          title={
            <h5>
              <span className="fw-semi-bold">Build Algorithm</span>
            </h5>
          }
          collapse
        >
          <SelectAlgorithm 
            SpaceRefiner={spaceRefiner}
            Sampler={sampler}
            Pretrain={pretrain}
            Model={model}
            ACF={acf}
            DataSelector={dataSelector}
            Normalizer={normalizer}
            updateTable={updateOptTable} 
          />
           <RunPage />
        </TitleCard>        
{/* 
        <TitleCard
          title={
            <h5>
              <span className="fw-semi-bold">Customize Auxiliary Data</span>
            </h5>
          }
          collapse
        >
          <SearchData set_dataset={set_dataset}/>
          <p>
            Choose the datasets you want to use in the experiment.
          </p>
          <SelectData 
            DatasetData={datasetData} 
            updateTable={updateDataTable} 
            DatasetSelector={datasetSelector}
          />
        </TitleCard> */}
    </div>
  );
};

export default Experiment;