import React, {useState, useEffect} from "react";
import {
    PartitionOutlined,
    ExperimentOutlined,
    RobotOutlined,
    ApiOutlined,
    AreaChartOutlined,
    SlidersOutlined,
    DatabaseOutlined,
    EditOutlined,
    DeleteOutlined,
    TagsOutlined,
    EyeOutlined
} from '@ant-design/icons';
import {Button, Select, Modal, Row, Col, Space, Tag, Divider, Typography, Tooltip, Checkbox, Input} from "antd";

import SearchData from './SearchData';

const {Text} = Typography;

const filterOption = (input, option) =>
    (option?.value ?? '').toLowerCase().includes(input.toLowerCase());

// 统一算法类型常量
const ALGORITHM_TYPES = [
    "SearchSpace",
    "Initialization",
    "Pretrain",
    "Model",
    "AcquisitionFunction",
    "Normalizer"
];

// 显示名称的映射
const ALGORITHM_TYPES_NAMES = {
    "SearchSpace": 'Search Space',
    "Initialization": "Initialization",
    "Pretrain": "Pretrain",
    "Model": "Model",
    "AcquisitionFunction": "Acquisition Function",
    "Normalizer": "Normalizer"
}

function SelectAlgorithm({
    SearchSpaceOptions,
    InitializationOptions,
    PretrainOptions,
    ModelOptions,
    AcquisitionFunctionOptions,
    NormalizerOptions,
    algorithmValue,
    setAlgorithmValue
}) {
    // Modal visibility states for each algorithm's data selection
    const [activeModal, setActiveModal] = useState(null);

    // 预览模态窗口状态
    const [previewModal, setPreviewModal] = useState({
        visible: false,
        algorithmType: '',
        datasets: []
    });

    // 保存对algorithmValue的本地引用，确保能正确更新UI
    const [localAlgorithmValue, setLocalAlgorithmValue] = useState([]);

    // 当外部algorithmValue变化时，更新本地状态
    useEffect(() => {
        if (algorithmValue && Array.isArray(algorithmValue)) {
            setLocalAlgorithmValue(algorithmValue);
        }
    }, [algorithmValue]);

    /**
     * 算法对应的下拉选项
     */
    const algorithmOptionsMap = {
        "SearchSpace": SearchSpaceOptions,
        "Initialization": InitializationOptions,
        "Pretrain": PretrainOptions,
        "Model": ModelOptions,
        "AcquisitionFunction": AcquisitionFunctionOptions,
        "Normalizer": NormalizerOptions
    };

    // 更新算法值并通知父组件
    const updateAlgorithmValue = (newValue) => {
        setLocalAlgorithmValue(newValue);
        if (setAlgorithmValue) {
            setAlgorithmValue(newValue);
        }
        // 保存到localStorage以持久化
        localStorage.setItem('algorithmFormData', JSON.stringify(newValue));
    };

    // Handler for opening a specific algorithm's data selection modal
    const openDataSelectionModal = (algorithmType) => {
        setActiveModal(algorithmType);
    };

    // Handler for closing the active modal
    const closeDataSelectionModal = () => {
        setActiveModal(null);
    };

    // 打开预览模态窗口
    const openPreviewModal = (algorithmType) => {
        const algorithm = localAlgorithmValue.find(alg => alg.name === algorithmType);
        const datasets = algorithm?.auxiliaryData || [];
        
        setPreviewModal({
            visible: true,
            algorithmType,
            datasets: datasets.map(name => ({ name }))
        });
    };

    // 关闭预览模态窗口
    const closePreviewModal = () => {
        setPreviewModal({
            visible: false,
            algorithmType: '',
            datasets: []
        });
    };

    // Handler for when data is selected from the SearchData modal
    const handleSelectData = (datasetData, algorithmType) => {
        // 从数据集中提取名称
        const auxiliaryData = datasetData.datasets.map(dataset => dataset.name || dataset.value);
        
        const updatedValue = localAlgorithmValue.map(alg => {
            if (alg.name === algorithmType) {
                return { ...alg, auxiliaryData };
            }
            return alg;
        });
        
        updateAlgorithmValue(updatedValue);
    };

    // 清除数据集
    const clearSelectedDatasets = (algorithmType) => {
        const updatedValue = localAlgorithmValue.map(alg => {
            if (alg.name === algorithmType) {
                return { ...alg, auxiliaryData: [] };
            }
            return alg;
        });
        
        updateAlgorithmValue(updatedValue);
    };
    
    // 处理自动选择复选框变更
    const handleAutoSelectChange = (e, algorithmType) => {
        const checked = e.target.checked;
        
        const updatedValue = localAlgorithmValue.map(alg => {
            if (alg.name === algorithmType) {
                return { ...alg, autoSelect: checked };
            }
            return alg;
        });
        
        updateAlgorithmValue(updatedValue);
    };
    
    // 获取特定算法的自动选择状态
    const getAutoSelectStatus = (algorithmType) => {
        const algorithm = localAlgorithmValue.find(alg => alg.name === algorithmType);
        return algorithm?.autoSelect || false;
    };
    
    // 处理算法类型选择变更
    const handleAlgorithmTypeChange = (value, algorithmType) => {
        console.log('value', value, 'algorithmType', algorithmType);
        
        const updatedValue = localAlgorithmValue.map(alg => {
            if (alg.name === algorithmType) {
                return { ...alg, type: value };
            }
            return alg;
        });
        
        updateAlgorithmValue(updatedValue);
    };

    // 更新initialization算法的初始数量
    const handleInitialNumberChange = (event) => {
        const value = parseInt(event.target.value, 10) || 0;
        
        const updatedValue = localAlgorithmValue.map(alg => {
            if (alg.name === "Initialization") {
                return { ...alg, InitNum: value };
            }
            return alg;
        });
        
        updateAlgorithmValue(updatedValue);
    };

    // 获取特定算法对象
    const getAlgorithm = (algorithmType) => {
        return localAlgorithmValue.find(alg => alg.name === algorithmType) || {};
    };

    // 获取特定算法的数据集
    const getAlgorithmDatasets = (algorithmType) => {
        const algorithm = getAlgorithm(algorithmType);
        if (algorithm && algorithm.auxiliaryData && algorithm.auxiliaryData.length > 0) {
            return algorithm.auxiliaryData.map(name => ({ name }));
        }
        return [];
    };

    // 渲染数据选择区域
    const renderDataSelectionArea = (algorithmType) => {
        const selectedDatasets = getAlgorithmDatasets(algorithmType);
        const hasSelectedData = selectedDatasets.length > 0;
        const isAutoSelected = getAutoSelectStatus(algorithmType);
        
        return (
            <div style={{marginTop: '8px'}}>
                {!hasSelectedData ? (
                    <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                        <Button
                            type="default"
                            size="small"
                            icon={<DatabaseOutlined/>}
                            onClick={() => openDataSelectionModal(algorithmType)}
                            disabled={isAutoSelected}
                        >
                            Select Auxiliary Data
                        </Button>
                        <Checkbox 
                            checked={isAutoSelected}
                            onChange={(e) => handleAutoSelectChange(e, algorithmType)}
                            id={`checkbox-${algorithmType.replace(/\s+/g, '-').toLowerCase()}`}
                        >
                          <span title="自动选择辅助数据">
                            {'Auto Select'}
                          </span>
                        </Checkbox>
                    </div>
                ) : (
                    <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                        <Text style={{marginRight: '8px'}}>
                            <TagsOutlined/> 已选择 {selectedDatasets.length} 条数据
                        </Text>
                        <Space size="small">
                            <Tooltip title="查看数据集">
                                <Button
                                    type="text"
                                    size="small"
                                    icon={<EyeOutlined/>}
                                    onClick={() => openPreviewModal(algorithmType)}
                                    disabled={isAutoSelected}
                                />
                            </Tooltip>
                            <Tooltip title="编辑选择">
                                <Button
                                    type="text"
                                    size="small"
                                    icon={<EditOutlined/>}
                                    onClick={() => openDataSelectionModal(algorithmType)}
                                    disabled={isAutoSelected}
                                />
                            </Tooltip>
                            <Tooltip title="清除选择">
                                <Button
                                    type="text"
                                    size="small"
                                    danger
                                    icon={<DeleteOutlined/>}
                                    onClick={() => clearSelectedDatasets(algorithmType)}
                                    disabled={isAutoSelected}
                                />
                            </Tooltip>
                            <Checkbox 
                                checked={isAutoSelected}
                                onChange={(e) => handleAutoSelectChange(e, algorithmType)}
                                id={`checkbox-${algorithmType.replace(/\s+/g, '-').toLowerCase()}`}
                            >
                              <span title="自动选择辅助数据">
                                {'Auto Select'}
                              </span>
                            </Checkbox>
                        </Space>
                    </div>
                )}
            </div>
        );
    };

    // 渲染算法选择项
    const renderAlgorithmSelect = (algorithmType) => {
        const algorithm = getAlgorithm(algorithmType);
        const options = algorithmOptionsMap[algorithmType] || [];
        const mappedOptions = options.map(item => ({
            label: item.name,
            value: item.name
        }));
        
        return (
           <>
               {
                   algorithmType === "Initialization" ?
                       <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'baseline'}}>
                           <Select
                               showSearch
                               placeholder={`Select ${algorithmType}`}
                               optionFilterProp="value"
                               filterOption={filterOption}
                               style={{width: '40%'}}
                               options={mappedOptions}
                               value={algorithm.type || undefined}
                               onChange={(value) => handleAlgorithmTypeChange(value, algorithmType)}
                           />
                          <div>
                               <span style={{ marginLeft: '8px'}}>
                               Initial number:
                           </span>
                              <Input
                                  placeholder={`Type initial number`}
                                  style={{width: '40%', marginTop: '8px', marginLeft: '4px'}}
                                  type="number"
                                  value={algorithm.InitNum || 0}
                                  onChange={handleInitialNumberChange}
                              />
                          </div>
                       </div>
                       :
                       <Select
                           showSearch
                           placeholder={`Select ${algorithmType}`}
                           optionFilterProp="value"
                           filterOption={filterOption}
                           style={{width: '100%'}}
                           options={mappedOptions}
                           value={algorithm.type || undefined}
                           onChange={(value) => handleAlgorithmTypeChange(value, algorithmType)}
                       />
               }
           </>
        );
    };

    return (
        <div style={{width: "100%"}}>
            <Row gutter={[16, 16]}>
                {ALGORITHM_TYPES.map(algorithmType => (
                    <Col xs={24} md={12} lg={8} key={algorithmType}>
                        <div className="stat shadow" style={{
                            height: '100%',
                            boxShadow: '0 2px 8px rgba(0,0,0,0.1)',
                            borderRadius: '8px',
                            padding: '16px',
                            backgroundColor: 'white'
                        }}>
                            <div style={{display: 'flex', alignItems: 'center', gap: '8px'}}>
                                {algorithmType === "SearchSpace" &&
                                    <PartitionOutlined style={{fontSize: '24px', color: '#1890ff'}}/>}
                                {algorithmType === "Initialization" &&
                                    <ExperimentOutlined style={{fontSize: '24px', color: '#52c41a'}}/>}
                                {algorithmType === "Pretrain" &&
                                    <RobotOutlined style={{fontSize: '24px', color: '#722ed1'}}/>}
                                {algorithmType === "Model" &&
                                    <ApiOutlined style={{fontSize: '24px', color: '#fa8c16'}}/>}
                                {algorithmType === "AcquisitionFunction" &&
                                    <AreaChartOutlined style={{fontSize: '24px', color: '#eb2f96'}}/>}
                                {algorithmType === "Normalizer" &&
                                    <SlidersOutlined style={{fontSize: '24px', color: '#13c2c2'}}/>}
                                <span
                                    style={{fontSize: '16px', fontWeight: 'bold', color: '#333'}}>{ALGORITHM_TYPES_NAMES[algorithmType] || algorithmType}</span>
                            </div>
                            <div>
                                {renderAlgorithmSelect(algorithmType)}
                            </div>
                            <Divider style={{margin: '8px 0 4px 0'}}/>
                            {renderDataSelectionArea(algorithmType)}
                        </div>
                    </Col>
                ))}
            </Row>

            {/* SearchData modals for each algorithm type */}
            {ALGORITHM_TYPES.map(algorithmType => (
                <SearchData
                    key={algorithmType}
                    visible={activeModal === algorithmType}
                    onCancel={closeDataSelectionModal}
                    algorithmType={algorithmType}
                    onSelectData={handleSelectData}
                />
            ))}

            {/* 数据集预览模态窗口 */}
            <Modal
                title={`Selected Datasets for ${previewModal.algorithmType}`}
                open={previewModal.visible}
                onCancel={closePreviewModal}
                footer={[
                    <Button key="close" onClick={closePreviewModal}>
                        Close
                    </Button>
                ]}
                width={600}
            >
                <div style={{maxHeight: '400px', overflowY: 'auto'}}>
                    {previewModal.datasets.length > 0 ? (
                        <div>
                            <div style={{marginBottom: '16px'}}>
                                Total: {previewModal.datasets.length} dataset(s)
                            </div>
                            {previewModal.datasets.map((dataset, index) => (
                                <Tag
                                    key={index}
                                    style={{margin: '0 4px 8px 0'}}
                                    color="blue"
                                >
                                    {dataset.name || dataset.value || `Dataset ${index + 1}`}
                                </Tag>
                            ))}
                        </div>
                    ) : (
                        <div>No datasets selected</div>
                    )}
                </div>
            </Modal>
        </div>
    );
}

export default SelectAlgorithm;
