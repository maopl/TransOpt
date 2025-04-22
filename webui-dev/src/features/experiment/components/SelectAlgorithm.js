import React, {useState, useEffect, useMemo} from "react";
import {
    PartitionOutlined,
    ExperimentOutlined,
    RobotOutlined,
    ApiOutlined,
    AreaChartOutlined,
    SlidersOutlined,
    SaveOutlined,
    DatabaseOutlined,
    EditOutlined,
    DeleteOutlined,
    TagsOutlined,
    EyeOutlined
} from '@ant-design/icons';
import {Button, Form, Select, Modal, Row, Col, Space, Tag, Divider, Typography, Tooltip, Checkbox} from "antd";

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
                             updateTable,
                             algorithmValue,
                             setAlgorithmValue,
                             transformedAlgorithmValue,
                             updateTransformedAlgorithmValue
                         }) {
    const [form] = Form.useForm();

    // Modal visibility states for each algorithm's data selection
    const [activeModal, setActiveModal] = useState(null);

    // 预览模态窗口状态
    const [previewModal, setPreviewModal] = useState({
        visible: false,
        algorithmType: '',
        datasets: []
    });
    
    // 初始化时从transformedAlgorithmValue同步autoSelect状态
    // useEffect(() => {
    //     if (transformedAlgorithmValue && transformedAlgorithmValue.algorithms) {
    //         const newAutoSelectState = {};
    //         transformedAlgorithmValue.algorithms.forEach(algorithm => {
    //             newAutoSelectState[algorithm.name] = algorithm.autoSelect || false;
    //         });
    //     }
    // }, [transformedAlgorithmValue]);

    /**
     * 算法对应的下拉选项
     * @type {{"Search Space", Initialization, Pretrain, Model, "Acquisition Function", Normalizer}}
     */
    const algorithmOptionsMap = useMemo(() => ({
        "SearchSpace": SearchSpaceOptions,
        "Initialization": InitializationOptions,
        "Pretrain": PretrainOptions,
        "Model": ModelOptions,
        "AcquisitionFunction": AcquisitionFunctionOptions,
        "Normalizer": NormalizerOptions
    }), [SearchSpaceOptions, InitializationOptions, PretrainOptions, ModelOptions, AcquisitionFunctionOptions, NormalizerOptions]);


    // 当表单数据变化时保存到localStorage
    const handleFormChange = (changedValues, allValues) => {
        setAlgorithmValue(allValues);
        localStorage.setItem('algorithmFormData', JSON.stringify(allValues));
        // 如果父组件提供了updateTable回调，则调用它
        if (updateTable) {
            updateTable(allValues);
        }
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
        const datasets = getSelectedDatasets(algorithmType);
        setPreviewModal({
            visible: true,
            algorithmType,
            datasets
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
        const updatedValues = {...algorithmValue};
        updatedValues[`${algorithmType}SelectedDatasets`] = datasetData.datasets;
        setAlgorithmValue(updatedValues);
        form.setFieldsValue(updatedValues);
        localStorage.setItem('algorithmFormData', JSON.stringify(updatedValues));
        if (updateTable) updateTable(updatedValues);
        
        // 同步更新transformedAlgorithmValue
        if (transformedAlgorithmValue && updateTransformedAlgorithmValue) {
            const updatedTransformedValue = {...transformedAlgorithmValue};
            const algorithmIndex = updatedTransformedValue.algorithms.findIndex(alg => alg.name === algorithmType);
            
            if (algorithmIndex !== -1) {
                // 从数据集中提取名称
                const auxiliaryData = datasetData.datasets.map(dataset => dataset.name || dataset.value);
                updatedTransformedValue.algorithms[algorithmIndex].auxiliaryData = auxiliaryData;
                updateTransformedAlgorithmValue(updatedTransformedValue);
            }
        }
    };

    // 获取特定算法的已选数据集
    const getSelectedDatasets = (algorithmType) => {
        return algorithmValue[`${algorithmType}SelectedDatasets`] || [];
    };

    // 清除数据集
    const clearSelectedDatasets = (algorithmType) => {
        const updatedValues = {...algorithmValue};
        updatedValues[`${algorithmType}SelectedDatasets`] = [];
        setAlgorithmValue(updatedValues);
        form.setFieldsValue(updatedValues);
        localStorage.setItem('algorithmFormData', JSON.stringify(updatedValues));
        if (updateTable) updateTable(updatedValues);
        
        // 同步更新transformedAlgorithmValue
        if (transformedAlgorithmValue && updateTransformedAlgorithmValue) {
            const updatedTransformedValue = {...transformedAlgorithmValue};
            const algorithmIndex = updatedTransformedValue.algorithms.findIndex(alg => alg.name === algorithmType);
            
            if (algorithmIndex !== -1) {
                updatedTransformedValue.algorithms[algorithmIndex].auxiliaryData = [];
                updateTransformedAlgorithmValue(updatedTransformedValue);
            }
        }
    };
    
    // 处理自动选择复选框变更
    const handleAutoSelectChange = (e, algorithmType) => {
        const checked = e.target.checked;
        console.log(`Auto Select for ${algorithmType} changed to: ${checked}`);
        
        // 直接操作Form的值以确保UI更新
        const currentFormValues = form.getFieldsValue();
        console.log('Current form values:', currentFormValues);
        
        if (transformedAlgorithmValue && updateTransformedAlgorithmValue) {
            // 必须创建深拷贝以确保React检测到变更
            const updatedTransformedValue = JSON.parse(JSON.stringify(transformedAlgorithmValue));
            console.log('Original transformed value:', updatedTransformedValue);
            
            const algorithmIndex = updatedTransformedValue.algorithms.findIndex(alg => alg.name === algorithmType);
            
            if (algorithmIndex !== -1) {
                // 更新自动选择状态
                updatedTransformedValue.algorithms[algorithmIndex].autoSelect = checked;
                console.log('Updated algorithm at index', algorithmIndex, 'to:', updatedTransformedValue.algorithms[algorithmIndex]);
                console.log('New transformed value:', updatedTransformedValue);
                
                // 提交到父组件更新状态
                updateTransformedAlgorithmValue(updatedTransformedValue);
            }
        }
    };
    
    // 获取特定算法的自动选择状态
    const getAutoSelectStatus = (algorithmType) => {
        if (transformedAlgorithmValue && transformedAlgorithmValue.algorithms) {
            const algorithm = transformedAlgorithmValue.algorithms.find(alg => alg.name === algorithmType);
            return algorithm?.autoSelect || false;
        }
        return false;
    };
    
    // 处理算法类型选择变更
    const handleAlgorithmTypeChange = (value, algorithmType) => {
        // 更新旧格式值
        const updatedValues = {...algorithmValue};
        updatedValues[algorithmType] = value;
        setAlgorithmValue(updatedValues);
        form.setFieldsValue(updatedValues);
        
        // 同步更新transformedAlgorithmValue
        if (transformedAlgorithmValue && updateTransformedAlgorithmValue) {
            const updatedTransformedValue = {...transformedAlgorithmValue};
            const algorithmIndex = updatedTransformedValue.algorithms.findIndex(alg => alg.name === algorithmType);
            
            if (algorithmIndex !== -1) {
                updatedTransformedValue.algorithms[algorithmIndex].type = value;
                updateTransformedAlgorithmValue(updatedTransformedValue);
            }
        }
    };

    // 渲染数据选择区域
    const renderDataSelectionArea = (algorithmType) => {
        const selectedDatasets = getSelectedDatasets(algorithmType);
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

    // 保留原有的提交逻辑，后续会重新处理
    const handleSubmit = () => {
        form
            .validateFields()
            .then(values => {
                // 保留原有网络请求代码，后续由用户重新处理
                fetch('/api/configuration/select_algorithm', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify(values),
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
                            content: 'Submit successfully!',
                        });
                    })
                    .catch(error => {
                        console.error('Error sending message:', error);
                        Modal.error({
                            title: 'Information',
                            content: 'Error: ' + error.message,
                        });
                    });
            })
            .catch(info => {
                console.log('Validate Failed:', info);
            });
    };

    /**
     * A helper function to render a form item with a select component.
     *
     * @param {string} name - The name of the form item.
     * @param {object[]} options - The options to be rendered in the select component.
     * Each option should have at least a `value` property and a `label` property.
     * @param {object[]} [rules=[]] - The validation rules for the form item.
     * @return {ReactElement} The rendered form item.
     */
    const renderFormItem = (name, options, rules = []) => {
        return (
            <Form.Item
                name={name}
                rules={rules}
                noStyle
            >
                <Select
                    showSearch
                    placeholder={`Select ${name}`}
                    optionFilterProp="value"
                    filterOption={filterOption}
                    style={{width: '100%'}}
                    options={options}
                    onChange={(value) => handleAlgorithmTypeChange(value, name)}
                />
            </Form.Item>
        );
    };

    return (
        <Form
            form={form}
            onValuesChange={handleFormChange}
            initialValues={algorithmValue}
            layout="vertical"
            style={{width: "100%"}}
        >
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
                            <div className="stat-value">
                                {renderFormItem(algorithmType, algorithmOptionsMap[algorithmType].map(item => ({label: item.name, value: item.name})), [{
                                    required: true,
                                    message: `Please select a ${algorithmType}!`
                                }])}
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
        </Form>
    );
}

export default SelectAlgorithm;
