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
    "Search Space",
    "Initialization",
    "Pretrain",
    "Model",
    "Acquisition Function",
    "Normalizer"
];

function SelectAlgorithm({
                             SearchSpaceOptions,
                             InitializationOptions,
                             PretrainOptions,
                             ModelOptions,
                             AcquisitionFunctionOptions,
                             NormalizerOptions,
                             updateTable
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

    /**
     * 算法对应的下拉选项
     * @type {{"Search Space", Initialization, Pretrain, Model, "Acquisition Function", Normalizer}}
     */
    const algorithmOptionsMap = useMemo(() => ({
        "Search Space": SearchSpaceOptions,
        "Initialization": InitializationOptions,
        "Pretrain": PretrainOptions,
        "Model": ModelOptions,
        "Acquisition Function": AcquisitionFunctionOptions,
        "Normalizer": NormalizerOptions
    }), [SearchSpaceOptions, InitializationOptions, PretrainOptions, ModelOptions, AcquisitionFunctionOptions, NormalizerOptions]);

    // 统一初始formValues结构
    const [formValues, setFormValues] = useState({
        "Search Space": SearchSpaceOptions?.[0]?.name || '',
        "Initialization": InitializationOptions?.[0]?.name || '',
        "Pretrain": PretrainOptions?.[0]?.name || '',
        "Model": ModelOptions?.[0]?.name || '',
        "Acquisition Function": AcquisitionFunctionOptions?.[0]?.name || '',
        "Normalizer": NormalizerOptions?.[0]?.name || '',
        // 下面是各自的数据集等参数
        "Search SpaceSelectedDatasets": [],
        "InitializationSelectedDatasets": [],
        "PretrainSelectedDatasets": [],
        "ModelSelectedDatasets": [],
        "Acquisition FunctionSelectedDatasets": [],
        "NormalizerSelectedDatasets": [],
        // 你可以继续添加其它参数
    });

    // 初始化时从localStorage读取数据
    // useEffect(() => {
    //     const savedData = localStorage.getItem('algorithmFormData');
    //     if (savedData) {
    //         const parsedData = JSON.parse(savedData);
    //         setFormValues(parsedData);
    //         form.setFieldsValue(parsedData);
    //     }
    // }, []);

    // 当表单数据变化时保存到localStorage
    const handleFormChange = (changedValues, allValues) => {
        setFormValues(allValues);
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
        const updatedValues = {...formValues};
        updatedValues[`${algorithmType}SelectedDatasets`] = datasetData.datasets;
        setFormValues(updatedValues);
        form.setFieldsValue(updatedValues);
        localStorage.setItem('algorithmFormData', JSON.stringify(updatedValues));
        if (updateTable) updateTable(updatedValues);
    };

    // 获取特定算法的已选数据集
    const getSelectedDatasets = (algorithmType) => {
        return formValues[`${algorithmType}SelectedDatasets`] || [];
    };

    // 清除数据集
    const clearSelectedDatasets = (algorithmType) => {
        const updatedValues = {...formValues};
        updatedValues[`${algorithmType}SelectedDatasets`] = [];
        setFormValues(updatedValues);
        form.setFieldsValue(updatedValues);
        localStorage.setItem('algorithmFormData', JSON.stringify(updatedValues));
        if (updateTable) updateTable(updatedValues);
    };

    // 渲染数据选择区域
    const renderDataSelectionArea = (algorithmType) => {
        const selectedDatasets = getSelectedDatasets(algorithmType);
        const hasSelectedData = selectedDatasets.length > 0;
        return (
            <div style={{marginTop: '8px'}}>
                {!hasSelectedData ? (
                    <div style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
                        <Button
                            type="default"
                            size="small"
                            icon={<DatabaseOutlined/>}
                            onClick={() => openDataSelectionModal(algorithmType)}
                        >
                            Select Auxiliary Data
                        </Button>
                        <Checkbox>
                          <span title={''}>
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
                                />
                            </Tooltip>
                            <Tooltip title="编辑选择">
                                <Button
                                    type="text"
                                    size="small"
                                    icon={<EditOutlined/>}
                                    onClick={() => openDataSelectionModal(algorithmType)}
                                />
                            </Tooltip>
                            <Tooltip title="清除选择">
                                <Button
                                    type="text"
                                    size="small"
                                    danger
                                    icon={<DeleteOutlined/>}
                                    onClick={() => clearSelectedDatasets(algorithmType)}
                                />
                            </Tooltip>
                            <Checkbox>
                              <span title={''}>
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
        console.log('options', options)
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
                />
            </Form.Item>
        );
    };

    return (
        <Form
            form={form}
            onValuesChange={handleFormChange}
            initialValues={formValues}
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
                                {algorithmType === "Search Space" &&
                                    <PartitionOutlined style={{fontSize: '24px', color: '#1890ff'}}/>}
                                {algorithmType === "Initialization" &&
                                    <ExperimentOutlined style={{fontSize: '24px', color: '#52c41a'}}/>}
                                {algorithmType === "Pretrain" &&
                                    <RobotOutlined style={{fontSize: '24px', color: '#722ed1'}}/>}
                                {algorithmType === "Model" &&
                                    <ApiOutlined style={{fontSize: '24px', color: '#fa8c16'}}/>}
                                {algorithmType === "Acquisition Function" &&
                                    <AreaChartOutlined style={{fontSize: '24px', color: '#eb2f96'}}/>}
                                {algorithmType === "Normalizer" &&
                                    <SlidersOutlined style={{fontSize: '24px', color: '#13c2c2'}}/>}
                                <span
                                    style={{fontSize: '16px', fontWeight: 'bold', color: '#333'}}>{algorithmType}</span>
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
