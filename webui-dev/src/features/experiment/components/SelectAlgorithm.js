import React, { useState, useEffect } from "react";
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
import { Button, Form, Select, Modal, Row, Col, Space, Tag, Divider, Typography, Tooltip } from "antd";
import DashboardStats from './DashboardStats';
import SearchData from './SearchData';

const { Text } = Typography;

const filterOption = (input, option) =>
  (option?.value ?? '').toLowerCase().includes(input.toLowerCase());

function SelectAlgorithm({ SpaceRefiner, Sampler, Pretrain, Model, ACF, Normalizer, updateTable }) {
  const [form] = Form.useForm();
  
  // Modal visibility states for each algorithm's data selection
  const [activeModal, setActiveModal] = useState(null);
  
  // 预览模态窗口状态
  const [previewModal, setPreviewModal] = useState({
    visible: false,
    algorithmType: '',
    datasets: []
  });
  
  const [formValues, setFormValues] = useState({
    SpaceRefiner: SpaceRefiner[0]?.name || '',
    Sampler: Sampler[0]?.name || '',
    Pretrain: Pretrain[0]?.name || '',
    Model: Model[0]?.name || '',
    ACF: ACF[0]?.name || '',
    Normalizer: Normalizer[0]?.name || '',
    SpaceRefinerParameters: '',
    SpaceRefinerDataSelector: 'None',
    SpaceRefinerDataSelectorParameters: '',
    SamplerParameters: '',
    SamplerInitNum: '11',
    SamplerDataSelector: 'None',
    SamplerDataSelectorParameters: '',
    PretrainParameters: '',
    PretrainDataSelector: 'None',
    PretrainDataSelectorParameters: '',
    ModelParameters: '',
    ModelDataSelector: 'None',
    ModelDataSelectorParameters: '',
    ACFParameters: '',
    ACFDataSelector: 'None',
    ACFDataSelectorParameters: '',
    NormalizerParameters: '',
    NormalizerDataSelector: 'None',
    NormalizerDataSelectorParameters: '',
  });
  
  // 初始化时从localStorage读取数据
  useEffect(() => {
    const savedData = localStorage.getItem('algorithmFormData');
    if (savedData) {
      const parsedData = JSON.parse(savedData);
      setFormValues(parsedData);
      form.setFieldsValue(parsedData);
    }
  }, []);
  
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
    const { datasets } = datasetData;
    
    // Update form values based on which algorithm's data was selected
    const updatedValues = { ...formValues };
    
    if (algorithmType === 'SpaceRefiner') {
      updatedValues.SpaceRefinerDataSelector = 'Custom';
      // 存储完整的数据集信息，方便展示和编辑
      updatedValues.SpaceRefinerDataSelectorParameters = JSON.stringify(datasets);
      updatedValues.SpaceRefinerSelectedDatasets = datasets; // 直接存储数据集对象数组
    } else if (algorithmType === 'Sampler') {
      updatedValues.SamplerDataSelector = 'Custom';
      updatedValues.SamplerDataSelectorParameters = JSON.stringify(datasets);
      updatedValues.SamplerSelectedDatasets = datasets;
    } else if (algorithmType === 'Pretrain') {
      updatedValues.PretrainDataSelector = 'Custom';
      updatedValues.PretrainDataSelectorParameters = JSON.stringify(datasets);
      updatedValues.PretrainSelectedDatasets = datasets;
    } else if (algorithmType === 'Model') {
      updatedValues.ModelDataSelector = 'Custom';
      updatedValues.ModelDataSelectorParameters = JSON.stringify(datasets);
      updatedValues.ModelSelectedDatasets = datasets;
    } else if (algorithmType === 'ACF') {
      updatedValues.ACFDataSelector = 'Custom';
      updatedValues.ACFDataSelectorParameters = JSON.stringify(datasets);
      updatedValues.ACFSelectedDatasets = datasets;
    } else if (algorithmType === 'Normalizer') {
      updatedValues.NormalizerDataSelector = 'Custom';
      updatedValues.NormalizerDataSelectorParameters = JSON.stringify(datasets);
      updatedValues.NormalizerSelectedDatasets = datasets;
    }
    
    // Update form with new values
    form.setFieldsValue(updatedValues);
    setFormValues(updatedValues);
    localStorage.setItem('algorithmFormData', JSON.stringify(updatedValues));
    
    // Notify parent component
    if (updateTable) {
      updateTable(updatedValues);
    }
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
          style={{ width: '100%' }}
          options={options}
        />
      </Form.Item>
    );
  };
  
  // 清除特定算法的已选数据集
  const clearSelectedDatasets = (algorithmType) => {
    const updatedValues = { ...formValues };
    
    if (algorithmType === 'SpaceRefiner') {
      updatedValues.SpaceRefinerDataSelector = 'None';
      updatedValues.SpaceRefinerDataSelectorParameters = '';
      updatedValues.SpaceRefinerSelectedDatasets = undefined;
    } else if (algorithmType === 'Sampler') {
      updatedValues.SamplerDataSelector = 'None';
      updatedValues.SamplerDataSelectorParameters = '';
      updatedValues.SamplerSelectedDatasets = undefined;
    } else if (algorithmType === 'Pretrain') {
      updatedValues.PretrainDataSelector = 'None';
      updatedValues.PretrainDataSelectorParameters = '';
      updatedValues.PretrainSelectedDatasets = undefined;
    } else if (algorithmType === 'Model') {
      updatedValues.ModelDataSelector = 'None';
      updatedValues.ModelDataSelectorParameters = '';
      updatedValues.ModelSelectedDatasets = undefined;
    } else if (algorithmType === 'ACF') {
      updatedValues.ACFDataSelector = 'None';
      updatedValues.ACFDataSelectorParameters = '';
      updatedValues.ACFSelectedDatasets = undefined;
    } else if (algorithmType === 'Normalizer') {
      updatedValues.NormalizerDataSelector = 'None';
      updatedValues.NormalizerDataSelectorParameters = '';
      updatedValues.NormalizerSelectedDatasets = undefined;
    }
    
    // 更新表单值
    form.setFieldsValue(updatedValues);
    setFormValues(updatedValues);
    localStorage.setItem('algorithmFormData', JSON.stringify(updatedValues));
    
    // 通知父组件
    if (updateTable) {
      updateTable(updatedValues);
    }
  };
  
  // 获取特定算法的已选数据集
  const getSelectedDatasets = (algorithmType) => {
    if (algorithmType === 'SpaceRefiner') {
      return formValues.SpaceRefinerSelectedDatasets || [];
    } else if (algorithmType === 'Sampler') {
      return formValues.SamplerSelectedDatasets || [];
    } else if (algorithmType === 'Pretrain') {
      return formValues.PretrainSelectedDatasets || [];
    } else if (algorithmType === 'Model') {
      return formValues.ModelSelectedDatasets || [];
    } else if (algorithmType === 'ACF') {
      return formValues.ACFSelectedDatasets || [];
    } else if (algorithmType === 'Normalizer') {
      return formValues.NormalizerSelectedDatasets || [];
    }
    return [];
  };
  
  // 为每个算法卡片渲染数据选择区域（包括预览和操作按钮）
  const renderDataSelectionArea = (algorithmType) => {
    const selectedDatasets = getSelectedDatasets(algorithmType);
    const hasSelectedData = selectedDatasets.length > 0;
    
    return (
      <div style={{ marginTop: '8px' }}>
        {!hasSelectedData ? (
          // 没有选择数据时显示选择按钮
          <Button 
            type="default"
            size="small"
            icon={<DatabaseOutlined />}
            onClick={() => openDataSelectionModal(algorithmType)}
          >
            Select Auxiliary Data
          </Button>
        ) : (
          // 已选择数据时显示数据预览和操作按钮
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Text style={{ marginRight: '8px' }}>
              <TagsOutlined /> Selected {selectedDatasets.length} dataset(s)
            </Text>
            <Space size="small">
              <Tooltip title="View datasets">
                <Button 
                  type="text" 
                  size="small" 
                  icon={<EyeOutlined />} 
                  onClick={() => openPreviewModal(algorithmType)}
                />
              </Tooltip>
              <Tooltip title="Edit selection">
                <Button 
                  type="text" 
                  size="small" 
                  icon={<EditOutlined />} 
                  onClick={() => openDataSelectionModal(algorithmType)}
                />
              </Tooltip>
              <Tooltip title="Clear selection">
                <Button 
                  type="text" 
                  size="small" 
                  danger 
                  icon={<DeleteOutlined />} 
                  onClick={() => clearSelectedDatasets(algorithmType)}
                />
              </Tooltip>
            </Space>
          </div>
        )}
      </div>
    );
  };

  return (
    <Form
      form={form}
      onValuesChange={handleFormChange}
      initialValues={formValues}
      layout="vertical"
      style={{ width: "100%" }}
    >
      <Row gutter={[16, 16]}>
        <Col xs={24} md={12} lg={8}>
          <div className="stats shadow" style={{ height: '100%', boxShadow: '0 2px 8px rgba(0,0,0,0.1)', borderRadius: '8px', padding: '16px', backgroundColor: 'white' }}>
            <div className="stat">
              <div className="stat-figure text-primary">
                <PartitionOutlined style={{ fontSize: '24px', color: '#1890ff' }} />
              </div>
              <div className="stat-title" style={{ fontSize: '16px', fontWeight: 'bold', color: '#333' }}>Narrow Search Space</div>
              <div className="stat-value">
                {renderFormItem('SpaceRefiner', SpaceRefiner.map(item => ({ value: item.name })), [{ required: true, message: 'Please select a SpaceRefiner!' }])}
              </div>
              <div className="stat-desc" style={{ fontSize: '12px', color: '#666' }}>
                <div style={{ width: '100%' }}>
                  <div>Details of SpaceRefiner</div>
                  <Divider style={{ margin: '8px 0' }} />
                  {renderDataSelectionArea('SpaceRefiner')}
                </div>
              </div>
            </div>
          </div>
        </Col>
        
        <Col xs={24} md={12} lg={8}>
          <div className="stats shadow" style={{ height: '100%', boxShadow: '0 2px 8px rgba(0,0,0,0.1)', borderRadius: '8px', padding: '16px', backgroundColor: 'white' }}>
            <div className="stat">
              <div className="stat-figure text-primary">
                <ExperimentOutlined style={{ fontSize: '24px', color: '#52c41a' }} />
              </div>
              <div className="stat-title" style={{ fontSize: '16px', fontWeight: 'bold', color: '#333' }}>Initialization</div>
              <div className="stat-value">
                {renderFormItem('Sampler', Sampler.map(item => ({ value: item.name })), [{ required: true, message: 'Please select a Sampler!' }])}
              </div>
              <div className="stat-desc" style={{ fontSize: '12px', color: '#666' }}>
                <div style={{ width: '100%' }}>
                  <div>Details of Initialization</div>
                  <Form.Item
                    name="SamplerInitNum"
                    style={{ marginBottom: 0, marginTop: '8px' }}
                  >
                    <Select
                      style={{ width: '100%' }}
                      options={[11, 21, 31, 41, 51].map(num => ({ value: num.toString() }))}
                    />
                  </Form.Item>
                  <Divider style={{ margin: '8px 0' }} />
                  {renderDataSelectionArea('Sampler')}
                </div>
              </div>
            </div>
          </div>
        </Col>
        
        <Col xs={24} md={12} lg={8}>
          <div className="stats shadow" style={{ height: '100%', boxShadow: '0 2px 8px rgba(0,0,0,0.1)', borderRadius: '8px', padding: '16px', backgroundColor: 'white' }}>
            <div className="stat">
              <div className="stat-figure text-primary">
                <RobotOutlined style={{ fontSize: '24px', color: '#722ed1' }} />
              </div>
              <div className="stat-title" style={{ fontSize: '16px', fontWeight: 'bold', color: '#333' }}>Pretrain</div>
              <div className="stat-value">
                {renderFormItem('Pretrain', Pretrain.map(item => ({ value: item.name })), [{ required: true, message: 'Please select a Pretrain!' }])}
              </div>
              <div className="stat-desc" style={{ fontSize: '12px', color: '#666' }}>
                <div style={{ width: '100%' }}>
                  <div>Details of Pretrain</div>
                  <Divider style={{ margin: '8px 0' }} />
                  {renderDataSelectionArea('Pretrain')}
                </div>
              </div>
            </div>
          </div>
        </Col>
        
        <Col xs={24} md={12} lg={8}>
          <div className="stats shadow" style={{ height: '100%', boxShadow: '0 2px 8px rgba(0,0,0,0.1)', borderRadius: '8px', padding: '16px', backgroundColor: 'white' }}>
            <div className="stat">
              <div className="stat-figure text-primary">
                <ApiOutlined style={{ fontSize: '24px', color: '#fa8c16' }} />
              </div>
              <div className="stat-title" style={{ fontSize: '16px', fontWeight: 'bold', color: '#333' }}>Model</div>
              <div className="stat-value">
                {renderFormItem('Model', Model.map(item => ({ value: item.name })), [{ required: true, message: 'Please select a Model!' }])}
              </div>
              <div className="stat-desc" style={{ fontSize: '12px', color: '#666' }}>
                <div style={{ width: '100%' }}>
                  <div>Details of Model</div>
                  <Divider style={{ margin: '8px 0' }} />
                  {renderDataSelectionArea('Model')}
                </div>
              </div>
            </div>
          </div>
        </Col>
        
        <Col xs={24} md={12} lg={8}>
          <div className="stats shadow" style={{ height: '100%', boxShadow: '0 2px 8px rgba(0,0,0,0.1)', borderRadius: '8px', padding: '16px', backgroundColor: 'white' }}>
            <div className="stat">
              <div className="stat-figure text-primary">
                <AreaChartOutlined style={{ fontSize: '24px', color: '#eb2f96' }} />
              </div>
              <div className="stat-title" style={{ fontSize: '16px', fontWeight: 'bold', color: '#333' }}>Acquisition Function</div>
              <div className="stat-value">
                {renderFormItem('ACF', ACF.map(item => ({ value: item.name })), [{ required: true, message: 'Please select an ACF!' }])}
              </div>
              <div className="stat-desc" style={{ fontSize: '12px', color: '#666' }}>
                <div style={{ width: '100%' }}>
                  <div>Details of ACF</div>
                  <Divider style={{ margin: '8px 0' }} />
                  {renderDataSelectionArea('ACF')}
                </div>
              </div>
            </div>
          </div>
        </Col>
        
        <Col xs={24} md={12} lg={8}>
          <div className="stats shadow" style={{ height: '100%', boxShadow: '0 2px 8px rgba(0,0,0,0.1)', borderRadius: '8px', padding: '16px', backgroundColor: 'white' }}>
            <div className="stat">
              <div className="stat-figure text-primary">
                <SlidersOutlined style={{ fontSize: '24px', color: '#13c2c2' }} />
              </div>
              <div className="stat-title" style={{ fontSize: '16px', fontWeight: 'bold', color: '#333' }}>Normalizer</div>
              <div className="stat-value">
                {renderFormItem('Normalizer', Normalizer.map(item => ({ value: item.name })), [{ required: true, message: 'Please select a Normalizer!' }])}
              </div>
              <div className="stat-desc" style={{ fontSize: '12px', color: '#666' }}>
                <div style={{ width: '100%' }}>
                  <div>Details of Normalizer</div>
                  <Divider style={{ margin: '8px 0' }} />
                  {renderDataSelectionArea('Normalizer')}
                </div>
              </div>
            </div>
          </div>
        </Col>
      </Row>

      <div style={{ marginTop: '20px', display: 'flex', justifyContent: 'flex-end' }}>
        {/* <Button 
          type="primary" 
          onClick={handleSubmit} 
          icon={<SaveOutlined />}
          style={{ width: "150px", backgroundColor: 'rgb(53, 162, 235)' }}
        >
          Apply
        </Button> */}
      </div>
      
      {/* SearchData modals for each algorithm type */}
      <SearchData 
        visible={activeModal === 'SpaceRefiner'}
        onCancel={closeDataSelectionModal}
        algorithmType="SpaceRefiner"
        onSelectData={handleSelectData}
      />
      <SearchData 
        visible={activeModal === 'Sampler'}
        onCancel={closeDataSelectionModal}
        algorithmType="Sampler"
        onSelectData={handleSelectData}
      />
      <SearchData 
        visible={activeModal === 'Pretrain'}
        onCancel={closeDataSelectionModal}
        algorithmType="Pretrain"
        onSelectData={handleSelectData}
      />
      <SearchData 
        visible={activeModal === 'Model'}
        onCancel={closeDataSelectionModal}
        algorithmType="Model"
        onSelectData={handleSelectData}
      />
      <SearchData 
        visible={activeModal === 'ACF'}
        onCancel={closeDataSelectionModal}
        algorithmType="ACF"
        onSelectData={handleSelectData}
      />
      <SearchData 
        visible={activeModal === 'Normalizer'}
        onCancel={closeDataSelectionModal}
        algorithmType="Normalizer"
        onSelectData={handleSelectData}
      />
      
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
        <div style={{ maxHeight: '400px', overflowY: 'auto' }}>
          {previewModal.datasets.length > 0 ? (
            <div>
              <div style={{ marginBottom: '16px' }}>
                Total: {previewModal.datasets.length} dataset(s)
              </div>
              {previewModal.datasets.map((dataset, index) => (
                <Tag 
                  key={index} 
                  style={{ margin: '0 4px 8px 0' }}
                  color="blue"
                >
                  {dataset.name || dataset.value || `Dataset ${index+1}`}
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
