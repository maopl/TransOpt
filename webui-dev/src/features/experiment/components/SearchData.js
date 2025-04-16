import React, { useState } from "react";

import {
    Row,
    Col,
    Button,
    InputNumber,
    Slider,
    Space,
    Input,
    Form,
    ConfigProvider,
    Select,
    Modal,
    Divider,
    Typography,
    Checkbox,
    Card,
    Tooltip,
} from "antd";

const { Text } = Typography;

/**
 * SearchData component - Now works as both standalone and modal popup
 * @param {function} set_dataset - Function to set dataset data (used in standalone mode)
 * @param {boolean} visible - Whether the modal is visible
 * @param {function} onCancel - Function to close the modal
 * @param {string} algorithmType - Type of algorithm this search is for
 * @param {function} onSelectData - Callback for when data is selected with the algorithm context
 */
function SearchData({ set_dataset, visible = false, onCancel, algorithmType = "", onSelectData }) {
  const [form] = Form.useForm();
  const [isModalMode] = useState(!!onCancel); // Check if being used as modal
  const [searchResults, setSearchResults] = useState([]);
  const [loading, setLoading] = useState(false);
  const [selectedDatasets, setSelectedDatasets] = useState([]);

  // 处理搜索提交
  const onFinish = (values) => {
    // Add algorithm type if provided (when used as modal)
    const messageToSend = algorithmType ? { ...values, algorithmType } : values;
    console.log('Request data:', messageToSend);
    
    // 设置加载状态
    setLoading(true);
    
    fetch('http://localhost:5001/api/configuration/search_dataset', {
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
    .then(message => {
      console.log('Message from back-end:', message);
      setLoading(false);
      
      if (isModalMode) {
        // 显示搜索结果，让用户选择，而不是立即关闭弹窗
        if (message && message.datasets) {
          // 为数据集添加key属性用于表格渲染
          // 处理字符串数组，将每个字符串转成对象
          const datasetsWithKeys = message.datasets.map((dataset, index) => {
            // 字符串类型直接创建对象
            if (typeof dataset === 'string') {
              return {
                value: dataset,
                name: dataset,
                key: index.toString()
              };
            }
            // 处理其他基本类型
            else if (typeof dataset !== 'object' || dataset === null) {
              return {
                value: dataset,
                name: String(dataset),
                key: index.toString()
              };
            }
            // 对象类型保持结构并添加key
            return {
              ...dataset,
              key: index.toString()
            };
          });
          setSearchResults(datasetsWithKeys);
          setSelectedDatasets([]);
        }
      } else if (set_dataset) {
        // Original behavior when used standalone
        set_dataset(message);
      }
    })
    .catch((error) => {
      console.error('Error sending message:', error);
      setLoading(false);
      let errorMessage = error.message || error.error || 'Unknown error';
      Modal.error({
        title: 'Information',
        content: 'Error: ' + errorMessage
      });
    });
  };
  
  // 处理选择变化
  const handleCheckboxChange = (e, dataset) => {
    const { checked } = e.target;
    if (checked) {
      // 添加到选中列表
      setSelectedDatasets(prev => [...prev, dataset]);
    } else {
      // 从选中列表移除
      setSelectedDatasets(prev => 
        prev.filter(item => (item.key !== dataset.key))
      );
    }
  };
  
  // 检查数据集是否被选中
  const isDatasetSelected = (dataset) => {
    return selectedDatasets.some(item => item.key === dataset.key);
  };
  
  // 全选/取消全选
  const handleSelectAll = (e) => {
    if (e.target.checked) {
      setSelectedDatasets([...searchResults]);
    } else {
      setSelectedDatasets([]);
    }
  };
  
  // 处理确认选择
  const handleConfirmSelection = () => {
    if (selectedDatasets.length > 0 && onSelectData) {
      // 创建包含选定数据集的消息对象
      const selectedData = {
        isExact: true,
        datasets: selectedDatasets
      };
      
      // 调用回调函数并关闭弹窗
      onSelectData(selectedData, algorithmType);
      onCancel();
    } else {
      Modal.warning({
        title: 'Warning',
        content: 'Please select at least one dataset'
      });
    }
  };

  // The form content
  const formContent = (
    <ConfigProvider
      theme={{
        components: {
          Input: {
            addonBg: "white"
          },
        },
      }}  
    >
    <Form
      name="SearchData"
      form={form}
      onFinish={onFinish}
      style={{width:"100%"}}
      autoComplete="off"
    >
      {/* Display algorithm type when in modal mode */}
      {algorithmType && (
        <div style={{ marginBottom: 16 }}>
          <h5 style={{ color: "#1890ff" }}>Selecting data for: {algorithmType}</h5>
        </div>
      )}
      
      <Space className="space" style={{ display: 'flex'}} align="baseline">
        <Form.Item
          name="task_name"
          style={{flexGrow: 1}}
        >
          <Input addonBefore={"Dataset Name"}/>
        </Form.Item>
        <Form.Item
          name="num_variables"
          style={{flexGrow: 1}}
        >
          <Input addonBefore={"Num of Variables"}/>
        </Form.Item>
      </Space>
      <Space className="space" style={{ display: 'flex'}} align="baseline">
        <Form.Item
          name="variables_name"
          style={{flexGrow: 1}}
        >
          <Input addonBefore={"Variable Name"}/>
        </Form.Item>
        <Form.Item
          name="num_objectives"
          style={{flexGrow: 1}}
        >
          <Input addonBefore={"Num of Objectives"}/>
        </Form.Item>
      </Space>
      <h6 style={{color:"black"}}>Search method:</h6>
      <Space className="space" style={{ display: 'flex'}} align="baseline">
        <Form.Item
          name="search_method"
        >
          <Select style={{minWidth: 150}}
            options={[ {value: "Hash"},
                       {value: "Fuzzy"},
                       {value: "LSH"},
                   ]}
          />
        </Form.Item>
        <Form.Item>
          <Button type="primary" htmlType="submit" style={{width:"120px"}}>
            Search
          </Button>
        </Form.Item>
      </Space>
    </Form>
    </ConfigProvider>
  );

  // 根据类别生成分组的数据集列表
  const generateDatasetGroups = () => {
    if (!searchResults || searchResults.length === 0) return [];
    
    // 如果数据集有类别信息，按类别分组
    const categorizedDatasets = {};
    
    searchResults.forEach(dataset => {
      const category = dataset.category || 'General';
      if (!categorizedDatasets[category]) {
        categorizedDatasets[category] = [];
      }
      categorizedDatasets[category].push(dataset);
    });
    
    // 给每个类别创建一个卡片
    return Object.entries(categorizedDatasets).map(([category, datasets]) => (
      <Card 
        title={category} 
        key={category} 
        size="small" 
        style={{ marginBottom: 16 }}
        extra={
          <Checkbox 
            onChange={(e) => {
              // 选中/取消选中该类别的所有数据集
              if (e.target.checked) {
                setSelectedDatasets(prev => {
                  const filtered = prev.filter(item => 
                    !datasets.some(d => d.key === item.key)
                  );
                  return [...filtered, ...datasets];
                });
              } else {
                setSelectedDatasets(prev => 
                  prev.filter(item => 
                    !datasets.some(d => d.key === item.key)
                  )
                );
              }
            }}
            checked={datasets.every(dataset => isDatasetSelected(dataset))}
            indeterminate={datasets.some(dataset => isDatasetSelected(dataset)) && 
                         !datasets.every(dataset => isDatasetSelected(dataset))}
          >
            Select All
          </Checkbox>
        }
      >
        <div style={{ maxHeight: '200px', overflowY: 'auto' }}>
          <Row gutter={[16, 8]}>
            {datasets.map((dataset, index) => (
              <Col span={8} key={dataset.key || index}>
                <Checkbox 
                  onChange={(e) => handleCheckboxChange(e, dataset)}
                  checked={isDatasetSelected(dataset)}
                  style={{ width: '100%', whiteSpace: 'nowrap', overflow: 'hidden', textOverflow: 'ellipsis' }}
                >
                  <span title={''}>
                    {dataset.name || dataset.value || String(dataset)}
                  </span>
                </Checkbox>
              </Col>
            ))}
          </Row>
        </div>
      </Card>
    ));
  };
  
  // If used as a modal, wrap in Modal component
  if (isModalMode) {
    return (
      <Modal
        title={`Select Data for ${algorithmType}`}
        open={visible}
        onCancel={onCancel}
        width={800}
        footer={[
          <Button key="cancel" onClick={onCancel}>
            Cancel
          </Button>,
          <Button 
            key="confirm" 
            type="primary" 
            onClick={handleConfirmSelection}
            disabled={selectedDatasets.length === 0}
          >
            Confirm Selection
          </Button>,
        ]}
      >
        {formContent}
        
        {/* 搜索结果以Checkbox形式展示 */}
        {searchResults.length > 0 && (
          <div style={{ marginTop: 16 }}>
            <Divider orientation="left">
              Search Results
              <span style={{ marginLeft: 16 }}>
                <Checkbox 
                  onChange={handleSelectAll}
                  checked={selectedDatasets.length === searchResults.length && searchResults.length > 0}
                  indeterminate={selectedDatasets.length > 0 && selectedDatasets.length < searchResults.length}
                >
                  Select All
                </Checkbox>
              </span>
            </Divider>
            
            <div style={{ marginBottom: 16 }}>
              {generateDatasetGroups()}
            </div>
            
            {selectedDatasets.length > 0 && (
              <Text type="secondary" style={{ display: 'block', marginBottom: 8 }}>
                Selected {selectedDatasets.length} dataset(s)
              </Text>
            )}
          </div>
        )}
      </Modal>
    );
  }

  // Otherwise return just the form (original behavior)
  return formContent;
}

export default SearchData