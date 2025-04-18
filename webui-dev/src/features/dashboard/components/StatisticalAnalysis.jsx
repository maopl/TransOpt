import React, { useState } from "react";
import { Button, Modal, Form, Input, Select, Table, Space, Tag, Row, Col, Divider, Popconfirm } from "antd";
import { LineChartOutlined, PlusOutlined, EditOutlined, DeleteOutlined } from "@ant-design/icons";
import Box from "../../analytics/charts/Box";
import LineChart from "../../analytics/components/LineChart";

const defaultForm = {
  TaskName: '', NumObjs: '', NumVars: '', Fidelity: '', Workload: '', Seed: '',
  Refiner: '', Sampler: '', Pretrain: '', Model: '', ACF: '', Normalizer: ''
};
const mockSelections = {
  Refiner: [
    'NoRefiner', 'RandomEmbedding', 'PCA', 'AutoEncoder', 'VAE', 'LDA', 'KernelPCA', 'tSNE', 'UMAP', 'Isomap', 'LaplacianEigenmaps'
  ],
  Sampler: [
    'RandomSampler', 'GridSampler', 'LatinHypercube', 'Sobol', 'Halton', 'Hammersley', 'BayesOpt', 'TPE', 'CMAES', 'Genetic', 'SMAC'
  ],
  Pretrain: [
    'None', 'WarmStart', 'TransferLearning', 'MetaLearning', 'SelfSupervised', 'Contrastive', 'AutoML', 'Ensemble'
  ],
  Model: [
    'RandomForest', 'GaussianProcess', 'SVM', 'XGBoost', 'LightGBM', 'CatBoost', 'MLP', 'CNN', 'RNN', 'Transformer', 'LinearRegression'
  ],
  ACF: [
    'EI', 'PI', 'UCB', 'Thompson', 'EntropySearch', 'MES', 'PES', 'LCB', 'ProbabilityImprovement', 'ExpectedImprovement', 'KnowledgeGradient'
  ],
  Normalizer: [
    'None', 'Standard', 'MinMax', 'Robust', 'Quantile', 'Power', 'ZScore', 'MaxAbs', 'L2', 'L1'
  ]
};
const fieldLabels = {
  TaskName: 'Task Name', NumObjs: 'Objectives', NumVars: 'Variables', Fidelity: 'Fidelity', Workload: 'Workload', Seed: 'Seed',
  Refiner: 'Refiner', Sampler: 'Sampler', Pretrain: 'Pretrain', Model: 'Model', ACF: 'Acquisition Function', Normalizer: 'Normalizer'
};

const StatisticalAnalysis = () => {
  const [visible, setVisible] = useState(false);
  const [forms, setForms] = useState([]); // Group list
  const [editingIdx, setEditingIdx] = useState(null); // Current editing index
  const [modalForm, setModalForm] = useState({ ...defaultForm }); // Modal form
  const [formModalVisible, setFormModalVisible] = useState(false);
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState([]);

  // Open add/edit modal
  const openFormModal = (idx = null) => {
    setEditingIdx(idx);
    setModalForm(idx !== null ? { ...forms[idx] } : { ...defaultForm });
    setFormModalVisible(true);
  };
  // Close modal
  const closeFormModal = () => {
    setFormModalVisible(false);
    setEditingIdx(null);
    setModalForm({ ...defaultForm });
  };
  // Submit modal form
  const handleFormModalOk = () => {
    if (editingIdx !== null) {
      const newForms = forms.map((f, i) => i === editingIdx ? { ...modalForm } : f);
      setForms(newForms);
    } else {
      setForms([...forms, { ...modalForm }]);
    }
    closeFormModal();
  };
  // Delete group
  const removeForm = idx => setForms(forms.filter((_, i) => i !== idx));

  // Batch search
  const handleSearch = async () => {
    setLoading(true);
    try {
      const res = await Promise.all(forms.map(async (form) => {
        const resp = await fetch('http://localhost:5001/api/comparison/choose_task', {
          method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify([form])
        });
        return resp.ok ? resp.json() : { BoxData: null, TrajectoryData: [] };
      }));
      setResults(res.map(r => ({ BoxData: r.BoxData, TrajectoryData: r.TrajectoryData })));
    } catch (e) { setResults([]); }
    setLoading(false);
  };

  // Table columns
  const columns = [
    ...Object.keys(fieldLabels).map(key => ({
      title: fieldLabels[key],
      dataIndex: key,
      render: v => v ? <Tag color="blue">{v}</Tag> : <span style={{ color: '#bbb' }}>-</span>,
      ellipsis: true,
    })),
    {
      title: 'Action',
      key: 'action',
      width: 80,
      render: (_, __, idx) => (
        <Space>
          <Button icon={<EditOutlined />} size="small" onClick={() => openFormModal(idx)}></Button>
          <Popconfirm title="Are you sure to delete this group?" onConfirm={() => removeForm(idx)} okText="Delete" cancelText="Cancel">
            <Button icon={<DeleteOutlined />} size="small" danger></Button>
          </Popconfirm>
        </Space>
      )
    }
  ];

  return (
    <div style={{ display: 'flex', justifyContent: 'right', marginTop: 25 }}>
      <Button type="primary" size="large" style={{ fontSize: 22, padding: '20px 30px' }}
              icon={<LineChartOutlined />} onClick={() => setVisible(true)}>
        Statistical Analysis
      </Button>
      <Modal
        open={visible}
        onCancel={() => setVisible(false)}
        title="Statistical Analysis"
        width={1200}
        footer={null}
        bodyStyle={{ padding: 24 }}
        destroyOnClose
      >
        {/* Group list and add button */}
        <div style={{ marginBottom: 24 }}>
          <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 8 }}>
            <span style={{ fontWeight: 'bold', fontSize: 16 }}>Group List</span>
            <Button icon={<PlusOutlined />} type="dashed" onClick={() => openFormModal()}>Add Group</Button>
          </div>
          <Table
            dataSource={forms}
            columns={columns}
            rowKey={(_, idx) => idx}
            pagination={false}
            bordered
            size="small"
            scroll={{ x: true }}
            locale={{ emptyText: 'No groups yet. Please add one.' }}
          />
          <div style={{ textAlign: 'right', marginTop: 16 }}>
            <Button type="primary" loading={loading} disabled={forms.length === 0} onClick={handleSearch}>Search</Button>
          </div>
        </div>
        <Divider />
        {/* Results area */}
        <div>
          {results.length === 0 && <div style={{ textAlign: 'center', color: '#bbb', padding: '40px 0' }}>No data. Please search first.</div>}
          {results.length > 0 && results.map((res, idx) => (
            <div key={idx} style={{ marginBottom: 48, background: '#fff', borderRadius: 8, boxShadow: '0 1px 3px #eee', padding: 24 }}>
              <div style={{ fontWeight: 'bold', marginBottom: 12 }}>Group {idx + 1}</div>
              <Row gutter={24}>
                <Col span={12}><LineChart TrajectoryData={res.TrajectoryData || []} /></Col>
                <Col span={12}><Box BoxData={res.BoxData} /></Col>
              </Row>
            </div>
          ))}
        </div>
        {/* Add/Edit group modal */}
        <Modal
          open={formModalVisible}
          onCancel={closeFormModal}
          title={editingIdx !== null ? 'Edit Group' : 'Add Group'}
          onOk={handleFormModalOk}
          okText={editingIdx !== null ? 'Save' : 'Add'}
          destroyOnClose
          maskClosable={false}
        >
          <Form layout="vertical">
            <Row gutter={16}>
              <Col span={12}><Form.Item label="Task Name"><Input value={modalForm.TaskName} onChange={e => setModalForm(f => ({ ...f, TaskName: e.target.value }))} /></Form.Item></Col>
              <Col span={12}><Form.Item label="Objectives"><Input value={modalForm.NumObjs} onChange={e => setModalForm(f => ({ ...f, NumObjs: e.target.value }))} /></Form.Item></Col>
            </Row>
            <Row gutter={16}>
              <Col span={12}><Form.Item label="Variables"><Input value={modalForm.NumVars} onChange={e => setModalForm(f => ({ ...f, NumVars: e.target.value }))} /></Form.Item></Col>
              <Col span={12}><Form.Item label="Fidelity"><Input value={modalForm.Fidelity} onChange={e => setModalForm(f => ({ ...f, Fidelity: e.target.value }))} /></Form.Item></Col>
            </Row>
            <Row gutter={16}>
              <Col span={12}><Form.Item label="Workload"><Input value={modalForm.Workload} onChange={e => setModalForm(f => ({ ...f, Workload: e.target.value }))} /></Form.Item></Col>
              <Col span={12}><Form.Item label="Seed"><Input value={modalForm.Seed} onChange={e => setModalForm(f => ({ ...f, Seed: e.target.value }))} /></Form.Item></Col>
            </Row>
            <Row gutter={16}>
              <Col span={12}><Form.Item label="Refiner"><Select value={modalForm.Refiner} options={mockSelections.Refiner.map(v => ({ label: v, value: v }))} onChange={v => setModalForm(f => ({ ...f, Refiner: v }))} allowClear showSearch /></Form.Item></Col>
              <Col span={12}><Form.Item label="Sampler"><Select value={modalForm.Sampler} options={mockSelections.Sampler.map(v => ({ label: v, value: v }))} onChange={v => setModalForm(f => ({ ...f, Sampler: v }))} allowClear showSearch /></Form.Item></Col>
            </Row>
            <Row gutter={16}>
              <Col span={12}><Form.Item label="Pretrain"><Select value={modalForm.Pretrain} options={mockSelections.Pretrain.map(v => ({ label: v, value: v }))} onChange={v => setModalForm(f => ({ ...f, Pretrain: v }))} allowClear showSearch /></Form.Item></Col>
              <Col span={12}><Form.Item label="Model"><Select value={modalForm.Model} options={mockSelections.Model.map(v => ({ label: v, value: v }))} onChange={v => setModalForm(f => ({ ...f, Model: v }))} allowClear showSearch /></Form.Item></Col>
            </Row>
            <Row gutter={16}>
              <Col span={12}><Form.Item label="Acquisition Function"><Select value={modalForm.ACF} options={mockSelections.ACF.map(v => ({ label: v, value: v }))} onChange={v => setModalForm(f => ({ ...f, ACF: v }))} allowClear showSearch /></Form.Item></Col>
              <Col span={12}><Form.Item label="Normalizer"><Select value={modalForm.Normalizer} options={mockSelections.Normalizer.map(v => ({ label: v, value: v }))} onChange={v => setModalForm(f => ({ ...f, Normalizer: v }))} allowClear showSearch /></Form.Item></Col>
            </Row>
          </Form>
        </Modal>
      </Modal>
    </div>
  );
};

export default StatisticalAnalysis;