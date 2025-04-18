import React, { useState } from "react";
import { Button, Modal, Form, Input, Select, Table, Space, Tag, Divider, Popconfirm } from "antd";
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

function mockTrajectoryGroup() {
  return [
    { name: 'Group 1', average: [{ FEs: 1, y: 0.5 }, { FEs: 2, y: 0.7 }], uncertainty: [{ FEs: 1, y: 0.1 }, { FEs: 2, y: 0.2 }] },
    { name: 'Group 2', average: [{ FEs: 1, y: 0.6 }, { FEs: 2, y: 0.8 }], uncertainty: [{ FEs: 1, y: 0.15 }, { FEs: 2, y: 0.25 }] }
  ];
}
function mockBoxGroups() {
  return [
    [10, 12, 14, 15, 20],
    [20, 22, 24, 25, 30],
    [15, 17, 19, 20, 25]
  ];
}

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
          {results.length > 0 && (
            <div style={{ background: '#fff', borderRadius: 8, boxShadow: '0 1px 3px #eee', padding: 32 }}>
              <div style={{ fontWeight: 'bold', marginBottom: 18, fontSize: 17 }}>Comparison Chart</div>
              <div style={{ display: 'flex', gap: 32, alignItems: 'stretch', justifyContent: 'space-between' }}>
                <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                  <LineChart TrajectoryData={(() => {
                    // 只要有一项为undefined/null/非数组/空数组/其内容不是对象数组，直接用mock
                    if (!Array.isArray(results) || results.length === 0) return [mockTrajectoryGroup()];
                    const arr = results.map(r => (r && Array.isArray(r.TrajectoryData) && r.TrajectoryData.length > 0 && r.TrajectoryData.every(g => g && typeof g === 'object' && Array.isArray(g.average) && Array.isArray(g.uncertainty))) ? r.TrajectoryData : null);
                    if (arr.some(v => !Array.isArray(v))) {
                      return [mockTrajectoryGroup()];
                    }
                    // 展平并过滤空项，且只保留有效group对象
                    return arr.flat().filter(g => g && typeof g === 'object' && Array.isArray(g.average) && Array.isArray(g.uncertainty));
                  })()} />
                </div>
                <div style={{ flex: 1, minWidth: 0, display: 'flex', flexDirection: 'column', justifyContent: 'center' }}>
                  <Box BoxData={(() => {
                    if (!Array.isArray(results) || results.length === 0) return mockBoxGroups();
                    const arr = results.map(r => (r && Array.isArray(r.BoxData) && r.BoxData.length > 0 && r.BoxData.every(x => typeof x === 'number')) ? r.BoxData : null);
                    if (arr.some(v => !Array.isArray(v))) {
                      return mockBoxGroups();
                    }
                    return arr;
                  })()} />
                </div>
              </div>
            </div>
          )}
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
            <div style={{ display: 'flex', gap: 16 }}>
              <div style={{ flex: 1 }}><Form.Item label="Task Name"><Input value={modalForm.TaskName} onChange={e => setModalForm(f => ({ ...f, TaskName: e.target.value }))} /></Form.Item></div>
              <div style={{ flex: 1 }}><Form.Item label="Objectives"><Input value={modalForm.NumObjs} onChange={e => setModalForm(f => ({ ...f, NumObjs: e.target.value }))} /></Form.Item></div>
            </div>
            <div style={{ display: 'flex', gap: 16 }}>
              <div style={{ flex: 1 }}><Form.Item label="Variables"><Input value={modalForm.NumVars} onChange={e => setModalForm(f => ({ ...f, NumVars: e.target.value }))} /></Form.Item></div>
              <div style={{ flex: 1 }}><Form.Item label="Fidelity"><Input value={modalForm.Fidelity} onChange={e => setModalForm(f => ({ ...f, Fidelity: e.target.value }))} /></Form.Item></div>
            </div>
            <div style={{ display: 'flex', gap: 16 }}>
              <div style={{ flex: 1 }}><Form.Item label="Workload"><Input value={modalForm.Workload} onChange={e => setModalForm(f => ({ ...f, Workload: e.target.value }))} /></Form.Item></div>
              <div style={{ flex: 1 }}><Form.Item label="Seed"><Input value={modalForm.Seed} onChange={e => setModalForm(f => ({ ...f, Seed: e.target.value }))} /></Form.Item></div>
            </div>
            <div style={{ display: 'flex', gap: 16 }}>
              <div style={{ flex: 1 }}><Form.Item label="Refiner"><Select value={modalForm.Refiner} options={mockSelections.Refiner.map(v => ({ label: v, value: v }))} onChange={v => setModalForm(f => ({ ...f, Refiner: v }))} allowClear showSearch /></Form.Item></div>
              <div style={{ flex: 1 }}><Form.Item label="Sampler"><Select value={modalForm.Sampler} options={mockSelections.Sampler.map(v => ({ label: v, value: v }))} onChange={v => setModalForm(f => ({ ...f, Sampler: v }))} allowClear showSearch /></Form.Item></div>
            </div>
            <div style={{ display: 'flex', gap: 16 }}>
              <div style={{ flex: 1 }}><Form.Item label="Pretrain"><Select value={modalForm.Pretrain} options={mockSelections.Pretrain.map(v => ({ label: v, value: v }))} onChange={v => setModalForm(f => ({ ...f, Pretrain: v }))} allowClear showSearch /></Form.Item></div>
              <div style={{ flex: 1 }}><Form.Item label="Model"><Select value={modalForm.Model} options={mockSelections.Model.map(v => ({ label: v, value: v }))} onChange={v => setModalForm(f => ({ ...f, Model: v }))} allowClear showSearch /></Form.Item></div>
            </div>
            <div style={{ display: 'flex', gap: 16 }}>
              <div style={{ flex: 1 }}><Form.Item label="Acquisition Function"><Select value={modalForm.ACF} options={mockSelections.ACF.map(v => ({ label: v, value: v }))} onChange={v => setModalForm(f => ({ ...f, ACF: v }))} allowClear showSearch /></Form.Item></div>
              <div style={{ flex: 1 }}><Form.Item label="Normalizer"><Select value={modalForm.Normalizer} options={mockSelections.Normalizer.map(v => ({ label: v, value: v }))} onChange={v => setModalForm(f => ({ ...f, Normalizer: v }))} allowClear showSearch /></Form.Item></div>
            </div>
          </Form>
        </Modal>
      </Modal>
    </div>
  );
};

export default StatisticalAnalysis;