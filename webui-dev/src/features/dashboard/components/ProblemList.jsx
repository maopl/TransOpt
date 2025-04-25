import React, {useEffect, useState} from "react";
import {Card, Tree, Space, Typography} from "antd";
import {DatabaseOutlined, ExperimentOutlined, FileOutlined, DeleteOutlined} from '@ant-design/icons';
import {Popconfirm} from "antd";

const {Text} = Typography;

// 统一的卡片样式
const cardStyle = {
    borderRadius: "8px",
    boxShadow: "0 1px 2px -2px rgba(0, 0, 0, 0.16), 0 3px 6px 0 rgba(0, 0, 0, 0.12), 0 5px 12px 4px rgba(0, 0, 0, 0.09)"
};

/**
 * 根据问题查找
 * @param filteredExperiments
 * @param currentProblem
 */
const getNodeKey = (filteredExperiments, currentProblem) => {
    if (!currentProblem || !filteredExperiments) return null

    for (let experimentIndex = 0; experimentIndex < filteredExperiments.length; experimentIndex++) {
        const problemList = filteredExperiments[experimentIndex].filteredProblems;
        const taskIndex = problemList.findIndex(
            problem => problem.problem_name === currentProblem.problem_name
        );

        if (taskIndex !== -1) {
            return `${experimentIndex}-${taskIndex}`;
        }
    }
    return null // 没找到时返回
};

const ProblemList = ({
                         filteredExperiments,
                         onDelete,
                         setCurrentProblem,
                         currentProblem
                     }) => {

    /**
     * 点击回调
     * @param _
     * @param e
     */
    const onSelect = (_, e) => {

        const currentKey = getNodeKey(filteredExperiments, currentProblem);
        const selectedKey = e.node.key

        // 如果点击的是当前选中的节点，不做任何操作
        if (e.node.key === currentKey) {
            return
        }
        const [experimentIndex, taskIndex] = selectedKey.split('-').map(Number);

        // 如果存在taskIndex那么点击的是问题, 否则点击的是实验, 就选择实验下的第一个问题
        const selectedProblem = filteredExperiments[experimentIndex]?.filteredProblems[taskIndex ?? 0];
        setCurrentProblem(selectedProblem);
    };

    /**
     * 选中节点变更后, 副作用
     */
    useEffect(() => {

    }, [currentProblem]);

    // 自定义树节点渲染函数
    const titleRender = (nodeData) => {

        const isExperiment = nodeData.key.indexOf('-') === -1;
        const [currentExperimentIndex, currentTaskIndex] = getNodeKey(filteredExperiments, currentProblem)?.split('-').map(Number) ?? [];

        // 处理实验节点
        if (isExperiment) {
            const experimentIndex = parseInt(nodeData.key);
            const isSelected = currentExperimentIndex === experimentIndex;

            return (
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    width: '100%',
                    color: isSelected ? '#1890ff' : 'rgba(0, 0, 0, 0.85)',
                    fontWeight: isSelected ? '500' : 'normal'
                }}>
                    <ExperimentOutlined style={{marginRight: '8px', color: isSelected ? '#1890ff' : '#666'}}/>
                    <span>{nodeData.title}</span>
                </div>
            );
        }
        // 处理问题节点
        else {
            const [experimentIndex, taskIndex] = nodeData.key.split('-').map(Number);
            const isSelected = currentExperimentIndex === experimentIndex && currentTaskIndex === taskIndex;

            return (
                <div style={{
                    display: 'flex',
                    alignItems: 'center',
                    justifyContent: 'space-between',
                    width: '300px',
                    color: isSelected ? '#1890ff' : 'rgba(0, 0, 0, 0.65)',
                    fontWeight: isSelected ? '500' : 'normal'
                }}>
                    <div style={{display: 'flex', alignItems: 'center'}}>
                        <FileOutlined style={{marginRight: '8px', color: isSelected ? '#1890ff' : '#999'}}/>
                        <span>{nodeData.title}</span>
                    </div>
                    <Popconfirm
                        title="Delete this task"
                        description="Are you sure you want to delete this task?"
                        onConfirm={e => {
                            e.stopPropagation();
                            const task = filteredExperiments[experimentIndex].filteredProblems[taskIndex];
                            onDelete(task.problem_name);
                        }}
                        onCancel={e => e.stopPropagation()}
                        okText="Yes"
                        cancelText="No"
                        placement="right"
                    >
                        <DeleteOutlined
                            style={{color: '#ff4d4f', fontSize: '14px'}}
                            onClick={(e) => e.stopPropagation?.()}
                        />
                    </Popconfirm>
                </div>
            );
        }
    };

    return (
        <div style={{
            minWidth: "420px",
            width: "420px",
            overflowY: "auto",
            height: "100%",
        }}>
            <Card
                style={{
                    ...cardStyle
                }}
                styles={{
                    overflow: "hidden",
                }}
            >
                {/* 列表头部 - 显示结果数量 */}
                <div style={{
                    display: "flex",
                    justifyContent: "space-between",
                    alignItems: "center",
                    marginBottom: "16px",
                    borderBottom: "1px solid #f0f0f0",
                    paddingBottom: "12px"
                }}>
                    <Space>
                        <DatabaseOutlined style={{color: "#1890ff"}}/>
                        <Text strong>{filteredExperiments.length} Results</Text>
                    </Space>
                </div>

                {/* 可滚动的树形列表容器 */}
                <div style={{
                    paddingRight: "10px",
                    marginBottom: "10px",
                    minHeight: "500px"
                }}>
                    <Tree
                        treeData={filteredExperiments.map((experiment, index) => ({
                            title: experiment.experimentName,
                            key: index.toString(),
                            icon: <ExperimentOutlined/>,
                            children: experiment.filteredProblems.map((task, taskIndex) => ({
                                title: task.displayName,
                                key: `${index}-${taskIndex}`,
                                icon: <FileOutlined/>,
                                isLeaf: true,
                            })),
                        }))}
                        onSelect={onSelect}
                        titleRender={titleRender}
                        showIcon={false}
                    />
                </div>
            </Card>
        </div>
    );
};

export default ProblemList;