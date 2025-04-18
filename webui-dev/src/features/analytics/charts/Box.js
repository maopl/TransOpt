import React from 'react';
import * as echarts from 'echarts';
import ReactECharts from 'echarts-for-react';
import defaultBoxData from './data/BoxData.json';
import my_theme from './my_theme.json';
import {Line} from "react-chartjs-2";

echarts.registerTheme('my_theme', my_theme);

const groupLabels = ['A', 'B', 'C']; // 可根据数据动态生成

function Box({ BoxData }) {
  // 只接受二维数值数组
  const dataArr = (Array.isArray(BoxData) && BoxData.length > 0 && BoxData.every(arr => Array.isArray(arr) && arr.length > 0 && arr.every(x => typeof x === 'number')))
    ? BoxData
    : defaultBoxData;

  // 检查数据有效性
  const valid = Array.isArray(dataArr) && dataArr.length > 0 && dataArr.every(arr => Array.isArray(arr) && arr.length > 0 && arr.every(x => typeof x === 'number'));

  if (!valid) {
    return <div style={{height: 300, display: 'flex', alignItems: 'center', justifyContent: 'center'}}>暂无数据</div>;
  }

  const option = {
    dataset: [
      { source: dataArr },
      {
        transform: {
          type: 'boxplot',
          config: {
            itemNameFormatter: function (value) {
              return groupLabels[value.value] || value.value;
            },
          },
        },
      },
      { fromDatasetIndex: 1, fromTransformResult: 1 }
    ],
    tooltip: { trigger: 'item', axisPointer: { type: 'shadow' } },
    toolbox: { feature: { saveAsImage: {} } },
    grid: { left: '5%', right: '5%', bottom: '10%', top: '10%' },
    xAxis: {
      type: 'category',
      nameGap: 30,
      data: groupLabels,
      axisLabel: { color: '#333' },
      lineStyle: { color: '#ccc' }
    },
    yAxis: {
      type: 'value',
      name: 'value',
      lineStyle: { color: '#ccc' },
      axisLabel: { color: '#333' },
      min: 'dataMin',
      max: 'dataMax'
    },
    series: [
      { name: 'boxplot', type: 'boxplot', datasetIndex: 1, itemStyle: { color: '#2EC7C9' } },
      { name: 'outlier', type: 'scatter', datasetIndex: 2, symbol: 'circle' }
    ]
  };

  return (

      <div style={{ width: '100%', height: 320, background: 'transparent' }}>
        <div style={{ fontWeight: 600, fontSize: 20, marginBottom: 14, marginLeft: 8 }}>Box</div>
        <ReactECharts
            option={option}
            style={{ height: '320px', width: '100%' }}
            theme="my_theme"
            notMerge={true}
            lazyUpdate={true}
        />
      </div>
  );
}

export default Box;