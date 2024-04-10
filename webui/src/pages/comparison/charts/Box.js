import React, { useState } from 'react';
import * as echarts from 'echarts';
import ReactECharts from 'echarts-for-react';
import BoxData from './data/BoxData.json';
import my_theme from './my_theme.json';

echarts.registerTheme('my_theme', my_theme.theme)

function Box() {
  const option = {
    // option
      dataset: [
        {
          // prettier-ignore
          source: BoxData
        },
        {
          transform: {
            type: 'boxplot',
            config: { itemNameFormatter: 'expr {value}' }
          }
        },
        {
          fromDatasetIndex: 1,
          fromTransformResult: 1
        }
      ],
      tooltip: {
        trigger: 'item',
        axisPointer: {
          type: 'shadow'
        }
      },
      toolbox:{
        feature: {
            saveAsImage: {}
        }
      },
      grid: {
        left: '10%',
        right: '10%',
        bottom: '15%'
      },
      xAxis: {
        type: 'category',
        boundaryGap: true,
        nameGap: 30,
        axisLabel:{
            color: '#ffffff'
        },
        lineStyle: {
            color: 'white'
        },
      },
      yAxis: {
        type: 'value',
        name: 'value',
        lineStyle: {
            color: 'white'
        },
        axisLabel:{
            color: '#ffffff'
        },
        min: 500,
      },
      series: [
        {
          name: 'boxplot',
          type: 'boxplot',
          datasetIndex: 1,
          itemStyle: {
            color: '#2EC7C9'
          },
        },
        {
          name: 'outlier',
          type: 'scatter',
          datasetIndex: 2,
          symbol: 'circle',
        }
      ]
  };
  
  return <ReactECharts
    option={option}
    style={{ height: 400 }}
    theme={"my_theme"}
  />;
};

export default Box;