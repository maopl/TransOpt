import React, { useState } from 'react';
import * as echarts from 'echarts';
import ReactECharts from 'echarts-for-react';
import ScatterData from './data/ScatterData.json';
import my_theme from './my_theme.json';
import { keys } from 'highcharts';

echarts.registerTheme('my_theme', my_theme.theme)

function Scatter() {
  const convertToseries = Object.keys(ScatterData).map(key=>({
    name: key,
    data: ScatterData[key],
    type: 'scatter',
    symbol: 'circle'
  }))  


  const option = {
    // option
    legend: {
        textStyle: {
            color: '#ffffff'
          }
    },
    toolbox:{
      feature: {
          saveAsImage: {}
      }
    },
    xAxis: {
        type:"value",
        show: false, 
        axisLabel: {
            color: '#ffffff'
        },
        lineStyle: {
            color: 'white'
        },
        splitLine: { show: false },
    },
    yAxis: {
        type: "value",
        show: false, 
        axisLabel: {
            color: '#ffffff'
        },
        lineStyle: {
            color: 'white'
        },
        splitLine: { show: false },
    },
    series: convertToseries
  };
  
  return <ReactECharts
    option={option}
    style={{ height: 400 }}
    theme={"my_theme"}
  />;
};

export default Scatter;