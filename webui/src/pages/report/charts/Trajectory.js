import React from 'react';
import * as echarts from 'echarts';
import ReactECharts from 'echarts-for-react';
import my_theme from './my_theme.json';

echarts.registerTheme('my_theme', my_theme.theme)

function Trajectory({TrajectoryData}) {
  // console.log("data:", TrajectoryData)
  var base = -TrajectoryData.reduce(function (min, val) {
      return Math.floor(Math.min(min, val.l));
  }, Infinity);

  const option = {
    tooltip: {
      trigger: 'axis',
      axisPointer: {
        type: 'cross',
        animation: false,
        label: {
          backgroundColor: '#ccc',
          borderColor: '#aaa',
          borderWidth: 1,
          shadowBlur: 0,
          shadowOffsetX: 0,
          shadowOffsetY: 0,
          color: '#222'
        }
      },
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    toolbox:{
      feature: {
          saveAsImage: {}
      }
    },
    xAxis: {
      type: 'category',
      data: TrajectoryData.map(function (item) {
        return item.FEs;
      }),
      axisLabel: {
      },
      axisLine: {
        lineStyle: {
          color: 'white'
        }
      },
      boundaryGap: false
    },
    yAxis: {

      axisLine: {
        lineStyle: {
          color: 'white'
        }
      },
      splitNumber: 3
    },
    series: [
      {
        name: 'L',
        type: 'line',
        data: TrajectoryData.map(function (item) {
          return item.l + base;
        }),
        lineStyle: {
          opacity: 0
        },
        stack: 'confidence-band',
        symbol: 'none'
      },
      {
        name: 'U',
        type: 'line',
        data: TrajectoryData.map(function (item) {
          return item.u - item.l;
        }),
        lineStyle: {
          opacity: 0
        },
        areaStyle: {
          color: '#2EC7C980'
        },
        stack: 'confidence-band',
        symbol: 'none'
      },
      {
        type: 'line',
        data: TrajectoryData.map(function (item) {
          return item.value + base;
        }),
        itemStyle: {
          color: '#2EC7C9'
        },
        showSymbol: false
      }
    ]
  };  

  // console.log("option",option)
  
  return <ReactECharts
    option={option}
    style={{ height: 400 }}
    theme={"my_theme"}
  />;
};

export default Trajectory;