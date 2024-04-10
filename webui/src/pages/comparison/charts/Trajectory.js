import React, { useState } from 'react';
import * as echarts from 'echarts';
import ReactECharts from 'echarts-for-react';
import data from './data/TrajectoryData.json';
import my_theme from './my_theme.json';

echarts.registerTheme('my_theme', my_theme.theme)

function Trajectory() {
  var base = -data.reduce(function (min, val) {
      return Math.floor(Math.min(min, val.l));
  }, Infinity);

  const DEFAULT_OPTION = {
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
      formatter: function (params) {
        return (
          params[2].name +
          '<br />' +
          ((params[2].value - base) * 100).toFixed(1) +
          '%'
        );
      }
    },
    grid: {
      left: '3%',
      right: '4%',
      bottom: '3%',
      containLabel: true
    },
    legend: {
      data: ["BO"],
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
      type: 'category',
      data: data.map(function (item) {
        return item.FEs;
      }),
      axisLabel: {
        formatter: function (value, idx) {
          var date = new Date(value);
          return idx === 0
            ? value
            : [date.getMonth() + 1, date.getDate()].join('-');
        }
      },
      axisLine: {
        lineStyle: {
          color: 'white'
        }
      },
      boundaryGap: false
    },
    yAxis: {
      axisLabel: {
        formatter: function (val) {
          return (val - base) * 100 + '%';
        }
      },
      axisPointer: {
        label: {
          formatter: function (params) {
            return ((params.value - base) * 100).toFixed(1) + '%';
          }
        }
      },
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
        data: data.map(function (item) {
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
        data: data.map(function (item) {
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
        name: "BO",
        type: 'line',
        data: data.map(function (item) {
          return item.value + base;
        }),
        itemStyle: {
          color: '#2EC7C9'
        },
        showSymbol: false
      }
    ]
  };

  const [option, setOption] = useState(DEFAULT_OPTION);
  
  return <ReactECharts
    option={option}
    style={{ height: 400 }}
    theme={"my_theme"}
  />;
};

export default Trajectory;