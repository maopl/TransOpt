import React, { useState } from 'react';
import * as echarts from 'echarts';
import ReactECharts from 'echarts-for-react';
import RadarData from './data/RadarData.json';
import theme from './my_theme.json';

echarts.registerTheme('my_theme', theme.theme)

function Radar() {
  const option = {
    // option
    // legend: {
    //   data: ['Allocated Budget'],
    //   textStyle: {
    //     color: '#ffffff'
    //   }
    // },
    radar: {
      // shape: 'circle',
      indicator: RadarData.indicator
    },
    toolbox:{
      feature: {
          saveAsImage: {}
      }
    },
    series: [
      {
        name: 'Budget vs spending',
        type: 'radar',
        data: RadarData.data,
        emphasis: {
            lineStyle: {
                width: 4
            },
            areaStyle: {
                color: '#2EC7C980'
            }
        },  
        areaStyle: {
            color: '#2EC7C980'
        }
      }
    ]
  };
  
  return <ReactECharts
    option={option}
    style={{ height: 400 }}
    theme={'my_theme'}
  />;
};

export default Radar;