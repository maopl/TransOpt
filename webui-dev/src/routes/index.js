import { lazy } from 'react';
import {
  AppstoreOutlined,
  BarChartOutlined,
  ExperimentOutlined,
  DashboardOutlined,
  MessageOutlined,
  QuestionCircleOutlined
} from '@ant-design/icons';

const index = [
  {
    path: '/dashboard',
    name: 'Dashboard',
    icon: <DashboardOutlined />,
    component: lazy(() => import('../pages/protected/Dashboard')),
  },
  {
    path: '/experiment',
    name: 'Experiment',
    icon: <ExperimentOutlined />,
    component: lazy(() => import('../pages/protected/Experiment')),
  },
  // {
  //   path: '/analytics',
  //   name: 'Analytics',
  //   icon: <BarChartOutlined />,
  //   component: lazy(() => import('../pages/protected/Analytics')),
  // },
  // {
  //   path: '/chatopt',
  //   name: 'ChatOpt',
  //   icon: <MessageOutlined />,
  //   component: lazy(() => import('../pages/protected/ChatOpt')),
  // },
  {
    path: '/welcome',
    name: 'Welcome',
    component: lazy(() => import('../pages/protected/Welcome')),
    hideInMenu: true,
  },
];

export default index;
