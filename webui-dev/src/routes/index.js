// All components mapping with path for internal routes

import { lazy } from 'react'

const Dashboard = lazy(() => import('../pages/protected/Dashboard'))
const Welcome = lazy(() => import('../pages/protected/Welcome'))
const ChatOpt = lazy(() => import('../pages/protected/ChatOpt'))
const Experiment = lazy(() => import('../pages/protected/Experiment'))
const Run = lazy(() => import('../pages/protected/Run'))
const Selectdatasets = lazy(() => import('../pages/protected/Seldata'))
const Analytics = lazy(() => import('../pages/protected/Analytics'))

const routes = [
  {
    path: '/dashboard', // the url
    component: Dashboard, // view rendered
  },
  {
    path: '/welcome', // the url
    component: Welcome, // view rendered
  },

  {
    path: '/chatopt', // the url
    component: ChatOpt, // view rendered
  },

  {
    path: '/optimization/problem', // the url
    component: Experiment, // view rendered
  },


  {
    path: '/optimization/selectdatasets', // the url
    component: Selectdatasets, // view rendered
  },

  {
    path: '/optimization/run', // the url
    component: Run, // view rendered
  },

  {
    path: '/analytics', // the url
    component: Analytics, // view rendered
  },
]

export default routes
