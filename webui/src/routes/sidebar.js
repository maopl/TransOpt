/** Icons are imported separatly to reduce build time */
import Squares2X2Icon from '@heroicons/react/24/outline/Squares2X2Icon'
import ChartBarIcon from '@heroicons/react/24/outline/ChartBarIcon'
import ChatBubbleLeftIcon from '@heroicons/react/24/outline/ChatBubbleLeftIcon'
import QuestionMarkCircleIcon from '@heroicons/react/24/outline/QuestionMarkCircleIcon'
import FolderOpenIcon from '@heroicons/react/24/outline/FolderOpenIcon'
import PlayIcon from '@heroicons/react/24/outline/PlayIcon'
import CogIcon from '@heroicons/react/24/outline/CogIcon'
import AdjustmentsHorizontalIcon from '@heroicons/react/24/outline/AdjustmentsHorizontalIcon'


const iconClasses = `h-6 w-6`
const submenuIconClasses = `h-5 w-5`

const routes = [

  {
    path: '/app/dashboard',
    icon: <Squares2X2Icon className={iconClasses}/>, 
    name: 'Dashboard',
  },

  // {
  //   path: '/app/leads', // url
  //   icon: <InboxArrowDownIcon className={iconClasses}/>, // icon component
  //   name: 'Leads', // name that appear in Sidebar
  // },
  // {
  //   path: '/app/transactions', // url
  //   icon: <CurrencyDollarIcon className={iconClasses}/>, // icon component
  //   name: 'Transactions', // name that appear in Sidebar
  // },

  // {
  //   path: '/app/integration', // url
  //   icon: <BoltIcon className={iconClasses}/>, // icon component
  //   name: 'Integration', // name that appear in Sidebar
  // },
  // {
  //   path: '/app/calendar', // url
  //   icon: <CalendarDaysIcon className={iconClasses}/>, // icon component
  //   name: 'Calendar', // name that appear in Sidebar
  // },

  {
    path: '/app/optimization', //no url needed as this has submenu
    icon: <CogIcon className={`${iconClasses} inline` }/>, // icon component
    name: 'Experiments', // name that appear in Sidebar
    submenu : [
      {
        path: '/app/optimization/problem',
        icon: <QuestionMarkCircleIcon className={submenuIconClasses}/>,
        name: 'Specify Problems',
      },
      {
        path: '/app/optimization/algorithm', //url
        icon: <AdjustmentsHorizontalIcon className={submenuIconClasses}/>, // icon component
        name: 'Choose Algorithm Objects', // name that appear in Sidebar
      },
      {
        path: '/app/optimization/selectdatasets',
        icon: <FolderOpenIcon className={submenuIconClasses}/>,
        name: 'Select Datasets',
      },
      {
        path: '/app/optimization/run',
        icon: <PlayIcon className={submenuIconClasses}/>,
        name: 'Run',
      },
      // {
      //   path: '/app/404',
      //   icon: <ExclamationTriangleIcon className={submenuIconClasses}/>,
      //   name: '404',
      // },
    ]
  },

  {
    path: '/app/analytics', // url
    icon: <ChartBarIcon className={iconClasses}/>, // icon component
    name: 'Analytics', // name that appear in Sidebar
  },
  
  {
    path: '/app/chatopt',
    icon: <ChatBubbleLeftIcon className={iconClasses}/>, 
    name: 'ChatOpt',
  },

  // {
  //   path: '', //no url needed as this has submenu
  //   icon: <Cog6ToothIcon className={`${iconClasses} inline` }/>, // icon component
  //   name: 'Settings', // name that appear in Sidebar
  //   submenu : [
  //     {
  //       path: '/app/settings-profile', //url
  //       icon: <UserIcon className={submenuIconClasses}/>, // icon component
  //       name: 'Profile', // name that appear in Sidebar
  //     },
  //     {
  //       path: '/app/settings-billing',
  //       icon: <WalletIcon className={submenuIconClasses}/>,
  //       name: 'Billing',
  //     },
  //     {
  //       path: '/app/settings-team', // url
  //       icon: <UsersIcon className={submenuIconClasses}/>, // icon component
  //       name: 'Team Members', // name that appear in Sidebar
  //     },
  //   ]
  // },
  // {
  //   path: '', //no url needed as this has submenu
  //   icon: <DocumentTextIcon className={`${iconClasses} inline` }/>, // icon component
  //   name: 'Documentation', // name that appear in Sidebar
  //   submenu : [
  //     {
  //       path: '/app/getting-started', // url
  //       icon: <DocumentTextIcon className={submenuIconClasses}/>, // icon component
  //       name: 'Getting Started', // name that appear in Sidebar
  //     },
  //     {
  //       path: '/app/features',
  //       icon: <TableCellsIcon className={submenuIconClasses}/>, 
  //       name: 'Features',
  //     },
  //     {
  //       path: '/app/components',
  //       icon: <CodeBracketSquareIcon className={submenuIconClasses}/>, 
  //       name: 'Components',
  //     }
  //   ]
  // },
  
]

export default routes


