import { Suspense, useEffect } from "react"
import { useSelector, useDispatch } from 'react-redux'
import { removeNotificationMessage } from "../features/common/headerSlice"
import { NotificationContainer, NotificationManager } from 'react-notifications';
import 'react-notifications/lib/notifications.css';
import { ProLayout } from '@ant-design/pro-components';
import { useNavigate, useLocation, Routes, Route, Navigate, NavLink } from 'react-router-dom';
import routes from '../routes';
import ChatRobot from '../features/chatRobot';

function flattenRoutes(routes) {
  let result = [];
  routes.forEach(route => {
    result.push(route);
    if (route.children) {
      result = result.concat(flattenRoutes(route.children));
    }
  });
  return result;
}

const Logo = (
    <div style={{ display: 'flex', alignItems: 'center' }}>
        <img src="/transopt.png" alt="TransOpt Logo" style={{ height: '54px', width: 'auto', marginRight: '5px' }} />
    </div>
)

const menuItemRender = (item, dom) => (
    <NavLink
        to={item.path}
        className={({ isActive }) => isActive ? 'ant-menu-item-selected' : ''}
        style={({ isActive }) => ({
            width: '100%',
            display: 'block',
            backgroundColor: isActive ? '#f4f4f4' : 'transparent',
            color: isActive ? '#464646' : 'inherit',
            padding: '0 16px',
            borderRadius: '4px',
            fontWeight: isActive ? 'bold' : 'normal',
            transition: 'all 0.3s'
        })}
    >
        {dom}
    </NavLink>
)

function Layout() {
  const dispatch = useDispatch();
  const { newNotificationMessage, newNotificationStatus } = useSelector(state => state.header);
  const navigate = useNavigate();
  const location = useLocation();

  useEffect(() => {
    if (newNotificationMessage !== "") {
      if (newNotificationStatus === 1) NotificationManager.success(newNotificationMessage, 'Success');
      if (newNotificationStatus === 0) NotificationManager.error(newNotificationMessage, 'Error');
      dispatch(removeNotificationMessage());
    }
  }, [newNotificationMessage]);

  // 处理路径前缀的辅助函数
  const addAppPrefix = (path) => '/app' + (path.startsWith('/') ? path : `/${path}`);

  // 菜单数据 path 补全 '/app' 前缀 (用于菜单项跳转)
  const menuData = routes
    .filter(item => !item.hideInMenu)
    .map(item => ({ 
      ...item, 
      path: addAppPrefix(item.path)
    }));
  
  // 获取扁平化的路由
  const flat = flattenRoutes(routes);


  return (
    <>
      <NotificationContainer />
      <ProLayout
        layout="top"
        title={""}
        logo={Logo}
        route={{ routes: menuData }}
        menuDataRender={() => menuData}
        menuItemRender={menuItemRender}
        onMenuHeaderClick={() => navigate('/app/welcome')}
        location={{ pathname: location.pathname }}
      >
          {/*悬浮的聊天机器人*/}
          <ChatRobot />

            <Suspense fallback={<div>Loading...</div>}>
            <Routes>
            {flat.map((route, idx) => route.component && (
              <Route
                key={route.path}
                path={route.path.replace(/^\//, '')} // 移除开头的斜杠，因为已经在App.jsx中定义了'/app/*'
                element={<route.component />}
              />
            ))}
                <Route path="*" element={<Navigate to="welcome" replace />} />
            </Routes>
            </Suspense>
      </ProLayout>
    </>
  );
}

export default Layout