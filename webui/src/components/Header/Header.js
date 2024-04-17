import React from "react";
import { withRouter } from "react-router";
import {
  Navbar,
  Nav,
  NavItem,
} from "reactstrap";

import s from "./Header.module.scss";
import "animate.css";
import LinksGroup from './LinksGroup';
import {changeActiveSidebarItem} from '../../actions/navigation';
import TypographyIcon from '../Icons/SidebarIcons/TypographyIcon';
import TablesIcon from '../Icons/SidebarIcons/TablesIcon';
import NotificationsIcon from "../Icons/SidebarIcons/NotificationsIcon";

class Header extends React.Component {
  render() {
    return (
      <div className="Header">
      <h1 className={s.logo}>
        <span className="fw-bold">Transfer Optimization System</span>
      </h1>
      <Navbar className={s.navbar}>
        <div className={`d-print-none ${s.root}`}>
          <Nav className="ml-md-0">
            <ul className={s.nav}>
              <LinksGroup
                onActiveSidebarItemChange={activeItem => this.props.dispatch(changeActiveSidebarItem(activeItem))}
                activeItem={this.props.activeItem}
                header="Configuration"
                isHeader
                iconName={<TypographyIcon className={s.menuIcon} />}
                link="/app/configuration"
                index="core"
              />
            </ul>
            <NavItem className={`${s.divider} d-none d-sm-block`} />
            <ul className={s.nav}>
              <LinksGroup
                onActiveSidebarItemChange={t => this.props.dispatch(changeActiveSidebarItem(t))}
                activeItem={this.props.activeItem}
                header="Report"
                isHeader
                iconName={<TablesIcon className={s.menuIcon} />}
                link="/app/report"
                index="tables"
              />
            </ul>
            <NavItem className={`${s.divider} d-none d-sm-block`} />
            <ul className={s.nav}>
              <LinksGroup
                onActiveSidebarItemChange={t => this.props.dispatch(changeActiveSidebarItem(t))}
                activeItem={this.props.activeItem}
                header="Comparison"
                isHeader
                iconName={<NotificationsIcon className={s.menuIcon} />}
                link="/app/comparison"
                index="tables"
              />
            </ul>
          </Nav>
        </div>
      </Navbar>
      </div>
    );
  }
}


export default withRouter(Header);
