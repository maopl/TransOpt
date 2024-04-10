import React from 'react';
import cx from 'classnames';
import {withRouter} from 'react-router-dom';
import s from './Sidebar.module.scss';
import LinksGroup from './LinksGroup';

import {changeActiveSidebarItem} from '../../actions/navigation';
import TypographyIcon from '../Icons/SidebarIcons/TypographyIcon';
import TablesIcon from '../Icons/SidebarIcons/TablesIcon';

class Sidebar extends React.Component {
    render() {
        return (
            <nav
                className={cx(s.root)}
                ref={(nav) => {
                    this.element = nav;
                }}
            >
                <header className={s.logo}>
                    <span className="fw-bold">Transfer Optimization System</span>
                </header>
                <ul className={s.nav}>
                    <LinksGroup
                        onActiveSidebarItemChange={activeItem => this.props.dispatch(changeActiveSidebarItem(activeItem))}
                        activeItem={this.props.activeItem}
                        header="Typography"
                        isHeader
                        iconName={<TypographyIcon className={s.menuIcon} />}
                        link="/app/typography"
                        index="core"
                    />
                    <LinksGroup
                        onActiveSidebarItemChange={t => this.props.dispatch(changeActiveSidebarItem(t))}
                        activeItem={this.props.activeItem}
                        header="Tables Basic"
                        isHeader
                        iconName={<TablesIcon className={s.menuIcon} />}
                        link="/app/tables"
                        index="tables"
                    />
                </ul>
            </nav>
        );
    }
}

export default withRouter(Sidebar);
