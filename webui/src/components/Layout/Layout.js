import React from 'react';
import PropTypes from 'prop-types';
import { Switch, Route, withRouter, Redirect } from 'react-router';
import { TransitionGroup, CSSTransition } from 'react-transition-group';

import Configuration from '../../pages/configuration/Configuration';
import Report from '../../pages/report/Report';
import Comparison from '../../pages/comparison/Comparison';

import Header from '../Header';
import s from './Layout.module.scss';

class Layout extends React.Component {
  static propTypes = {
    sidebarStatic: PropTypes.bool,
    sidebarOpened: PropTypes.bool,
    dispatch: PropTypes.func.isRequired,
  };

  static defaultProps = {
    sidebarStatic: false,
    sidebarOpened: false,
  };
  constructor(props) {
    super(props);
  }


  render() {
    return (
      <div
        className={[
          s.root
        ].join(' ')}
      >
        <div className={s.wrap}>
          <Header />
            <main className={s.content}>
              <TransitionGroup>
                <CSSTransition
                  key={this.props.location.key}
                  classNames="fade"
                  timeout={200}
                >
                  <Switch>
                    <Route path="/app/main" exact render={() => <Redirect to="/app/configuration" />} />
                    <Route path="/app/configuration" exact component={Configuration} />
                    <Route path="/app/report" exact component={Report} />
                    <Route path="/app/comparison" exact component={Comparison} />
                  </Switch>
                </CSSTransition>
              </TransitionGroup>
            </main>
        </div>
      </div>
    );
  }
}

export default withRouter(Layout);
