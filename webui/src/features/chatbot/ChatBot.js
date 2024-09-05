
import React from 'react';
import { Row, Col } from 'reactstrap';
import TitleCard from "../../components/Cards/TitleCard"
import ChatUI from './components/ChatUI'

class ChatBot extends React.Component {

    render() {
        return (
            <div >
                <Row>
                <Col lg={12} sm={12}>
                <TitleCard
                  title={
                    <h5>
                      ChatOpt
                    </h5>
                  }
                >
                  <div>
                    <ChatUI />
                  </div>
                </TitleCard>
                </Col>
                <Col lg={12} sm={12}>
                </Col>
              </Row>
            </div>
        );
    }
}


export default ChatBot