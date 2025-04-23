import React, { useState, useRef, useEffect } from 'react';
import './styles.css';
import robotIcon from '../../assets/images/robot-white.svg';
import ChatUI from "./components/ChatUI";

const ChatRobot = () => {
    const [isOpen, setIsOpen] = useState(false);
    const [messages, setMessages] = useState([
        { id: 1, text: 'Hello! I\'m the TransOpt Assistant. How can I help you today?', sender: 'bot', timestamp: new Date() }
    ]);
    const [inputMessage, setInputMessage] = useState('');
    const [isTyping, setIsTyping] = useState(false);
    const messagesEndRef = useRef(null);

    // Handle sending message
    const handleSendMessage = () => {
        if (inputMessage.trim() === '') return;
        
        // Add user message to the list
        const newUserMessage = {
            id: messages.length + 1,
            text: inputMessage,
            sender: 'user',
            timestamp: new Date()
        };
        setMessages([...messages, newUserMessage]);
        setInputMessage('');
        
        // Simulate bot typing
        setIsTyping(true);
        
        // Simulate backend API call - replace with actual API call in production
        setTimeout(() => {
            const botReply = {
                id: messages.length + 2,
                text: `This is a response to "${inputMessage}". In actual development, this would be a reply from the backend API.`,
                sender: 'bot',
                timestamp: new Date()
            };
            setMessages(prevMessages => [...prevMessages, botReply]);
            setIsTyping(false);
        }, 1000);
    };

    // Handle key event - press Enter to send message
    const handleKeyPress = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            handleSendMessage();
        }
    };

    // Auto-scroll to the latest message
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
    }, [messages]);

    // Format message time
    const formatTime = (date) => {
        return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
    };

    return (
        <div className="chat-robot-container">
            {/* Chat button */}
            {!isOpen && (
                <button 
                    className="chat-robot-button"
                    onClick={() => setIsOpen(true)}
                    aria-label="Open chat window"
                >
                    <img src={robotIcon} alt="Robot" className="robot-icon" style={{ width: '45px', height: '45px', color: 'white' }} />
                </button>
            )}

            {/* Chat window */}
            {isOpen && (
                <div className="chat-window">
                    {/* Chat window header */}
                    <div className="chat-header">
                        <h3>TransOpt Assistant</h3>
                        <button
                            className="close-button"
                            onClick={() => setIsOpen(false)}
                            aria-label="Close chat window"
                        >
                            <svg width="16" height="16" viewBox="0 0 16 16" fill="none" xmlns="http://www.w3.org/2000/svg">
                                <path d="M12.7071 3.29289C13.0976 3.68342 13.0976 4.31658 12.7071 4.70711L8.70711 8.70711C8.31658 9.09763 7.68342 9.09763 7.29289 8.70711L3.29289 4.70711C2.90237 4.31658 2.90237 3.68342 3.29289 3.29289C3.68342 2.90237 4.31658 2.90237 4.70711 3.29289L8 6.58579L11.2929 3.29289C11.6834 2.90237 12.3166 2.90237 12.7071 3.29289Z" fill="currentColor"/>
                        </svg>
                    </button>
                    </div>
                    <ChatUI />
                    {/*/!* Chat message area *!/*/}
                    {/*<div className="chat-messages">*/}
                    {/*    {messages.map((message) => (*/}
                    {/*        <div*/}
                    {/*            key={message.id}*/}
                    {/*            className={`message ${message.sender === 'user' ? 'user-message' : 'bot-message'}`}*/}
                    {/*        >*/}
                    {/*            <div className="message-bubble">*/}
                    {/*                <p>{message.text}</p>*/}
                    {/*                <span className="message-time">{formatTime(message.timestamp)}</span>*/}
                    {/*            </div>*/}
                    {/*        </div>*/}
                    {/*    ))}*/}
                    {/*    {isTyping && (*/}
                    {/*        <div className="message bot-message">*/}
                    {/*            <div className="message-bubble typing">*/}
                    {/*                <span className="typing-dot"></span>*/}
                    {/*                <span className="typing-dot"></span>*/}
                    {/*                <span className="typing-dot"></span>*/}
                    {/*            </div>*/}
                    {/*        </div>*/}
                    {/*    )}*/}
                    {/*    <div ref={messagesEndRef} />*/}
                    {/*</div>*/}

                    {/*/!* Chat input area *!/*/}
                    {/*<div className="chat-input-area">*/}
                    {/*    <textarea*/}
                    {/*        className="chat-input"*/}
                    {/*        value={inputMessage}*/}
                    {/*        onChange={(e) => setInputMessage(e.target.value)}*/}
                    {/*        onKeyPress={handleKeyPress}*/}
                    {/*        placeholder="Type your question..."*/}
                    {/*        rows={1}*/}
                    {/*    />*/}
                    {/*    <button*/}
                    {/*        className="send-button"*/}
                    {/*        onClick={handleSendMessage}*/}
                    {/*        disabled={inputMessage.trim() === ''}*/}
                    {/*        aria-label="Send message"*/}
                    {/*    >*/}
                    {/*        <svg width="20" height="20" viewBox="0 0 20 20" fill="none" xmlns="http://www.w3.org/2000/svg">*/}
                    {/*            <path d="M2.5 10L2.5 8.5L15 2L10.5 10L15 18L2.5 11.5V10Z" fill="currentColor"/>*/}
                    {/*        </svg>*/}
                    {/*    </button>*/}
                    {/*</div>*/}
                </div>
            )}
        </div>
    );
};

export default ChatRobot;