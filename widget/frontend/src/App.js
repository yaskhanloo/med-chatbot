import React, { useState, useRef, useEffect } from 'react';
import { chatAPI } from './services/api';
import './App.css';

function App() {
  const [messages, setMessages] = useState([]);
  const [input, setInput] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [isBackendReady, setIsBackendReady] = useState(false);
  const messagesEndRef = useRef(null);

  // Check backend health on mount
  useEffect(() => {
    checkBackendHealth();
  }, []);

  // Auto-scroll to bottom when new messages arrive
  useEffect(() => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  }, [messages]);

  const checkBackendHealth = async () => {
    try {
      const health = await chatAPI.checkHealth();
      setIsBackendReady(health.status === 'healthy');
    } catch (error) {
      setIsBackendReady(false);
      console.error('Backend not ready:', error);
    }
  };

  const handleSendMessage = async (e) => {
    e.preventDefault();
    
    if (!input.trim() || isLoading) return;

    const userMessage = input.trim();
    setInput('');

    // Add user message to chat
    setMessages((prev) => [...prev, { role: 'user', content: userMessage }]);
    setIsLoading(true);

    try {
      // Send to backend
      const response = await chatAPI.sendMessage(userMessage);
      
      // Add bot response to chat
      setMessages((prev) => [
        ...prev,
        { role: 'assistant', content: response.response },
      ]);
    } catch (error) {
      console.error('Error:', error);
      setMessages((prev) => [
        ...prev,
        { 
          role: 'assistant', 
          content: 'Please try again.' 
        },
      ]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleClearChat = () => {
    if (window.confirm('Clear all?')) {
      setMessages([]);
      chatAPI.clearConversation();
    }
  };

  return (
    <div className="App">
      <div className="chat-container">
        {/* Header */}
        <div className="chat-header">
          <div className="header-content">
            <h1>medGEMMA chatbot</h1>
            <div className="header-status">
              <span className={`status-dot ${isBackendReady ? 'online' : 'offline'}`}></span>
              <span>{isBackendReady ? 'Online' : 'Offline'}</span>
            </div>
          </div>
          <button onClick={handleClearChat} className="clear-btn">
            Clear Chat
          </button>
        </div>

        {/* Messages */}
        <div className="messages-container">
          {messages.length === 0 ? (
            <div className="welcome-message">
              <h2>Welcome!</h2>
              <p>How can I help you today?</p>
            </div>
          ) : (
            messages.map((msg, index) => (
              <div key={index} className={`message ${msg.role}`}>
                <div className="message-content">
                  {msg.content}
                </div>
              </div>
            ))
          )}
          {isLoading && (
            <div className="message assistant">
              <div className="message-content loading">
                <span>Thinking</span>
                <div className="loading-dots">
                  <span></span>
                  <span></span>
                  <span></span>
                </div>
              </div>
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* Input */}
        <form onSubmit={handleSendMessage} className="input-container">
          <input
            type="text"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            placeholder="Ask me anything..."
            disabled={isLoading || !isBackendReady}
            className="message-input"
          />
          <button 
            type="submit" 
            disabled={isLoading || !input.trim() || !isBackendReady}
            className="send-btn"
          >
            Send
          </button>
        </form>
      </div>
    </div>
  );
}

export default App;
