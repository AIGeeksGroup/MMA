import React, { useState, useRef, useEffect } from 'react';
import ChatBubble from './ChatBubble';
import MessageInput from './MessageInput';
import ApiKeyModal from './ApiKeyModal';
import ClearChatModal from './ClearChatModal';
import queuedFetch from '../utils/requestQueue';
import './ChatWindow.css';

const ChatWindow = ({ settings, messages, setMessages }) => {
  const [includeScreenshots, setIncludeScreenshots] = useState(true);
  const [showApiKeyModal, setShowApiKeyModal] = useState(false);
  const [missingApiKeys, setMissingApiKeys] = useState([]);
  const [currentModelType, setCurrentModelType] = useState('');
  // Track active streaming requests
  const [activeStreamingRequests, setActiveStreamingRequests] = useState(new Map());
  // Clear chat modal state
  const [showClearModal, setShowClearModal] = useState(false);
  const [isClearing, setIsClearing] = useState(false);
  const messagesEndRef = useRef(null);
  const abortControllersRef = useRef(new Map());

  // Calculate derived values from state early
  const hasActiveStreaming = activeStreamingRequests.size > 0;
  const currentStreamingContent = hasActiveStreaming 
    ? Array.from(activeStreamingRequests.values())[activeStreamingRequests.size - 1].streamingContent
    : '';

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, currentStreamingContent]);

  useEffect(() => {
    return () => {
      // Cleanup all abort controllers on unmount
      abortControllersRef.current.forEach((controller) => {
        controller.abort();
      });
      abortControllersRef.current.clear();
    };
  }, []);

  // Load initial screenshot setting
  useEffect(() => {
    const loadScreenshotSetting = async () => {
      try {
        const response = await queuedFetch(`${settings.serverUrl}/screenshot_setting`);
        if (response.ok) {
          const data = await response.json();
          setIncludeScreenshots(data.include_recent_screenshots);
        }
      } catch (error) {
        console.error('Error loading screenshot setting:', error);
      }
    };
    
    loadScreenshotSetting();
  }, [settings.serverUrl]);

  const sendMessage = async (messageText, imageFiles = []) => {
    if (!messageText.trim() && imageFiles.length === 0) return;

    const sanitizedImages = imageFiles.map(file => ({
      name: file.name,
      path: file.path,
      type: file.type,
      size: file.size,
      isScreenshot: file.isScreenshot || false,
      ...(file.lastModified && { lastModified: file.lastModified })
    }));

    const userMessage = {
      id: Date.now(),
      type: 'user',
      content: messageText,
      images: sanitizedImages,
      timestamp: new Date().toISOString()
    };

    setMessages(prev => [...prev, userMessage]);
    
    // Generate unique request ID for this specific message
    const requestId = `req_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
    
    // Create abort controller for this request
    const abortController = new AbortController();
    abortControllersRef.current.set(requestId, abortController);
    
    // Add to active streaming requests
    setActiveStreamingRequests(prev => new Map([...prev, [requestId, { streamingContent: '' }]]));

    let cleanup = null;
    try {
      let imageUris = null;
      if (imageFiles.length > 0) {
        imageUris = imageFiles.map(file => {
          if (file.isScreenshot) {
            return file.path;
          } else if (file.path) {
            return file.path;
          } else {
            return file.name;
          }
        });
      }

      const requestData = {
        message: messageText || null,
        image_uris: imageUris,
        memorizing: false
      };

      const result = await queuedFetch(`${settings.serverUrl}/send_streaming_message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
        signal: abortController.signal,
        isStreaming: true
      });

      const response = result.response;
      cleanup = result.cleanup;

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'intermediate') {
                // Update streaming content for this specific request
                setActiveStreamingRequests(prev => {
                  const updated = new Map(prev);
                  const current = updated.get(requestId);
                  if (current) {
                    let newContent = current.streamingContent;
                    if (data.message_type === 'internal_monologue') {
                      newContent += '\n[Thinking] ' + data.content;
                    } else if (data.message_type === 'response') {
                      newContent += '\n' + data.content;
                    }
                    updated.set(requestId, { ...current, streamingContent: newContent });
                  }
                  return updated;
                });
              } else if (data.type === 'missing_api_keys') {
                // Handle missing API keys by showing the modal
                setMissingApiKeys(data.missing_keys);
                setCurrentModelType(data.model_type);
                setShowApiKeyModal(true);
                return; // Don't continue processing
              } else if (data.type === 'final') {
                const assistantMessage = {
                  id: Date.now() + 1,
                  type: 'assistant',
                  content: data.response,
                  timestamp: new Date().toISOString()
                };
                setMessages(prev => [...prev, assistantMessage]);
                break;
              } else if (data.type === 'error') {
                throw new Error(data.error);
              }
            } catch (parseError) {
              console.error('Error parsing SSE data:', parseError);
            }
          }
        }
      }
    } catch (error) {
      if (error.name === 'AbortError') {
        console.log('Request was aborted');
      } else {
        console.error('Error sending message:', error);
        const errorMessage = {
          id: Date.now() + 1,
          type: 'error',
          content: `Error: ${error.message}`,
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, errorMessage]);
      }
    } finally {
      // Clean up this request
      setActiveStreamingRequests(prev => {
        const updated = new Map(prev);
        updated.delete(requestId);
        return updated;
      });
      
      abortControllersRef.current.delete(requestId);
      
      // Call cleanup to notify request queue
      if (cleanup) {
        cleanup();
      }
    }
  };

  const clearChatLocal = () => {
    setMessages([]);
    // Abort all active requests
    abortControllersRef.current.forEach((controller) => {
      controller.abort();
    });
    abortControllersRef.current.clear();
    setActiveStreamingRequests(new Map());
    setShowClearModal(false);
  };

  const clearChatPermanent = async () => {
    setIsClearing(true);
    
    try {
      const response = await queuedFetch(`${settings.serverUrl}/conversation/clear`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        throw new Error(`Failed to clear conversation: ${response.status}`);
      }

      const result = await response.json();
      
      // Clear local messages too
      setMessages([]);
      
      // Abort all active requests
      abortControllersRef.current.forEach((controller) => {
        controller.abort();
      });
      abortControllersRef.current.clear();
      setActiveStreamingRequests(new Map());

      // Show success message briefly
      const successMessage = {
        id: Date.now(),
        type: 'assistant',
        content: `✅ ${result.message}`,
        timestamp: new Date().toISOString()
      };
      setMessages([successMessage]);

      setShowClearModal(false);
    } catch (error) {
      console.error('Error clearing conversation:', error);
      
      // Show error message
      const errorMessage = {
        id: Date.now(),
        type: 'error',
        content: `❌ Failed to clear conversation history: ${error.message}`,
        timestamp: new Date().toISOString()
      };
      setMessages(prev => [...prev, errorMessage]);
    } finally {
      setIsClearing(false);
    }
  };

  const handleClearClick = () => {
    setShowClearModal(true);
  };

  const stopGeneration = () => {
    // Abort all active requests
    abortControllersRef.current.forEach((controller) => {
      controller.abort();
    });
  };

  const toggleScreenshotSetting = async () => {
    try {
      const newSetting = !includeScreenshots;
      const response = await queuedFetch(`${settings.serverUrl}/screenshot_setting/set`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          include_recent_screenshots: newSetting
        })
      });

      if (response.ok) {
        const data = await response.json();
        if (data.success) {
          setIncludeScreenshots(newSetting);
        } else {
          console.error('Error setting screenshot setting:', data.message);
        }
      } else {
        console.error('HTTP error setting screenshot setting:', response.status);
      }
    } catch (error) {
      console.error('Error toggling screenshot setting:', error);
    }
  };

  const handleApiKeySubmit = () => {
    // After API keys are submitted, the modal will close
    setShowApiKeyModal(false);
    setMissingApiKeys([]);
    setCurrentModelType('');
  };

  const closeApiKeyModal = () => {
    setShowApiKeyModal(false);
    setMissingApiKeys([]);
    setCurrentModelType('');
  };

  return (
    <div className="chat-window">
      <div className="chat-header">
        <div className="chat-info">
          <span className="model-info">Model: {settings.model}</span>
          <span className="persona-info">Persona: {settings.persona}</span>
        </div>
        <div className="chat-actions">
          <button 
            className={`screenshot-toggle ${includeScreenshots ? 'enabled' : 'disabled'}`}
            onClick={toggleScreenshotSetting}
            title={includeScreenshots ? "Allow assistant to see your recent screenshots" : "Assistant cannot see your recent screenshots"}
          >
            📷 {includeScreenshots ? 'ON' : 'OFF'}
          </button>
          {hasActiveStreaming && (
            <button 
              className="stop-button"
              onClick={stopGeneration}
              title="Stop generation"
            >
              ⏹️ Stop
            </button>
          )}
                      <button 
              className="clear-button"
              onClick={handleClearClick}
              title="Clear chat"
            >
              🗑️ Clear
            </button>
          </div>
        </div>

        <div className="messages-container">
          {messages.length === 0 && (
            <div className="welcome-message">
              <h2>Welcome to MIRIX!</h2>
              <p>Start a conversation with your AI assistant.</p>
              {window.electronAPI ? (
                <p>💡 MIRIX is running in the desktop app environment.</p>
              ) : (
                <p>💡 Download the desktop app for an enhanced experience and more features!</p>
              )}
            </div>
          )}
          
          {messages.map(message => (
            <ChatBubble key={message.id} message={message} />
          ))}
          
          {currentStreamingContent && (
            <ChatBubble 
              message={{
                id: 'streaming',
                type: 'assistant',
                content: currentStreamingContent,
                timestamp: new Date().toISOString(),
                isStreaming: true
              }} 
            />
          )}
          
          <div ref={messagesEndRef} />
        </div>

        <MessageInput 
          onSendMessage={sendMessage}
        />

      <ApiKeyModal
        isOpen={showApiKeyModal}
        onClose={closeApiKeyModal}
        missingKeys={missingApiKeys}
        modelType={currentModelType}
        onSubmit={handleApiKeySubmit}
        serverUrl={settings.serverUrl}
      />

      <ClearChatModal
        isOpen={showClearModal}
        onClose={() => setShowClearModal(false)}
        onClearLocal={clearChatLocal}
        onClearPermanent={clearChatPermanent}
        isClearing={isClearing}
      />
    </div>
  );
};

export default ChatWindow; 