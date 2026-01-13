import React, { useState, useEffect } from 'react';
import ChatWindow from './components/ChatWindow';
import SettingsPanel from './components/SettingsPanel';
import ScreenshotMonitor from './components/ScreenshotMonitor';
import ExistingMemory from './components/ExistingMemory';
import ApiKeyModal from './components/ApiKeyModal';
import Logo from './components/Logo';
import queuedFetch from './utils/requestQueue';
import './App.css';

function App() {
  const [activeTab, setActiveTab] = useState('chat');
  const [settings, setSettings] = useState({
    model: 'gpt-4o-mini',
    persona: 'helpful_assistant',
    timezone: 'America/New_York',
    serverUrl: 'http://localhost:8000'
  });

  // Lift chat messages state to App level to persist across tab switches
  const [chatMessages, setChatMessages] = useState([]);

  // API Key modal state
  const [apiKeyModal, setApiKeyModal] = useState({
    isOpen: false,
    missingKeys: [],
    modelType: ''
  });

  // Check for missing API keys on startup
  useEffect(() => {
    checkApiKeys();
  }, [settings.serverUrl]);

  // Also check API keys when model changes
  useEffect(() => {
    checkApiKeys();
  }, [settings.model]);

  const checkApiKeys = async (forceOpen = false) => {
    try {
      console.log(`Checking API keys for model: ${settings.model}`);
      const response = await queuedFetch(`${settings.serverUrl}/api_keys/check`);
      if (response.ok) {
        const data = await response.json();
        console.log('API key status:', data);
        
        if (forceOpen || (data.requires_api_key && data.missing_keys.length > 0)) {
          if (forceOpen) {
            console.log('Manual API key update requested');
          } else {
            console.log(`Missing API keys detected: ${data.missing_keys.join(', ')}`);
          }
          setApiKeyModal({
            isOpen: true,
            missingKeys: data.missing_keys,
            modelType: data.model_type
          });
        } else {
          console.log('All required API keys are available');
          setApiKeyModal({
            isOpen: false,
            missingKeys: [],
            modelType: ''
          });
        }
      } else {
        console.error('Failed to check API keys:', response.statusText);
      }
    } catch (error) {
      console.error('Error checking API keys:', error);
    }
  };

  const handleApiKeyModalClose = () => {
    setApiKeyModal(prev => ({ ...prev, isOpen: false }));
  };

  const handleApiKeySubmit = async () => {
    // Refresh API key status after submission
    await checkApiKeys();
  };

  useEffect(() => {
    // Listen for menu events from Electron
    const cleanupFunctions = [];
    
    if (window.electronAPI) {
      const cleanupNewChat = window.electronAPI.onMenuNewChat(() => {
        setActiveTab('chat');
        // Clear chat messages when creating new chat
        setChatMessages([]);
      });
      cleanupFunctions.push(cleanupNewChat);

      const cleanupOpenTerminal = window.electronAPI.onMenuOpenTerminal(() => {
        // Open terminal logic here
        console.log('Open terminal requested');
      });
      cleanupFunctions.push(cleanupOpenTerminal);

      const cleanupTakeScreenshot = window.electronAPI.onMenuTakeScreenshot(() => {
        // Switch to chat tab and let ChatWindow handle the screenshot
        setActiveTab('chat');
      });
      cleanupFunctions.push(cleanupTakeScreenshot);
    }

    // Cleanup listeners on unmount
    return () => {
      cleanupFunctions.forEach(cleanup => {
        if (cleanup) cleanup();
      });
    };
  }, []);

  const handleSettingsChange = (newSettings) => {
    setSettings(prev => ({ ...prev, ...newSettings }));
  };



  return (
    <div className="App">
      <div className="app-header">
        <div className="app-title">
          <Logo 
            size="small" 
            showText={false} 
          />
          <span className="version">v0.1.0</span>
        </div>
        <div className="tabs">
          <button 
            className={`tab ${activeTab === 'chat' ? 'active' : ''}`}
            onClick={() => setActiveTab('chat')}
          >
            💬 Chat
          </button>
          <button 
            className={`tab ${activeTab === 'screenshots' ? 'active' : ''}`}
            onClick={() => setActiveTab('screenshots')}
          >
            📸 Screenshots
          </button>
          <button 
            className={`tab ${activeTab === 'memory' ? 'active' : ''}`}
            onClick={() => setActiveTab('memory')}
          >
            🧠 Existing Memory
          </button>
          <button 
            className={`tab ${activeTab === 'settings' ? 'active' : ''}`}
            onClick={() => setActiveTab('settings')}
          >
            ⚙️ Settings
          </button>
        </div>
      </div>

      <div className="app-content">
        {/* Keep ChatWindow always mounted to maintain streaming state */}
        <div style={{ 
          display: activeTab === 'chat' ? 'flex' : 'none',
          flexDirection: 'column',
          height: '100%'
        }}>
          <ChatWindow
            settings={settings}
            messages={chatMessages}
            setMessages={setChatMessages}
            onApiKeyRequired={(missingKeys, modelType) => {
              setApiKeyModal({
                isOpen: true,
                missingKeys,
                modelType
              });
            }}
          />
        </div>
        {/* Keep ScreenshotMonitor always mounted to maintain monitoring state */}
        <div style={{ display: activeTab === 'screenshots' ? 'block' : 'none' }}>
          <ScreenshotMonitor settings={settings} />
        </div>
        {activeTab === 'memory' && (
          <ExistingMemory settings={settings} />
        )}
        {activeTab === 'settings' && (
          <SettingsPanel
            settings={settings}
            onSettingsChange={handleSettingsChange}
            onApiKeyCheck={checkApiKeys}
          />
        )}
      </div>

      {/* API Key Modal */}
      <ApiKeyModal
        isOpen={apiKeyModal.isOpen}
        missingKeys={apiKeyModal.missingKeys}
        modelType={apiKeyModal.modelType}
        onClose={handleApiKeyModalClose}
        serverUrl={settings.serverUrl}
        onSubmit={handleApiKeySubmit}
      />
    </div>
  );
}

export default App; 