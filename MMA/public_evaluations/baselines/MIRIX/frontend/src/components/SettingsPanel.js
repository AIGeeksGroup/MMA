import React, { useState, useEffect, useCallback } from 'react';
import './SettingsPanel.css';
import queuedFetch from '../utils/requestQueue';

const SettingsPanel = ({ settings, onSettingsChange, onApiKeyCheck, isVisible }) => {
  const [personaDetails, setPersonaDetails] = useState({});
  const [selectedPersonaText, setSelectedPersonaText] = useState('');
  const [isUpdatingPersona, setIsUpdatingPersona] = useState(false);
  const [isApplyingTemplate, setIsApplyingTemplate] = useState(false);
  const [updateMessage, setUpdateMessage] = useState('');
  const [isChangingModel, setIsChangingModel] = useState(false);
  const [modelUpdateMessage, setModelUpdateMessage] = useState('');
  const [isChangingMemoryModel, setIsChangingMemoryModel] = useState(false);
  const [memoryModelUpdateMessage, setMemoryModelUpdateMessage] = useState('');
  const [isChangingTimezone, setIsChangingTimezone] = useState(false);
  const [timezoneUpdateMessage, setTimezoneUpdateMessage] = useState('');
  const [isCheckingApiKeys, setIsCheckingApiKeys] = useState(false);
  const [apiKeyMessage, setApiKeyMessage] = useState('');

  // Debug logging for settings
  useEffect(() => {
    console.log('SettingsPanel: settings changed:', {
      serverUrl: settings.serverUrl,
      model: settings.model,
      persona: settings.persona,
      timezone: settings.timezone
    });
  }, [settings]);

  const handleInputChange = (key, value) => {
    onSettingsChange({ [key]: value });
  };

  const fetchPersonaDetails = useCallback(async () => {
    if (!settings.serverUrl) {
      console.log('fetchPersonaDetails: serverUrl not available yet');
      return;
    }
    try {
      const response = await queuedFetch(`${settings.serverUrl}/personas`);
      if (response.ok) {
        const data = await response.json();
        setPersonaDetails(data.personas);
      } else {
        console.error('Failed to fetch persona details');
      }
    } catch (error) {
      console.error('Error fetching persona details:', error);
    }
  }, [settings.serverUrl]);

  const fetchCoreMemoryPersona = useCallback(async () => {
    if (!settings.serverUrl) {
      console.log('fetchCoreMemoryPersona: serverUrl not available yet');
      return;
    }
    try {
      const response = await queuedFetch(`${settings.serverUrl}/personas/core_memory`);
      if (response.ok) {
        const data = await response.json();
        setSelectedPersonaText(data.text);
      } else {
        console.error('Failed to fetch core memory persona');
      }
    } catch (error) {
      console.error('Error fetching core memory persona:', error);
    }
  }, [settings.serverUrl]);

  const fetchCurrentModel = useCallback(async () => {
    if (!settings.serverUrl) {
      console.log('fetchCurrentModel: serverUrl not available yet');
      return;
    }
    try {
      const response = await queuedFetch(`${settings.serverUrl}/models/current`);
      if (response.ok) {
        const data = await response.json();
        // Only update if the model is different from current settings
        if (data.current_model !== settings.model) {
          onSettingsChange({ model: data.current_model });
        }
      } else {
        console.error('Failed to fetch current model');
      }
    } catch (error) {
      console.error('Error fetching current model:', error);
    }
  }, [settings.serverUrl, settings.model, onSettingsChange]);

  const fetchCurrentMemoryModel = useCallback(async () => {
    if (!settings.serverUrl) {
      console.log('fetchCurrentMemoryModel: serverUrl not available yet');
      return;
    }
    try {
      const response = await queuedFetch(`${settings.serverUrl}/models/memory/current`);
      if (response.ok) {
        const data = await response.json();
        // Only update if the memory model is different from current settings
        if (data.current_model !== settings.memoryModel) {
          onSettingsChange({ memoryModel: data.current_model });
        }
      } else {
        console.error('Failed to fetch current memory model');
      }
    } catch (error) {
      console.error('Error fetching current memory model:', error);
    }
  }, [settings.serverUrl, settings.memoryModel, onSettingsChange]);

  const applyPersonaTemplate = useCallback(async (personaName) => {
    if (!settings.serverUrl) {
      console.log('applyPersonaTemplate: serverUrl not available yet');
      return;
    }
    try {
      const response = await queuedFetch(`${settings.serverUrl}/personas/apply_template`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          persona_name: personaName,
        }),
      });

      if (response.ok) {
        // Refresh the core memory persona text
        fetchCoreMemoryPersona();
        console.log(`Applied persona template: ${personaName}`);
      } else {
        console.error('Failed to apply persona template');
      }
    } catch (error) {
      console.error('Error applying persona template:', error);
    }
  }, [settings.serverUrl, fetchCoreMemoryPersona]);

  // Fetch initial data only when serverUrl is available
  useEffect(() => {
    if (settings.serverUrl) {
      console.log('SettingsPanel: serverUrl is available, fetching initial data');
      fetchPersonaDetails();
      fetchCoreMemoryPersona();
      fetchCurrentModel();
      fetchCurrentMemoryModel();
    }
  }, [settings.serverUrl, fetchPersonaDetails, fetchCoreMemoryPersona, fetchCurrentModel, fetchCurrentMemoryModel]);

  // Fetch current models whenever settings panel becomes visible
  useEffect(() => {
    if (isVisible && settings.serverUrl) {
      console.log('SettingsPanel: became visible, refreshing current models');
      fetchCurrentModel();
      fetchCurrentMemoryModel();
    }
  }, [isVisible, settings.serverUrl, fetchCurrentModel, fetchCurrentMemoryModel]);

  // Apply persona template only on initial load when we have both persona data and no text loaded yet
  useEffect(() => {
    if (settings.persona && Object.keys(personaDetails).length > 0 && !selectedPersonaText) {
      applyPersonaTemplate(settings.persona);
    }
  }, [settings.persona, personaDetails, selectedPersonaText, applyPersonaTemplate]);

  const handlePersonaChange = async (newPersona) => {
    console.log('handlePersonaChange called with:', newPersona);
    console.log('personaDetails:', personaDetails);
    console.log('settings.serverUrl:', settings.serverUrl);
    
    if (!settings.serverUrl) {
      console.error('Cannot change persona: serverUrl not available');
      setUpdateMessage('❌ Server not available');
      return;
    }
    
    handleInputChange('persona', newPersona);
    
    // Apply the template regardless of whether personaDetails is loaded
    // The backend will handle checking if the persona exists
    setIsApplyingTemplate(true);
    setUpdateMessage('Applying persona template...');
    
    try {
      console.log(`Applying persona template: ${newPersona}`);
      
      const response = await queuedFetch(`${settings.serverUrl}/personas/apply_template`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          persona_name: newPersona,
        }),
      });

      console.log('Response status:', response.status);
      
      if (response.ok) {
        // Immediately refresh the core memory persona text
        await fetchCoreMemoryPersona();
        setUpdateMessage('✅ Persona template applied successfully!');
        console.log(`Successfully applied and updated persona: ${newPersona}`);
      } else {
        const errorData = await response.text();
        console.error('Failed to apply persona template:', errorData);
        setUpdateMessage('❌ Failed to apply persona template');
      }
    } catch (error) {
      console.error('Error applying persona template:', error);
      setUpdateMessage('❌ Error applying persona template');
    } finally {
      setIsApplyingTemplate(false);
      // Clear message after 2 seconds
      setTimeout(() => setUpdateMessage(''), 2000);
    }
  };

  const handleModelChange = async (newModel) => {
    console.log('handleModelChange called with:', newModel);
    
    if (!settings.serverUrl) {
      console.error('Cannot change model: serverUrl not available');
      setModelUpdateMessage('❌ Server not available');
      return;
    }
    
    handleInputChange('model', newModel);
    
    setIsChangingModel(true);
    setModelUpdateMessage('Changing chat agent model...');
    
    try {
      console.log(`Setting chat agent model: ${newModel}`);
      
      const response = await queuedFetch(`${settings.serverUrl}/models/set`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: newModel,
        }),
      });

      console.log('Response status:', response.status);
      
      if (response.ok) {
        const data = await response.json();
        setModelUpdateMessage('✅ Chat agent model set successfully!');
        console.log(`Successfully set chat agent model: ${newModel}`);
        
        // Automatically check for API keys after model change
        if (onApiKeyCheck) {
          console.log('Checking API keys for new model...');
          setModelUpdateMessage('✅ Model set! Checking API keys...');
          setTimeout(() => {
            onApiKeyCheck();
          }, 500); // Small delay to allow backend to update
        }
      } else {
        const errorData = await response.text();
        console.error('Failed to set chat agent model:', errorData);
        setModelUpdateMessage('❌ Failed to set chat agent model');
      }
    } catch (error) {
      console.error('Error setting chat agent model:', error);
      setModelUpdateMessage('❌ Error setting chat agent model');
    } finally {
      setIsChangingModel(false);
      // Clear message after 2 seconds
      setTimeout(() => setModelUpdateMessage(''), 2000);
    }
  };

  const handleMemoryModelChange = async (newModel) => {
    console.log('handleMemoryModelChange called with:', newModel);
    
    if (!settings.serverUrl) {
      console.error('Cannot change memory model: serverUrl not available');
      setMemoryModelUpdateMessage('❌ Server not available');
      return;
    }
    
    handleInputChange('memoryModel', newModel);
    
    setIsChangingMemoryModel(true);
    setMemoryModelUpdateMessage('Changing memory manager model...');
    
    try {
      console.log(`Setting memory manager model: ${newModel}`);
      
      const response = await queuedFetch(`${settings.serverUrl}/models/memory/set`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          model: newModel,
        }),
      });

      console.log('Response status:', response.status);
      
      if (response.ok) {
        const data = await response.json();
        
        if (data.success) {
          setMemoryModelUpdateMessage('✅ Memory manager model set successfully!');
          console.log(`Successfully set memory manager model: ${newModel}`);
          
          // Automatically check for API keys after memory model change
          if (onApiKeyCheck) {
            console.log('Checking API keys for new memory model...');
            setMemoryModelUpdateMessage('✅ Memory model set! Checking API keys...');
            setTimeout(() => {
              onApiKeyCheck();
            }, 500); // Small delay to allow backend to update
          }
        } else {
          // Handle case where backend returned success: false
          let errorMessage = data.message || 'Failed to set memory manager model';
          if (data.missing_keys && data.missing_keys.length > 0) {
            errorMessage += ` (Missing API keys: ${data.missing_keys.join(', ')})`;
            // Trigger API key check for missing keys
            if (onApiKeyCheck) {
              setTimeout(() => {
                onApiKeyCheck();
              }, 1000);
            }
          }
          setMemoryModelUpdateMessage(`❌ ${errorMessage}`);
          console.error('Memory model set failed:', data);
        }
      } else {
        const errorData = await response.text();
        console.error('Failed to set memory manager model:', errorData);
        setMemoryModelUpdateMessage('❌ Failed to set memory manager model');
      }
    } catch (error) {
      console.error('Error setting memory manager model:', error);
      setMemoryModelUpdateMessage('❌ Error setting memory manager model');
    } finally {
      setIsChangingMemoryModel(false);
      // Clear message after 2 seconds
      setTimeout(() => setMemoryModelUpdateMessage(''), 2000);
    }
  };

  const handleTimezoneChange = async (newTimezone) => {
    console.log('handleTimezoneChange called with:', newTimezone);
    
    if (!settings.serverUrl) {
      console.error('Cannot change timezone: serverUrl not available');
      setTimezoneUpdateMessage('❌ Server not available');
      return;
    }
    
    handleInputChange('timezone', newTimezone);
    
    setIsChangingTimezone(true);
    setTimezoneUpdateMessage('Changing timezone...');
    
    try {
      console.log(`Setting timezone: ${newTimezone}`);
      
      const response = await queuedFetch(`${settings.serverUrl}/timezone/set`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          timezone: newTimezone,
        }),
      });

      console.log('Response status:', response.status);
      
      if (response.ok) {
        const data = await response.json();
        setTimezoneUpdateMessage('✅ Timezone set successfully!');
        console.log(`Successfully set timezone: ${newTimezone}`);
      } else {
        const errorData = await response.text();
        console.error('Failed to set timezone:', errorData);
        setTimezoneUpdateMessage('❌ Failed to set timezone');
      }
    } catch (error) {
      console.error('Error setting timezone:', error);
      setTimezoneUpdateMessage('❌ Error setting timezone');
    } finally {
      setIsChangingTimezone(false);
      // Clear message after 2 seconds
      setTimeout(() => setTimezoneUpdateMessage(''), 2000);
    }
  };

  const handlePersonaTextChange = (event) => {
    setSelectedPersonaText(event.target.value);
  };

  const updatePersonaText = async () => {
    if (!settings.serverUrl) {
      console.error('Cannot update persona text: serverUrl not available');
      setUpdateMessage('❌ Server not available');
      return;
    }
    
    setIsUpdatingPersona(true);
    setUpdateMessage('Updating core memory persona...');
    
    try {
      const response = await queuedFetch(`${settings.serverUrl}/personas/update`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          text: selectedPersonaText,
        }),
      });

      if (response.ok) {
        setUpdateMessage('✅ Core memory persona updated successfully!');
      } else {
        const errorData = await response.text();
        console.error('Failed to update core memory persona:', errorData);
        setUpdateMessage('❌ Failed to update core memory persona');
      }
    } catch (error) {
      console.error('Error updating core memory persona:', error);
      setUpdateMessage('❌ Error updating core memory persona');
    } finally {
      setIsUpdatingPersona(false);
      // Clear message after 2 seconds
      setTimeout(() => setUpdateMessage(''), 2000);
    }
  };

  const models = [
    'gpt-4o-mini',
    'gpt-4o',
    'gpt-4.1-mini',
    'gpt-4.1',
    'claude-3-5-sonnet-20241022',
    'gemini-2.0-flash',
    'gemini-2.5-flash',
    'gemini-2.5-flash-preview-04-17',
    'gemini-1.5-pro',
    'gemini-2.0-flash-lite'
  ];

  // Memory models are restricted to specific Gemini models only
  const memoryModels = [
    'gemini-2.0-flash',
    'gemini-2.5-flash',
    'gemini-2.5-flash-preview-04-17'
  ];

  // Convert personaDetails object to array format for dropdown
  const personas = Object.keys(personaDetails).map(key => ({
    value: key,
    label: key.split('_').map(word => word.charAt(0).toUpperCase() + word.slice(1)).join(' ')
  }));

  const timezones = [
    'America/New_York (UTC-5:00)',
    'America/Los_Angeles (UTC-8:00)',
    'America/Chicago (UTC-6:00)',
    'America/Denver (UTC-7:00)',
    'America/Toronto (UTC-5:00)',
    'America/Vancouver (UTC-8:00)',
    'Europe/London (UTC+0:00)',
    'Europe/Paris (UTC+1:00)',
    'Europe/Berlin (UTC+1:00)',
    'Europe/Rome (UTC+1:00)',
    'Europe/Madrid (UTC+1:00)',
    'Europe/Amsterdam (UTC+1:00)',
    'Europe/Stockholm (UTC+1:00)',
    'Europe/Moscow (UTC+3:00)',
    'Asia/Tokyo (UTC+9:00)',
    'Asia/Shanghai (UTC+8:00)',
    'Asia/Seoul (UTC+9:00)',
    'Asia/Hong_Kong (UTC+8:00)',
    'Asia/Singapore (UTC+8:00)',
    'Asia/Bangkok (UTC+7:00)',
    'Asia/Jakarta (UTC+7:00)',
    'Asia/Manila (UTC+8:00)',
    'Asia/Kolkata (UTC+5:30)',
    'Asia/Dubai (UTC+4:00)',
    'Asia/Tehran (UTC+3:30)',
    'Australia/Sydney (UTC+10:00)',
    'Australia/Melbourne (UTC+10:00)',
    'Australia/Perth (UTC+8:00)',
    'Pacific/Auckland (UTC+12:00)',
    'Pacific/Honolulu (UTC-10:00)',
    'Africa/Cairo (UTC+2:00)',
    'Africa/Lagos (UTC+1:00)',
    'Africa/Johannesburg (UTC+2:00)',
    'America/Sao_Paulo (UTC-3:00)',
    'America/Buenos_Aires (UTC-3:00)',
    'America/Mexico_City (UTC-6:00)'
  ];

  return (
    <div className="settings-panel">
      <div className="settings-header">
        <h2>Settings</h2>
        <p>Configure your MIRIX assistant</p>
      </div>

      <div className="settings-content">
        <div className="settings-section">
          <h3>Model Configuration</h3>
          
          <div className="setting-item">
            <label htmlFor="model-select">Chat Agent Model</label>
            <select
              id="model-select"
              value={settings.model}
              onChange={(e) => handleModelChange(e.target.value)}
              className="setting-select"
              disabled={isChangingModel}
            >
              {models.map(model => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
            <span className="setting-description">
              {isChangingModel ? 'Changing chat agent model...' : 'Choose the AI model for chat responses'}
            </span>
            {modelUpdateMessage && (
              <span className={`update-message ${modelUpdateMessage.includes('✅') ? 'success' : modelUpdateMessage.includes('Changing') ? 'info' : 'error'}`}>
                {modelUpdateMessage}
              </span>
            )}
          </div>

          <div className="setting-item">
            <label htmlFor="memory-model-select">Memory Manager Model</label>
            <select
              id="memory-model-select"
              value={settings.memoryModel || settings.model}
              onChange={(e) => handleMemoryModelChange(e.target.value)}
              className="setting-select"
              disabled={isChangingMemoryModel}
            >
              {memoryModels.map(model => (
                <option key={model} value={model}>
                  {model}
                </option>
              ))}
            </select>
            <span className="setting-description">
              {isChangingMemoryModel ? 'Changing memory manager model...' : 'Choose the AI model for memory management operations'}
            </span>
            {memoryModelUpdateMessage && (
              <span className={`update-message ${memoryModelUpdateMessage.includes('✅') ? 'success' : memoryModelUpdateMessage.includes('Changing') ? 'info' : 'error'}`}>
                {memoryModelUpdateMessage}
              </span>
            )}
          </div>

          <div className="setting-item">
            <label htmlFor="persona-select">Persona</label>
            <select
              id="persona-select"
              value={settings.persona}
              onChange={(e) => handlePersonaChange(e.target.value)}
              className="setting-select"
              disabled={isApplyingTemplate}
            >
              {personas.map(persona => (
                <option key={persona.value} value={persona.value}>
                  {persona.label}
                </option>
              ))}
            </select>
            <span className="setting-description">
              {isApplyingTemplate ? 'Applying persona template...' : 'Set the personality style of the assistant'}
            </span>
          </div>

          {/* Persona Text Editor */}
          <div className="setting-item persona-editor">
            <label htmlFor="persona-text">Agent's Active Persona (Core Memory)</label>
            <textarea
              id="persona-text"
              value={selectedPersonaText}
              onChange={handlePersonaTextChange}
              className="persona-textarea"
              placeholder="This shows the agent's current active persona from core memory..."
              rows={6}
              disabled={isApplyingTemplate}
            />
            <div className="persona-editor-actions">
              <button
                onClick={updatePersonaText}
                disabled={isUpdatingPersona || isApplyingTemplate}
                className="update-persona-btn"
              >
                {isUpdatingPersona ? 'Updating...' : 'Update Core Memory'}
              </button>
              {updateMessage && (
                <span className={`update-message ${updateMessage.includes('✅') ? 'success' : updateMessage.includes('Applying') ? 'info' : 'error'}`}>
                  {updateMessage}
                </span>
              )}
            </div>
            <span className="setting-description">
              This shows the agent's current active persona from core memory. Select a persona from the dropdown above to apply a template, then edit and save custom changes here.
            </span>
          </div>
        </div>

        <div className="settings-section">
          <h3>Preferences</h3>
          
          <div className="setting-item">
            <label htmlFor="timezone-select">Timezone</label>
            <select
              id="timezone-select"
              value={settings.timezone}
              onChange={(e) => handleTimezoneChange(e.target.value)}
              className="setting-select"
              disabled={isChangingTimezone}
            >
              {timezones.map(tz => (
                <option key={tz} value={tz}>
                  {tz.replace(/_/g, ' ')}
                </option>
              ))}
            </select>
            <span className="setting-description">
              {isChangingTimezone ? 'Changing timezone...' : 'Your local timezone for timestamps'}
            </span>
            {timezoneUpdateMessage && (
              <span className={`update-message ${timezoneUpdateMessage.includes('✅') ? 'success' : timezoneUpdateMessage.includes('Changing') ? 'info' : 'error'}`}>
                {timezoneUpdateMessage}
              </span>
            )}
          </div>
        </div>

        <div className="settings-section">
          <h3>API Keys</h3>
          
          <div className="setting-item">
            <label>API Key Management</label>
            <div className="api-key-actions">
              <button
                onClick={() => {
                  // Force API key modal to open for manual updates
                  if (onApiKeyCheck) {
                    onApiKeyCheck(true); // Pass true to force modal open regardless of missing keys
                  }
                }}
                className="api-key-update-btn"
              >
                🔧 Update API Keys
              </button>
              {apiKeyMessage && (
                <span className={`update-message ${apiKeyMessage.includes('✅') ? 'success' : apiKeyMessage.includes('Checking') ? 'info' : 'error'}`}>
                  {apiKeyMessage}
                </span>
              )}
            </div>
            <span className="setting-description">
              Configure and update your API keys for different AI models and services.
            </span>
          </div>
        </div>

        <div className="settings-section">
          <h3>About</h3>
          <div className="about-info">
            <p><strong>MIRIX Desktop</strong></p>
            <p>Version 1.0.0</p>
            <p>AI Assistant powered by advanced language models</p>
            <div className="about-links">
              <button 
                className="link-button"
                onClick={() => window.open('https://github.com/your-repo/mirix', '_blank')}
              >
                📖 Documentation
              </button>
              <button 
                className="link-button"
                onClick={() => window.open('https://github.com/your-repo/mirix/issues', '_blank')}
              >
                🐛 Report Issue
              </button>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
};

export default SettingsPanel; 