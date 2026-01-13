import React, { useState } from 'react';
import './ApiKeyModal.css';
import queuedFetch from '../utils/requestQueue';

const ApiKeyModal = ({ isOpen, onClose, missingKeys, modelType, onSubmit, serverUrl }) => {
  const [selectedService, setSelectedService] = useState('');
  const [apiKeyValue, setApiKeyValue] = useState('');
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [error, setError] = useState('');

  // Define all available API services
  const apiServices = [
    { value: 'OPENAI_API_KEY', label: 'OpenAI API Key', description: 'For GPT models (starts with sk-)' },
    { value: 'ANTHROPIC_API_KEY', label: 'Anthropic API Key', description: 'For Claude models' },
    { value: 'GEMINI_API_KEY', label: 'Gemini API Key', description: 'For Google Gemini models' },
    { value: 'GROQ_API_KEY', label: 'Groq API Key', description: 'For Groq models' },
    { value: 'TOGETHER_API_KEY', label: 'Together AI API Key', description: 'For Together AI models' },
    { value: 'AZURE_API_KEY', label: 'Azure OpenAI API Key', description: 'For Azure OpenAI service' },
    { value: 'AZURE_BASE_URL', label: 'Azure Base URL', description: 'Azure OpenAI endpoint URL' },
    { value: 'AZURE_API_VERSION', label: 'Azure API Version', description: 'e.g., 2024-09-01-preview' },
    { value: 'AWS_ACCESS_KEY_ID', label: 'AWS Access Key ID', description: 'For AWS Bedrock' },
    { value: 'AWS_SECRET_ACCESS_KEY', label: 'AWS Secret Access Key', description: 'For AWS Bedrock' },
    { value: 'AWS_REGION', label: 'AWS Region', description: 'e.g., us-east-1' },
  ];

  const getKeyPlaceholder = (keyName) => {
    const placeholders = {
      'OPENAI_API_KEY': 'sk-...',
      'ANTHROPIC_API_KEY': 'sk-ant-...',
      'GEMINI_API_KEY': 'AI...',
      'AZURE_BASE_URL': 'https://your-resource.openai.azure.com',
      'AZURE_API_VERSION': '2024-09-01-preview',
      'AWS_REGION': 'us-east-1',
    };
    return placeholders[keyName] || 'Enter your API key...';
  };

  const isMissingKeysMode = missingKeys && missingKeys.length > 0;

  const handleSubmit = async (e) => {
    e.preventDefault();
    setIsSubmitting(true);
    setError('');

    try {
      if (isMissingKeysMode) {
        // Handle missing keys mode - submit all missing keys
        const apiKeys = {};
        missingKeys.forEach(keyName => {
          const input = document.getElementById(keyName);
          if (input && input.value) {
            apiKeys[keyName] = input.value;
          }
        });

        for (const keyName of missingKeys) {
          if (apiKeys[keyName]) {
            const response = await queuedFetch(`${serverUrl}/api_keys/update`, {
              method: 'POST',
              headers: {
                'Content-Type': 'application/json',
              },
              body: JSON.stringify({
                key_name: keyName,
                key_value: apiKeys[keyName]
              }),
            });

            if (!response.ok) {
              const errorData = await response.json();
              throw new Error(errorData.detail || `Failed to update ${keyName}`);
            }
          }
        }
      } else {
        // Handle manual update mode - submit selected service
        if (!selectedService || !apiKeyValue) {
          setError('Please select a service and enter an API key');
          return;
        }

        const response = await queuedFetch(`${serverUrl}/api_keys/update`, {
          method: 'POST',
          headers: {
            'Content-Type': 'application/json',
          },
          body: JSON.stringify({
            key_name: selectedService,
            key_value: apiKeyValue
          }),
        });

        if (!response.ok) {
          const errorData = await response.json();
          throw new Error(errorData.detail || `Failed to update ${selectedService}`);
        }
      }

      // Show initialization message
      setError('🔄 Initializing Agents with new API key...');
      
      // Small delay to show the message before closing
      await new Promise(resolve => setTimeout(resolve, 1500));

      // Call the onSubmit callback to refresh the parent component
      onSubmit();
      onClose();
    } catch (err) {
      setError(err.message || 'Failed to update API keys');
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleServiceChange = (e) => {
    setSelectedService(e.target.value);
    setApiKeyValue(''); // Clear the input when service changes
    setError(''); // Clear any errors
  };

  const getSelectedServiceInfo = () => {
    return apiServices.find(service => service.value === selectedService);
  };

  if (!isOpen) return null;

  return (
    <div className="api-key-modal-overlay">
      <div className="api-key-modal">
        <div className="api-key-modal-header">
          <h2>🔑 {isMissingKeysMode ? 'API Keys Required' : 'Update API Keys'}</h2>
          {isMissingKeysMode ? (
            <p>
              The <strong>{modelType}</strong> model requires the following API keys to function properly:
            </p>
          ) : (
            <p>
              Select the API service you want to update and enter your new API key:
            </p>
          )}
        </div>

        <form onSubmit={handleSubmit} className="api-key-form">
          {isMissingKeysMode ? (
            // Missing keys mode - show all missing keys
            missingKeys.map((keyName) => {
              const serviceInfo = apiServices.find(s => s.value === keyName);
              return (
                <div key={keyName} className="api-key-field">
                  <label htmlFor={keyName}>
                    <strong>{serviceInfo ? serviceInfo.label : keyName}</strong>
                    <span className="key-description">{serviceInfo ? serviceInfo.description : `Your ${keyName}`}</span>
                  </label>
                  <input
                    type={keyName.includes('SECRET') || keyName.includes('KEY') ? 'password' : 'text'}
                    id={keyName}
                    placeholder={getKeyPlaceholder(keyName)}
                    required
                    className="api-key-input"
                  />
                </div>
              );
            })
          ) : (
            // Manual update mode - show dropdown and single input
            <>
              <div className="api-key-field">
                <label htmlFor="service-select">
                  <strong>Select API Service</strong>
                  <span className="key-description">Choose which API service you want to update</span>
                </label>
                <select
                  id="service-select"
                  value={selectedService}
                  onChange={handleServiceChange}
                  required
                  className="api-key-select"
                >
                  <option value="">-- Select a service --</option>
                  {apiServices.map((service) => (
                    <option key={service.value} value={service.value}>
                      {service.label}
                    </option>
                  ))}
                </select>
              </div>

              {selectedService && (
                <div className="api-key-field">
                  <label htmlFor="api-key-input">
                    <strong>{getSelectedServiceInfo()?.label}</strong>
                    <span className="key-description">{getSelectedServiceInfo()?.description}</span>
                  </label>
                  <input
                    type={selectedService.includes('SECRET') || selectedService.includes('KEY') ? 'password' : 'text'}
                    id="api-key-input"
                    value={apiKeyValue}
                    onChange={(e) => setApiKeyValue(e.target.value)}
                    placeholder={getKeyPlaceholder(selectedService)}
                    required
                    className="api-key-input"
                  />
                </div>
              )}
            </>
          )}

          {error && (
            <div className={`api-key-error ${error.includes('Initializing') ? 'api-key-info' : ''}`}>
              {error.includes('Initializing') ? error : `❌ ${error}`}
            </div>
          )}

          <div className="api-key-modal-actions">
            <button
              type="button"
              onClick={onClose}
              className="api-key-cancel-btn"
              disabled={isSubmitting}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="api-key-submit-btn"
              disabled={isSubmitting || (isMissingKeysMode ? false : (!selectedService || !apiKeyValue))}
            >
              {isSubmitting ? (error.includes('Initializing') ? '🔄 Initializing...' : '⏳ Saving...') : '✅ Save API Keys'}
            </button>
          </div>
        </form>

        <div className="api-key-note">
          <p>
            <strong>Note:</strong> Your API keys will be saved securely to your .env file for permanent storage and will persist across sessions.
          </p>
        </div>
      </div>
    </div>
  );
};

export default ApiKeyModal; 