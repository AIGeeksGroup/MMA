import React, { useState, useEffect } from 'react';
import './ScreenshotCapture.css';

const ScreenshotCapture = ({ onScreenshotTaken, onClose }) => {
  const [displays, setDisplays] = useState([]);
  const [isCapturing, setIsCapturing] = useState(false);
  const [error, setError] = useState(null);
  const [isElectronAvailable, setIsElectronAvailable] = useState(false);

  useEffect(() => {
    // Check if we're running in Electron
    if (window.electronAPI && window.electronAPI.takeScreenshot) {
      setIsElectronAvailable(true);
      loadDisplays();
    } else {
      setError('Screenshot functionality is only available in the desktop app. Please use the desktop version of MMA.');
    }
  }, []);

  const loadDisplays = async () => {
    if (!window.electronAPI || !window.electronAPI.listDisplays) {
      setError('Display listing not available');
      return;
    }

    try {
      const result = await window.electronAPI.listDisplays();
      if (result.success) {
        setDisplays(result.displays);
      } else {
        setError('Failed to load displays: ' + result.error);
      }
    } catch (err) {
      setError('Failed to load displays: ' + err.message);
    }
  };

  const takeScreenshot = async (displayId = null) => {
    if (!window.electronAPI || !window.electronAPI.takeScreenshot) {
      setError('Screenshot functionality not available. Please use the desktop app.');
      return;
    }

    setIsCapturing(true);
    setError(null);

    try {
      let result;
      if (displayId !== null) {
        result = await window.electronAPI.takeScreenshotDisplay(displayId);
      } else {
        result = await window.electronAPI.takeScreenshot();
      }

      if (result.success) {
        // Create a file object-like structure for the screenshot
        const screenshotFile = {
          name: result.filename,
          path: result.filepath,
          type: 'image/png',
          size: result.size,
          isScreenshot: true,
          url: `file://${result.filepath}` // For preview
        };

        onScreenshotTaken(screenshotFile);
        onClose(); // Close the modal after successful screenshot
      } else {
        setError('Failed to take screenshot: ' + result.error);
      }
    } catch (err) {
      setError('Failed to take screenshot: ' + err.message);
    } finally {
      setIsCapturing(false);
    }
  };

  const formatBytes = (bytes) => {
    if (bytes === 0) return '0 Bytes';
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
  };

  return (
    <div className="screenshot-capture-modal">
      <div className="screenshot-capture-content">
        <div className="screenshot-header">
          <h3>📸 Take Screenshot</h3>
          <button className="close-button" onClick={onClose}>×</button>
        </div>

        {error && (
          <div className="error-message">
            ⚠️ {error}
            {!isElectronAvailable && (
              <div style={{ marginTop: '8px', fontSize: '12px' }}>
                <strong>Tip:</strong> Download the desktop app to use screenshot features.
              </div>
            )}
          </div>
        )}

        {isElectronAvailable && (
          <div className="screenshot-options">
            <div className="option-section">
              <h4>Quick Capture</h4>
              <button
                className="screenshot-button primary"
                onClick={() => takeScreenshot()}
                disabled={isCapturing}
              >
                {isCapturing ? '📸 Capturing...' : '📸 Capture All Screens'}
              </button>
            </div>

            {displays.length > 1 && (
              <div className="option-section">
                <h4>Select Display</h4>
                <div className="displays-grid">
                  {displays.map((display) => (
                    <button
                      key={display.index}
                      className="display-button"
                      onClick={() => takeScreenshot(display.index)}
                      disabled={isCapturing}
                    >
                      <div className="display-info">
                        <span className="display-name">{display.name}</span>
                        {display.bounds && (
                          <span className="display-resolution">
                            {display.bounds.width}×{display.bounds.height}
                          </span>
                        )}
                      </div>
                    </button>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        <div className="screenshot-info">
          <p>📁 Screenshots are saved to: <code>~/.mma/tmp/images/</code></p>
          <p>🔑 Keyboard shortcut: <kbd>Ctrl+Shift+S</kbd> (or <kbd>Cmd+Shift+S</kbd> on Mac)</p>
          {!isElectronAvailable && (
            <p style={{ color: '#dc3545', fontWeight: 'bold' }}>
              🖥️ Screenshot functionality requires the desktop app
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

export default ScreenshotCapture; 