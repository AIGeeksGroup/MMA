import React, { useState, useRef, useCallback } from 'react';
// import VoiceRecorder from './VoiceRecorder';
import './ScreenshotMonitor.css';
import queuedFetch from '../utils/requestQueue';

const ScreenshotMonitor = ({ settings }) => {
  const [isMonitoring, setIsMonitoring] = useState(false);
  const [screenshotCount, setScreenshotCount] = useState(0);
  const [deletedCount, setDeletedCount] = useState(0);
  const [lastProcessedTime, setLastProcessedTime] = useState(null);
  const [status, setStatus] = useState('idle');
  const [error, setError] = useState(null);
  const [skipSimilarityCheck, setSkipSimilarityCheck] = useState(false);
  const [isRequestInProgress, setIsRequestInProgress] = useState(false);
  const [isProcessingScreenshot, setIsProcessingScreenshot] = useState(false);
  
  // Voice recording state - COMMENTED OUT
  // const [voiceData, setVoiceData] = useState([]);
  // const voiceRecorderRef = useRef(null);
  
  const intervalRef = useRef(null);
  const lastImageDataRef = useRef(null);
  const abortControllerRef = useRef(null);

  // Configuration (matches main.py defaults)
  const INTERVAL = 2000; // 2 seconds (changed from 1 second)
  const SIMILARITY_THRESHOLD = 0.99;

  // Handle voice data from the recorder - COMMENTED OUT
  // const handleVoiceData = useCallback((data) => {
  //   setVoiceData(prev => [...prev, data]);
  //   console.log('Voice data accumulated:', data);
  // }, []);

  // Calculate image similarity using a simple pixel difference approach
  // Note: This is a simplified version compared to SSIM in main.py
  const calculateImageSimilarity = useCallback((imageData1, imageData2) => {
    if (!imageData1 || !imageData2) return 0;
    if (imageData1.length !== imageData2.length) return 0;

    let totalDiff = 0;
    const pixelCount = imageData1.length / 4; // RGBA channels

    for (let i = 0; i < imageData1.length; i += 4) {
      // Calculate grayscale values for comparison
      const gray1 = 0.299 * imageData1[i] + 0.587 * imageData1[i + 1] + 0.114 * imageData1[i + 2];
      const gray2 = 0.299 * imageData2[i] + 0.587 * imageData2[i + 1] + 0.114 * imageData2[i + 2];
      
      totalDiff += Math.abs(gray1 - gray2);
    }

    const averageDiff = totalDiff / pixelCount;
    const similarity = 1 - (averageDiff / 255); // Normalize to 0-1 range
    return Math.max(0, Math.min(1, similarity));
  }, []);

  // Convert canvas to image data for similarity comparison
  const getImageDataFromCanvas = useCallback((canvas) => {
    const ctx = canvas.getContext('2d');
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    return imageData.data;
  }, []);

  // Delete screenshot that is too similar
  const deleteSimilarScreenshot = useCallback(async (filepath) => {
    if (!window.electronAPI || !window.electronAPI.deleteScreenshot) {
      console.warn('Delete screenshot functionality not available');
      return;
    }

    try {
      const result = await window.electronAPI.deleteScreenshot(filepath);
      if (result.success) {
        console.log(`✅ Deleted similar screenshot: ${filepath}`);
        setDeletedCount(prev => prev + 1);
      } else {
        console.warn(`⚠️ Failed to delete screenshot: ${result.error}`);
      }
    } catch (error) {
      console.error('Error deleting screenshot:', error);
    }
  }, []);

  // Send screenshot to backend with memorizing=true and accumulated audio - VOICE FUNCTIONALITY COMMENTED OUT
  const sendScreenshotToBackend = useCallback(async (screenshotFile) => {
    if (!screenshotFile) return;

    // Skip if another request is already in progress
    if (isRequestInProgress) {
      console.log('Skipping screenshot - request already in progress');
      return;
    }

    let currentAbortController = null;
    let cleanup = null;

    try {
      setIsRequestInProgress(true);
      setStatus('sending');
      
      // Get accumulated audio from voice recorder - COMMENTED OUT
      // let accumulatedAudio = [];
      // let voiceFiles = [];
      
      // if (voiceRecorderRef.current && typeof voiceRecorderRef.current.getAccumulatedAudio === 'function') {
      //   accumulatedAudio = voiceRecorderRef.current.getAccumulatedAudio();
      // }

      // Convert audio blobs to base64 for sending with screenshot - COMMENTED OUT
      // if (accumulatedAudio.length > 0) {
      //   try {
      //     console.log(`Converting ${accumulatedAudio.length} audio chunks to base64`);
      //     
      //     for (const audioData of accumulatedAudio) {
      //       const arrayBuffer = await audioData.blob.arrayBuffer();
      //       const base64Data = btoa(String.fromCharCode(...new Uint8Array(arrayBuffer)));
      //       voiceFiles.push(base64Data);
      //     }
      //     
      //     console.log(`Successfully converted ${voiceFiles.length} audio chunks to base64`);
      //   } catch (audioError) {
      //     console.error('Error converting audio to base64:', audioError);
      //   }
      // }

      // Prepare the message with voice context info - COMMENTED OUT
      // let message = null;
      // if (voiceFiles.length > 0) {
      //   const totalDuration = accumulatedAudio.reduce((sum, audio) => sum + audio.duration, 0);
      //   message = `[Screenshot with voice recording: ${voiceFiles.length} audio chunks, ${(totalDuration/1000).toFixed(1)}s total]`;
      // }

      const requestData = {
        // message: message,
        image_uris: [screenshotFile.path],
        // voice_files: voiceFiles.length > 0 ? voiceFiles : null, // COMMENTED OUT
        memorizing: true // This is the key difference from chat
      };

      // Use a fresh abort controller for this request
      currentAbortController = new AbortController();
      abortControllerRef.current = currentAbortController;

      const result = await queuedFetch(`${settings.serverUrl}/send_streaming_message`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(requestData),
        signal: currentAbortController.signal,
        isStreaming: true
      });

      const response = result.response;
      cleanup = result.cleanup;

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      // We don't need to process the streaming response for monitoring
      // Just consume it to complete the request
      const reader = response.body.getReader();
      while (true) {
        const { done } = await reader.read();
        if (done) break;
      }

      setScreenshotCount(prev => prev + 1);
      setLastProcessedTime(new Date().toISOString());
      setStatus('monitoring');
      setError(null);

      console.log(`✅ Screenshot #${screenshotCount + 1} sent successfully to backend`);

      // Clear accumulated audio after successful send - COMMENTED OUT
      // if (voiceRecorderRef.current && typeof voiceRecorderRef.current.clearAccumulatedAudio === 'function') {
      //   voiceRecorderRef.current.clearAccumulatedAudio();
      // }

    } catch (err) {
      if (err.name === 'AbortError') {
        console.log('Screenshot request aborted');
      } else {
        console.error('Error sending screenshot:', err);
        setError(`Failed to send screenshot: ${err.message}`);
      }
    } finally {
      setIsRequestInProgress(false);
      // Clear the abort controller if it's still the current one
      if (abortControllerRef.current?.signal === currentAbortController?.signal) {
        abortControllerRef.current = null;
      }
      
      // Call cleanup to notify request queue
      if (cleanup) {
        cleanup();
      }
    }
  }, [settings.serverUrl, isRequestInProgress]);

  // Take and process a screenshot
  const processScreenshot = useCallback(async () => {
    if (!window.electronAPI || !window.electronAPI.takeScreenshot) {
      setError('Screenshot functionality requires desktop app');
      return;
    }

    // Skip if already processing a screenshot or if a request is in progress
    if (isProcessingScreenshot || isRequestInProgress) {
      console.log('Skipping screenshot - already processing or request in progress');
      return;
    }

    try {
      setIsProcessingScreenshot(true);
      setStatus('capturing');

      // Take screenshot
      const result = await window.electronAPI.takeScreenshot();
      
      if (!result.success) {
        throw new Error(result.error || 'Failed to take screenshot');
      }

      const screenshotFile = {
        name: result.filename,
        path: result.filepath,
        type: 'image/png',
        size: result.size,
        isScreenshot: true
      };

      // If similarity check is disabled, send every screenshot
      if (skipSimilarityCheck) {
        console.log('Similarity check disabled, sending screenshot');
        // Check again right before sending to prevent race condition
        if (!isRequestInProgress) {
          sendScreenshotToBackend(screenshotFile);
        } else {
          console.log('Request started while processing, skipping this screenshot');
        }
        setStatus('monitoring');
        return;
      }

      // Read image as base64 for similarity comparison
      const imageResult = await window.electronAPI.readImageAsBase64(result.filepath);
      
      if (!imageResult.success) {
        console.warn('Failed to read image for similarity comparison, sending anyway:', imageResult.error);
        // If we can't read the image for comparison, just send it
        // Check again right before sending to prevent race condition
        if (!isRequestInProgress) {
          sendScreenshotToBackend(screenshotFile);
        } else {
          console.log('Request started while processing, skipping this screenshot');
          // Delete the screenshot since it won't be sent
          deleteSimilarScreenshot(screenshotFile.path);
        }
        setStatus('monitoring');
        return;
      }

      // Create a temporary canvas to get image data for similarity comparison
      const img = new Image();
      img.onload = () => {
        // CRITICAL: Check again here since this is async and may execute much later
        if (isRequestInProgress) {
          console.log('Request started while processing similarity, skipping this screenshot');
          setStatus('monitoring');
          setIsProcessingScreenshot(false);
          return;
        }

        const canvas = document.createElement('canvas');
        canvas.width = img.naturalWidth;
        canvas.height = img.naturalHeight;
        const ctx = canvas.getContext('2d');
        ctx.drawImage(img, 0, 0);

        const currentImageData = getImageDataFromCanvas(canvas);
        
        // Check similarity with last image
        let similarity = 0;
        if (lastImageDataRef.current) {
          similarity = calculateImageSimilarity(lastImageDataRef.current, currentImageData);
        }

        console.log(`Screenshot similarity: ${similarity.toFixed(3)} (threshold: ${SIMILARITY_THRESHOLD}) - ${similarity < SIMILARITY_THRESHOLD ? 'DIFFERENT ENOUGH → will send' : 'TOO SIMILAR → will delete'}`);

        // Only send if different enough (below threshold)
        if (similarity < SIMILARITY_THRESHOLD) {
          // Final check right before sending
          if (!isRequestInProgress) {
            sendScreenshotToBackend(screenshotFile);
            lastImageDataRef.current = currentImageData;
          } else {
            console.log('Request started during similarity check, skipping this screenshot');
            // Delete the screenshot since it won't be sent
            deleteSimilarScreenshot(screenshotFile.path);
          }
        } else {
          console.log(`Screenshot too similar (${similarity.toFixed(3)} >= ${SIMILARITY_THRESHOLD}), deleting file`);
          // Delete the screenshot since it's too similar
          deleteSimilarScreenshot(screenshotFile.path);
          setStatus('monitoring');
        }
        setIsProcessingScreenshot(false);
      };

      img.onerror = () => {
        console.error('Failed to load screenshot for similarity comparison');
        // If image loading fails, just send the screenshot anyway
        // Check again right before sending to prevent race condition
        if (!isRequestInProgress) {
          sendScreenshotToBackend(screenshotFile);
        } else {
          console.log('Request started while processing, skipping this screenshot');
          // Delete the screenshot since it won't be sent
          deleteSimilarScreenshot(screenshotFile.path);
        }
        setStatus('monitoring');
        setIsProcessingScreenshot(false);
      };

      // Use the base64 data URL instead of file:// URL
      img.src = imageResult.dataUrl;

    } catch (err) {
      console.error('Error processing screenshot:', err);
      setError(`Error processing screenshot: ${err.message}`);
      setStatus('monitoring');
      setIsProcessingScreenshot(false);
    }
  }, [calculateImageSimilarity, getImageDataFromCanvas, sendScreenshotToBackend, skipSimilarityCheck, isRequestInProgress, isProcessingScreenshot]);

  // Start monitoring
  const startMonitoring = useCallback(() => {
    if (isMonitoring) return;

    if (!window.electronAPI || !window.electronAPI.takeScreenshot) {
      setError('Screenshot functionality is only available in the desktop app');
      return;
    }

    setIsMonitoring(true);
    setStatus('monitoring');
    setError(null);
    setScreenshotCount(0);
    setDeletedCount(0);
    lastImageDataRef.current = null;

    // Start the interval
    intervalRef.current = setInterval(processScreenshot, INTERVAL);

    // Take first screenshot immediately
    processScreenshot();
  }, [isMonitoring, processScreenshot]);

  // Stop monitoring
  const stopMonitoring = useCallback(() => {
    if (!isMonitoring) return;

    setIsMonitoring(false);
    setStatus('idle');

    // Clear interval
    if (intervalRef.current) {
      clearInterval(intervalRef.current);
      intervalRef.current = null;
    }

    // Abort any pending request
    if (abortControllerRef.current) {
      abortControllerRef.current.abort();
      abortControllerRef.current = null;
    }

    // Clear image data reference
    lastImageDataRef.current = null;
    
    // Reset request and processing state
    setIsRequestInProgress(false);
    setIsProcessingScreenshot(false);
    
    // Reset counters (optional - you might want to keep them for reference)
    // setScreenshotCount(0);
    // setDeletedCount(0);
  }, [isMonitoring]);

  // Cleanup on unmount
  React.useEffect(() => {
    return () => {
      stopMonitoring();
    };
  }, [stopMonitoring]);

  const getStatusIcon = () => {
    switch (status) {
      case 'monitoring': return '👁️';
      case 'capturing': return '📸';
      case 'sending': return '📤';
      default: return '⏹️';
    }
  };

  const getStatusColor = () => {
    switch (status) {
      case 'monitoring': return '#28a745';
      case 'capturing': return '#ffc107';
      case 'sending': return '#17a2b8';
      default: return '#6c757d';
    }
  };

  return (
    <div className="screenshot-monitor">
      <div className="monitor-header">
        <h3>🎯 Screen Monitor</h3>
        <div className="monitor-controls">
          <label className="similarity-toggle">
            <input
              type="checkbox"
              checked={skipSimilarityCheck}
              onChange={(e) => setSkipSimilarityCheck(e.target.checked)}
              disabled={isMonitoring}
            />
            <span>Send all screenshots (skip similarity check)</span>
          </label>
          <button
            className={`monitor-toggle ${isMonitoring ? 'active' : ''}`}
            onClick={isMonitoring ? stopMonitoring : startMonitoring}
            style={{
              backgroundColor: isMonitoring ? '#dc3545' : '#28a745',
              color: 'white'
            }}
          >
            {isMonitoring ? '⏹️ Stop Monitor' : '▶️ Start Monitor'}
          </button>
        </div>
      </div>

      <div className="monitor-status">
        <div className="status-item">
          <span className="status-icon" style={{ color: getStatusColor() }}>
            {getStatusIcon()}
          </span>
          <span className="status-text">
            Status: <strong style={{ color: getStatusColor() }}>{status}</strong>
          </span>
        </div>
        
        <div className="status-item">
          <span>📊 Screenshots sent: <strong>{screenshotCount}</strong></span>
        </div>
        
        <div className="status-item">
          <span>🗑️ Similar screenshots deleted: <strong>{deletedCount}</strong></span>
        </div>
        
        {lastProcessedTime && (
          <div className="status-item">
            <span>🕒 Last sent: <strong>{new Date(lastProcessedTime).toLocaleTimeString()}</strong></span>
          </div>
        )}
      </div>

      {error && (
        <div className="monitor-error">
          ⚠️ {error}
        </div>
      )}

      {/* Voice Recording Component - COMMENTED OUT */}
      {/* <VoiceRecorder 
        ref={voiceRecorderRef}
        settings={settings}
        isMonitoring={isMonitoring}
        onVoiceData={handleVoiceData}
      /> */}
    </div>
  );
};

export default ScreenshotMonitor; 