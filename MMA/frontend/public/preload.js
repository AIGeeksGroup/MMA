const { contextBridge, ipcRenderer } = require('electron');

// Expose protected methods that allow the renderer process to use
// the ipcRenderer without exposing the entire object
contextBridge.exposeInMainWorld('electronAPI', {
  selectFiles: () => ipcRenderer.invoke('select-files'),
  selectSavePath: (options) => ipcRenderer.invoke('select-save-path', options),
  

  
  // Screenshot functions
  takeScreenshot: (options) => ipcRenderer.invoke('take-screenshot', options),
  takeScreenshotDisplay: (displayId) => ipcRenderer.invoke('take-screenshot-display', displayId),
  listDisplays: () => ipcRenderer.invoke('list-displays'),
  cleanupScreenshots: (maxAge) => ipcRenderer.invoke('cleanup-screenshots', maxAge),
  
  // Image reading function for similarity comparison
  readImageAsBase64: (filepath) => ipcRenderer.invoke('read-image-base64', filepath),
  
  // Delete screenshot function (for removing similar screenshots)
  deleteScreenshot: (filepath) => ipcRenderer.invoke('delete-screenshot', filepath),
  
  // Menu event listeners - wrap callbacks to prevent passing non-serializable event objects
  onMenuNewChat: (callback) => {
    const wrappedCallback = (event, ...args) => callback(...args);
    ipcRenderer.on('menu-new-chat', wrappedCallback);
    return () => ipcRenderer.removeListener('menu-new-chat', wrappedCallback);
  },
  onMenuOpenTerminal: (callback) => {
    const wrappedCallback = (event, ...args) => callback(...args);
    ipcRenderer.on('menu-open-terminal', wrappedCallback);
    return () => ipcRenderer.removeListener('menu-open-terminal', wrappedCallback);
  },
  onMenuTakeScreenshot: (callback) => {
    const wrappedCallback = (event, ...args) => callback(...args);
    ipcRenderer.on('menu-take-screenshot', wrappedCallback);
    return () => ipcRenderer.removeListener('menu-take-screenshot', wrappedCallback);
  },
  
  // Remove listeners
  removeAllListeners: (channel) => ipcRenderer.removeAllListeners(channel),
  
  // Platform info
  platform: process.platform
}); 