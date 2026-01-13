const { app, BrowserWindow, Menu, shell, ipcMain, dialog } = require('electron');
const path = require('path');
const fs = require('fs');
const os = require('os');
const isDev = require('electron-is-dev');
const { spawn } = require('child_process');
const screenshot = require('screenshot-desktop');
const http = require('http');

// Safe console logging that only works in development mode
const safeLog = {
  log: (...args) => {
    if (isDev) {
      console.log(...args);
    }
  },
  error: (...args) => {
    if (isDev) {
      console.error(...args);
    }
  },
  warn: (...args) => {
    if (isDev) {
      console.warn(...args);
    }
  }
};

let mainWindow;
let backendProcess = null;
const backendPort = 8000; // Fixed backend port

// Create screenshots directory
function ensureScreenshotDirectory() {
  const mirixDir = path.join(os.homedir(), '.mirix');
  const tmpDir = path.join(mirixDir, 'tmp');
  const imagesDir = path.join(tmpDir, 'images');
  
  if (!fs.existsSync(mirixDir)) {
    fs.mkdirSync(mirixDir, { recursive: true });
  }
  if (!fs.existsSync(tmpDir)) {
    fs.mkdirSync(tmpDir, { recursive: true });
  }
  if (!fs.existsSync(imagesDir)) {
    fs.mkdirSync(imagesDir, { recursive: true });
  }
  
  return imagesDir;
}



function startBackendServer() {
  if (isDev) {
    // In development, assume backend is running separately
    safeLog.log('Development mode: Backend should be running separately');
    return Promise.resolve();
  }

  return new Promise((resolve, reject) => {
    try {
      // In production, start the bundled backend
      const platform = process.platform;
      const executableName = platform === 'win32' ? 'mirix-server.exe' : 'mirix-server';
      const backendPath = path.join(process.resourcesPath, 'backend', executableName);
      
      safeLog.log(`Looking for backend executable at: ${backendPath}`);
      
      // Check if backend executable exists
      if (!fs.existsSync(backendPath)) {
        const error = `Backend executable not found at: ${backendPath}`;
        safeLog.error(error);
        safeLog.error('Available files in backend directory:');
        const backendDir = path.join(process.resourcesPath, 'backend');
        if (fs.existsSync(backendDir)) {
          const files = fs.readdirSync(backendDir);
          files.forEach(file => safeLog.error(`  - ${file}`));
        } else {
          safeLog.error('Backend directory does not exist');
        }
        reject(new Error(error));
        return;
      }
      
      safeLog.log(`Starting backend server on port ${backendPort}: ${backendPath}`);
      
      // Set working directory to the backend directory
      const workingDir = path.join(process.resourcesPath, 'backend');
      
      // Start backend with fixed port
      backendProcess = spawn(backendPath, ['--host', '0.0.0.0', '--port', backendPort.toString()], {
        stdio: ['pipe', 'pipe', 'pipe'],
        detached: false,
        cwd: workingDir,
        env: {
          ...process.env,
          PORT: backendPort.toString(),
          PYTHONPATH: workingDir
        }
      });

      let startupOutput = '';
      let errorOutput = '';
      let healthCheckStarted = false;

      backendProcess.stdout.on('data', (data) => {
        const output = data.toString().trim();
        startupOutput += output;
        safeLog.log(`Backend STDOUT: ${output}`);
        
        // Check if the server has started successfully
        if (output.includes('Uvicorn running on') || 
            output.includes('Application startup complete') ||
            output.includes('Started server process')) {
          
          if (!healthCheckStarted) {
            healthCheckStarted = true;
            safeLog.log('Backend server startup detected, starting health check...');
            setTimeout(() => {
              checkBackendHealth().then(() => {
                safeLog.log('Backend health check passed, resolving startup');
                resolve();
              }).catch((healthError) => {
                safeLog.error('Backend health check failed:', healthError);
                reject(healthError);
              });
            }, 3000);
          }
        }
      });

      backendProcess.stderr.on('data', (data) => {
        const output = data.toString();
        errorOutput += output;
        safeLog.error(`Backend STDERR: ${output}`);
      });

      backendProcess.on('close', (code) => {
        safeLog.log(`Backend process exited with code ${code}`);
        if (code !== 0 && !healthCheckStarted) {
          reject(new Error(`Backend process exited with code ${code}. Error output: ${errorOutput}`));
        }
      });

      backendProcess.on('error', (error) => {
        safeLog.error('Failed to start backend process:', error);
        reject(error);
      });

      // Timeout fallback
      setTimeout(() => {
        if (backendProcess && backendProcess.exitCode === null && !healthCheckStarted) {
          safeLog.warn('Backend startup timeout, trying health check...');
          checkBackendHealth().then(() => {
            safeLog.log('Health check passed despite timeout');
            resolve();
          }).catch((healthError) => {
            safeLog.error('Backend health check failed after timeout:', healthError);
            reject(new Error(`Backend startup timeout: ${healthError.message}`));
          });
        }
      }, 30000); // 30 second timeout

      safeLog.log('Backend server started');
    } catch (error) {
      safeLog.error('Failed to start backend server:', error);
      reject(error);
    }
  });
}

async function checkBackendHealth() {
  const maxRetries = 5;
  const retryDelay = 1000;
  
  for (let i = 0; i < maxRetries; i++) {
    try {
      safeLog.log(`Health check attempt ${i + 1}/${maxRetries}`);
      
      const healthCheckResult = await new Promise((resolve, reject) => {
        const req = http.get(`http://localhost:${backendPort}/health`, { timeout: 5000 }, (res) => {
          let data = '';
          
          res.on('data', chunk => {
            data += chunk;
          });
          
          res.on('end', () => {
            if (res.statusCode === 200) {
              resolve(data);
            } else {
              reject(new Error(`Health check failed with status: ${res.statusCode}`));
            }
          });
        });
        
        req.on('error', (error) => {
          reject(error);
        });
        
        req.on('timeout', () => {
          req.destroy();
          reject(new Error('Health check timeout'));
        });
      });
      
      safeLog.log('Backend health check passed');
      return true;
    } catch (error) {
      safeLog.log(`Health check ${i + 1} failed:`, error.message);
      if (i === maxRetries - 1) {
        throw new Error(`Backend health check failed after ${maxRetries} attempts`);
      }
      await new Promise(resolve => setTimeout(resolve, retryDelay));
    }
  }
}

function stopBackendServer() {
  if (backendProcess) {
    backendProcess.kill();
    backendProcess = null;
    safeLog.log('Backend server stopped');
  }
}

function createWindow() {
  // Ensure screenshot directory exists
  ensureScreenshotDirectory();

  // Create the browser window
  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    webPreferences: {
      nodeIntegration: false,
      contextIsolation: true,
      enableRemoteModule: false,
      preload: path.join(__dirname, 'preload.js')
    },
    icon: path.join(__dirname, 'icon.png'), // Add your app icon here
    titleBarStyle: 'default',
    show: false
  });

  // Load the app
  const startUrl = isDev 
    ? 'http://localhost:3000' 
    : `file://${path.join(__dirname, '../build/index.html')}`;
  
  mainWindow.loadURL(startUrl);

  // Show window when ready to prevent visual flash
  mainWindow.once('ready-to-show', () => {
    mainWindow.show();
    safeLog.log('MainWindow is ready to show');
  });

  // Open DevTools in development
  if (isDev) {
    mainWindow.webContents.openDevTools();
  }

  // Handle window closed
  mainWindow.on('closed', () => {
    mainWindow = null;
  });

  // Handle external links
  mainWindow.webContents.setWindowOpenHandler(({ url }) => {
    shell.openExternal(url);
    return { action: 'deny' };
  });
}

// This method will be called when Electron has finished initialization
app.whenReady().then(async () => {
  safeLog.log('Electron ready - creating window immediately and starting backend in parallel...');
  
  // Create window immediately for better UX
  createWindow();
  
  // Start backend in the background (don't wait for it)
  startBackendInBackground();
  
  // On macOS, re-create window when dock icon is clicked
  app.on('activate', () => {
    if (BrowserWindow.getAllWindows().length === 0) {
      createWindow();
    }
  });
});

// Start backend in background
function startBackendInBackground() {
  safeLog.log('Starting backend server in background...');
  
  startBackendServer().then(() => {
    safeLog.log('Backend initialization complete');
  }).catch((error) => {
    safeLog.error('Backend initialization failed:', error);
    
    // Show error dialog in production
    if (!isDev) {
      dialog.showErrorBox(
        'Backend Startup Error', 
        `Failed to start the backend server: ${error.message}`
      );
    }
  });
}

// Quit when all windows are closed
app.on('window-all-closed', () => {
  stopBackendServer();
  if (process.platform !== 'darwin') {
    app.quit();
  }
});

// Clean up on app quit
app.on('before-quit', () => {
  stopBackendServer();
});

// Security: Prevent new window creation
app.on('web-contents-created', (event, contents) => {
  contents.on('new-window', (event, navigationUrl) => {
    event.preventDefault();
    shell.openExternal(navigationUrl);
  });
});

// IPC handlers for file operations
ipcMain.handle('select-files', async () => {
  const result = await dialog.showOpenDialog(mainWindow, {
    properties: ['openFile', 'multiSelections'],
    filters: [
      { name: 'Images', extensions: ['jpg', 'jpeg', 'png', 'gif', 'bmp', 'webp'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });
  
  return result.filePaths;
});

// IPC handler for selecting save file path
ipcMain.handle('select-save-path', async (event, options = {}) => {
  const result = await dialog.showSaveDialog(mainWindow, {
    title: options.title || 'Save File',
    defaultPath: options.defaultName || 'memories_export.xlsx',
    filters: [
      { name: 'Excel Files', extensions: ['xlsx'] },
      { name: 'CSV Files', extensions: ['csv'] },
      { name: 'All Files', extensions: ['*'] }
    ]
  });
  
  return {
    canceled: result.canceled,
    filePath: result.filePath
  };
});



// IPC handler for taking screenshots
ipcMain.handle('take-screenshot', async (event, options = {}) => {
  try {
    const imagesDir = ensureScreenshotDirectory();
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `screenshot-${timestamp}.png`;
    const filepath = path.join(imagesDir, filename);

    // Take screenshot
    const imgBuffer = await screenshot();
    
    // Save screenshot
    fs.writeFileSync(filepath, imgBuffer);
    
    safeLog.log(`Screenshot saved: ${filepath}`);
    
    return {
      success: true,
      filepath: filepath,
      filename: filename,
      size: imgBuffer.length
    };
  } catch (error) {
    safeLog.error('Failed to take screenshot:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// IPC handler for taking screenshot of specific display
ipcMain.handle('take-screenshot-display', async (event, displayId = 0) => {
  try {
    const imagesDir = ensureScreenshotDirectory();
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
    const filename = `screenshot-display-${displayId}-${timestamp}.png`;
    const filepath = path.join(imagesDir, filename);

    // Get list of displays and take screenshot of specific display
    const displays = await screenshot.listDisplays();
    if (displayId >= displays.length) {
      throw new Error(`Display ${displayId} not found. Available displays: ${displays.length}`);
    }

    const imgBuffer = await screenshot({ screen: displays[displayId].id });
    
    // Save screenshot
    fs.writeFileSync(filepath, imgBuffer);
    
    safeLog.log(`Screenshot of display ${displayId} saved: ${filepath}`);
    
    return {
      success: true,
      filepath: filepath,
      filename: filename,
      size: imgBuffer.length,
      displayId: displayId
    };
  } catch (error) {
    safeLog.error('Failed to take screenshot of display:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// IPC handler for getting available displays
ipcMain.handle('list-displays', async () => {
  try {
    const displays = await screenshot.listDisplays();
    return {
      success: true,
      displays: displays.map((display, index) => ({
        id: display.id,
        index: index,
        name: display.name || `Display ${index + 1}`,
        bounds: display.bounds
      }))
    };
  } catch (error) {
    safeLog.error('Failed to list displays:', error);
    return {
      success: false,
      error: error.message,
      displays: []
    };
  }
});

// IPC handler for cleaning up old screenshots
ipcMain.handle('cleanup-screenshots', async (event, maxAge = 24 * 60 * 60 * 1000) => {
  try {
    const imagesDir = ensureScreenshotDirectory();
    const files = fs.readdirSync(imagesDir);
    const now = Date.now();
    let deletedCount = 0;

    for (const file of files) {
      if (file.startsWith('screenshot-') && file.endsWith('.png')) {
        const filepath = path.join(imagesDir, file);
        const stats = fs.statSync(filepath);
        const age = now - stats.mtime.getTime();
        
        if (age > maxAge) {
          fs.unlinkSync(filepath);
          deletedCount++;
        }
      }
    }

    return {
      success: true,
      deletedCount: deletedCount
    };
  } catch (error) {
    safeLog.error('Failed to cleanup screenshots:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// IPC handler for reading image as base64 (for similarity comparison)
ipcMain.handle('read-image-base64', async (event, filepath) => {
  try {
    if (!fs.existsSync(filepath)) {
      throw new Error(`File does not exist: ${filepath}`);
    }

    const imageBuffer = fs.readFileSync(filepath);
    const base64Data = imageBuffer.toString('base64');
    const mimeType = 'image/png'; // Assuming PNG format for screenshots
    const dataUrl = `data:${mimeType};base64,${base64Data}`;

    return {
      success: true,
      dataUrl: dataUrl,
      base64: base64Data,
      size: imageBuffer.length
    };
  } catch (error) {
    safeLog.error('Failed to read image as base64:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// IPC handler for deleting screenshot files (used when screenshots are too similar)
ipcMain.handle('delete-screenshot', async (event, filepath) => {
  try {
    if (!fs.existsSync(filepath)) {
      safeLog.warn(`Tried to delete non-existent file: ${filepath}`);
      return {
        success: true, // Consider it successful if file doesn't exist
        message: 'File does not exist'
      };
    }

    // Only allow deletion of files in the screenshots directory for security
    const imagesDir = ensureScreenshotDirectory();
    const normalizedFilepath = path.resolve(filepath);
    const normalizedImagesDir = path.resolve(imagesDir);
    
    if (!normalizedFilepath.startsWith(normalizedImagesDir)) {
      throw new Error('Can only delete files in the screenshots directory');
    }

    fs.unlinkSync(filepath);
    safeLog.log(`Screenshot deleted (too similar): ${filepath}`);

    return {
      success: true,
      message: 'File deleted successfully'
    };
  } catch (error) {
    safeLog.error('Failed to delete screenshot:', error);
    return {
      success: false,
      error: error.message
    };
  }
});

// Handle app protocol for deep linking (optional)
if (process.defaultApp) {
  if (process.argv.length >= 2) {
    app.setAsDefaultProtocolClient('mirix', process.execPath, [path.resolve(process.argv[1])]);
  }
} else {
  app.setAsDefaultProtocolClient('mirix');
}

// Create application menu
const createMenu = () => {
  const template = [
    {
      label: 'File',
      submenu: [
        {
          label: 'New Chat',
          accelerator: 'CmdOrCtrl+N',
          click: () => {
            mainWindow.webContents.send('menu-new-chat');
          }
        },
        {
          label: 'Take Screenshot',
          accelerator: 'CmdOrCtrl+Shift+S',
          click: () => {
            mainWindow.webContents.send('menu-take-screenshot');
          }
        },
        {
          label: 'Open Terminal',
          accelerator: 'CmdOrCtrl+T',
          click: () => {
            mainWindow.webContents.send('menu-open-terminal');
          }
        },
        { type: 'separator' },
        {
          label: 'Quit',
          accelerator: process.platform === 'darwin' ? 'Cmd+Q' : 'Ctrl+Q',
          click: () => {
            app.quit();
          }
        }
      ]
    },
    {
      label: 'Edit',
      submenu: [
        { role: 'undo' },
        { role: 'redo' },
        { type: 'separator' },
        { role: 'cut' },
        { role: 'copy' },
        { role: 'paste' },
        { role: 'selectall' }
      ]
    },
    {
      label: 'View',
      submenu: [
        { role: 'reload' },
        { role: 'forceReload' },
        { role: 'toggleDevTools' },
        { type: 'separator' },
        { role: 'resetZoom' },
        { role: 'zoomIn' },
        { role: 'zoomOut' },
        { type: 'separator' },
        { role: 'togglefullscreen' }
      ]
    },
    {
      label: 'Window',
      submenu: [
        { role: 'minimize' },
        { role: 'close' }
      ]
    }
  ];

  if (process.platform === 'darwin') {
    template.unshift({
      label: app.getName(),
      submenu: [
        { role: 'about' },
        { type: 'separator' },
        { role: 'services' },
        { type: 'separator' },
        { role: 'hide' },
        { role: 'hideOthers' },
        { role: 'unhide' },
        { type: 'separator' },
        { role: 'quit' }
      ]
    });
  }

  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
};

app.whenReady().then(() => {
  createMenu();
}); 