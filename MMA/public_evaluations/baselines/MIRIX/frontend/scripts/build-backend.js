const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

console.log('🔨 Building Python backend...');

// Create backend directory if it doesn't exist
const backendDir = path.join(__dirname, '..', 'backend');
if (!fs.existsSync(backendDir)) {
  fs.mkdirSync(backendDir, { recursive: true });
}

// Copy Python files to backend directory
const sourceDir = path.join(__dirname, '..', '..');
const filesToCopy = [
  'main.py',
  'requirements.txt',
  'mirix/',
  'configs/'
];

// Cross-platform copy function
function copyRecursive(src, dest) {
  if (fs.statSync(src).isDirectory()) {
    if (!fs.existsSync(dest)) {
      fs.mkdirSync(dest, { recursive: true });
    }
    const files = fs.readdirSync(src);
    files.forEach(file => {
      copyRecursive(path.join(src, file), path.join(dest, file));
    });
  } else {
    fs.copyFileSync(src, dest);
  }
}

console.log('📁 Copying Python files...');
filesToCopy.forEach(file => {
  const sourcePath = path.join(sourceDir, file);
  const destPath = path.join(backendDir, file);
  
  if (fs.existsSync(sourcePath)) {
    try {
      copyRecursive(sourcePath, destPath);
      console.log(`✅ Copied ${file}`);
    } catch (error) {
      console.log(`⚠️  Warning: Failed to copy ${file}: ${error.message}`);
    }
  } else {
    console.log(`⚠️  Warning: ${file} not found`);
  }
});

// Check if PyInstaller is available
try {
  execSync('pyinstaller --version', { stdio: 'pipe' });
  console.log('✅ PyInstaller found');
} catch (error) {
  console.log('❌ PyInstaller not found. Installing...');
  try {
    execSync('pip install pyinstaller', { stdio: 'inherit' });
    console.log('✅ PyInstaller installed');
  } catch (installError) {
    console.error('❌ Failed to install PyInstaller:', installError.message);
    console.log('📝 Manual installation required:');
    console.log('   pip install pyinstaller');
    process.exit(1);
  }
}

// Build executable with PyInstaller
console.log('🔨 Building executable with PyInstaller...');
const platform = process.platform;
const executableName = platform === 'win32' ? 'mirix-server.exe' : 'mirix-server';

try {
  // Build PyInstaller command
  const pyinstallerArgs = [
    'pyinstaller',
    '--onefile',
    '--name', 'mirix-server',
    '--distpath', backendDir,
    '--workpath', path.join(backendDir, 'build'),
    '--specpath', path.join(backendDir, 'spec'),
    '--hidden-import', 'mirix',
    '--hidden-import', 'uvicorn',
    '--hidden-import', 'fastapi',
    '--hidden-import', 'pydantic',
    '--hidden-import', 'yaml',
    '--hidden-import', 'numpy',
    '--hidden-import', 'tiktoken',
    '--collect-all', 'mirix'
  ];

  // Add data directories if they exist
  const mirixPath = path.join(backendDir, 'mirix');
  const configsPath = path.join(backendDir, 'configs');
  
  if (fs.existsSync(mirixPath)) {
    if (platform === 'win32') {
      pyinstallerArgs.push('--add-data', `"${mirixPath};mirix"`);
    } else {
      pyinstallerArgs.push('--add-data', `"${mirixPath}:mirix"`);
    }
  }
  
  if (fs.existsSync(configsPath)) {
    if (platform === 'win32') {
      pyinstallerArgs.push('--add-data', `"${configsPath};configs"`);
    } else {
      pyinstallerArgs.push('--add-data', `"${configsPath}:configs"`);
    }
  }

  // Add the main script
  pyinstallerArgs.push(path.join(backendDir, 'main.py'));

  const pyinstallerCmd = pyinstallerArgs.join(' ');
  console.log('Running:', pyinstallerCmd);

  execSync(pyinstallerCmd, { 
    stdio: 'inherit',
    cwd: backendDir
  });

  // Verify executable was created
  const executablePath = path.join(backendDir, executableName);
  if (fs.existsSync(executablePath)) {
    console.log(`✅ Backend executable created: ${executablePath}`);
    
    // Clean up build artifacts
    const buildDir = path.join(backendDir, 'build');
    const specDir = path.join(backendDir, 'spec');
    const specFile = path.join(backendDir, 'mirix-server.spec');
    
    [buildDir, specDir, specFile].forEach(dir => {
      if (fs.existsSync(dir)) {
        fs.rmSync(dir, { recursive: true, force: true });
      }
    });
    
    console.log('🎉 Backend build complete!');
  } else {
    throw new Error('Executable not found after build');
  }

} catch (error) {
  console.error('❌ Failed to build backend:', error.message);
  console.log('');
  console.log('📝 Manual build instructions:');
  console.log('1. Install PyInstaller: pip install pyinstaller');
  console.log('2. Navigate to backend directory');
  console.log('3. Run: pyinstaller --onefile main.py');
  console.log('');
  console.log('⚠️  Continuing without backend executable...');
  console.log('   Users will need to run the Python backend separately.');
} 