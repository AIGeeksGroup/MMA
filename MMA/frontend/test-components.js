#!/usr/bin/env node

const { spawn } = require('child_process');
const { execSync } = require('child_process');
const path = require('path');
const fs = require('fs');
const http = require('http');

console.log('🧪 MMA Component Testing Suite\n');

// Test 1: Check if all build artifacts exist
function testBuildArtifacts() {
  console.log('📁 Test 1: Checking build artifacts...');
  
  const artifacts = [
    'build/index.html',
    'build/static/js',
    'build/static/css',
    'backend/mma-server',
    'backend/configs/mma.yaml',
    'public/electron.js',
    'public/preload.js'
  ];
  
  let allExist = true;
  artifacts.forEach(artifact => {
    if (fs.existsSync(artifact)) {
      console.log(`  ✅ ${artifact}`);
    } else {
      console.log(`  ❌ ${artifact} - MISSING`);
      allExist = false;
    }
  });
  
  console.log(allExist ? '  🎉 All artifacts present\n' : '  💥 Some artifacts missing\n');
  return allExist;
}

// Test 2: Test backend executable
function testBackend() {
  return new Promise((resolve) => {
    console.log('🐍 Test 2: Testing backend executable...');
    
    const backendPath = path.join(__dirname, 'backend', 'mma-server');
    const testPort = 8003;
    
    // Check if executable exists and is executable
    try {
      fs.accessSync(backendPath, fs.constants.F_OK | fs.constants.X_OK);
      console.log('  ✅ Backend executable found and executable');
    } catch (error) {
      console.log('  ❌ Backend executable not found or not executable');
      console.log('  💥 Backend test failed\n');
      resolve(false);
      return;
    }
    
    // Start backend
    const backend = spawn(backendPath, ['--host', '0.0.0.0', '--port', testPort.toString()], {
      stdio: ['pipe', 'pipe', 'pipe'],
      cwd: path.join(__dirname, 'backend')
    });
    
    let output = '';
    let startupDetected = false;
    
    backend.stdout.on('data', (data) => {
      const text = data.toString();
      output += text;
      console.log(`  [BACKEND] ${text.trim()}`);
      
      if (text.includes('Uvicorn running on') || 
          text.includes('Application startup complete')) {
        startupDetected = true;
        
        // Test health endpoint
        setTimeout(() => {
          testHealthEndpoint(testPort).then((healthy) => {
            backend.kill();
            console.log(healthy ? '  🎉 Backend test passed\n' : '  💥 Backend health check failed\n');
            resolve(healthy);
          });
        }, 2000);
      }
    });
    
    backend.stderr.on('data', (data) => {
      console.log(`  [BACKEND ERROR] ${data.toString().trim()}`);
    });
    
    backend.on('close', (code) => {
      if (!startupDetected) {
        console.log(`  💥 Backend exited with code ${code} before startup\n`);
        resolve(false);
      }
    });
    
    // Timeout
    setTimeout(() => {
      if (!startupDetected) {
        backend.kill();
        console.log('  💥 Backend startup timeout\n');
        resolve(false);
      }
    }, 30000);
  });
}

// Test 3: Test health endpoint
function testHealthEndpoint(port) {
  return new Promise((resolve) => {
    console.log(`  🔍 Testing health endpoint on port ${port}...`);
    
    const req = http.get(`http://localhost:${port}/health`, { timeout: 5000 }, (res) => {
      let data = '';
      res.on('data', chunk => data += chunk);
      res.on('end', () => {
        if (res.statusCode === 200) {
          try {
            const healthData = JSON.parse(data);
            console.log(`  ✅ Health endpoint OK: ${JSON.stringify(healthData)}`);
            resolve(true);
          } catch (e) {
            console.log('  ❌ Health endpoint returned invalid JSON');
            resolve(false);
          }
        } else {
          console.log(`  ❌ Health endpoint returned status: ${res.statusCode}`);
          resolve(false);
        }
      });
    });
    
    req.on('error', (error) => {
      console.log(`  ❌ Health endpoint error: ${error.message}`);
      resolve(false);
    });
    
    req.on('timeout', () => {
      req.destroy();
      console.log('  ❌ Health endpoint timeout');
      resolve(false);
    });
  });
}

// Test 4: Test React build
function testReactBuild() {
  console.log('⚛️  Test 3: Testing React build...');
  
  const indexPath = path.join(__dirname, 'build', 'index.html');
  if (!fs.existsSync(indexPath)) {
    console.log('  ❌ build/index.html not found');
    console.log('  💥 React build test failed\n');
    return false;
  }
  
  const indexContent = fs.readFileSync(indexPath, 'utf8');
  
  // Check for expected content
  const checks = [
    { name: 'HTML structure', test: indexContent.includes('<div id="root">') },
    { name: 'CSS bundle', test: indexContent.includes('.css') },
    { name: 'JS bundle', test: indexContent.includes('.js') },
    { name: 'React elements', test: indexContent.includes('React') || indexContent.includes('static/js') }
  ];
  
  let allPassed = true;
  checks.forEach(check => {
    if (check.test) {
      console.log(`  ✅ ${check.name}`);
    } else {
      console.log(`  ❌ ${check.name}`);
      allPassed = false;
    }
  });
  
  console.log(allPassed ? '  🎉 React build test passed\n' : '  💥 React build test failed\n');
  return allPassed;
}

// Test 5: Test Electron configuration
function testElectronConfig() {
  console.log('⚡ Test 4: Testing Electron configuration...');
  
  const electronPath = path.join(__dirname, 'public', 'electron.js');
  const preloadPath = path.join(__dirname, 'public', 'preload.js');
  const packagePath = path.join(__dirname, 'package.json');
  
  let allPassed = true;
  
  // Check electron.js
  if (fs.existsSync(electronPath)) {
    const electronContent = fs.readFileSync(electronPath, 'utf8');
    const electronChecks = [
      { name: 'Port detection function', test: electronContent.includes('findAvailablePort') },
      { name: 'Backend startup function', test: electronContent.includes('startBackendServer') },
      { name: 'Health check function', test: electronContent.includes('checkBackendHealth') },
      { name: 'IPC handlers', test: electronContent.includes('ipcMain.handle') }
    ];
    
    electronChecks.forEach(check => {
      if (check.test) {
        console.log(`  ✅ ${check.name}`);
      } else {
        console.log(`  ❌ ${check.name}`);
        allPassed = false;
      }
    });
  } else {
    console.log('  ❌ electron.js not found');
    allPassed = false;
  }
  
  // Check preload.js
  if (fs.existsSync(preloadPath)) {
    const preloadContent = fs.readFileSync(preloadPath, 'utf8');
    if (preloadContent.includes('getBackendPort')) {
      console.log('  ✅ Preload script with port function');
    } else {
      console.log('  ❌ Preload script missing port function');
      allPassed = false;
    }
  } else {
    console.log('  ❌ preload.js not found');
    allPassed = false;
  }
  
  // Check package.json
  if (fs.existsSync(packagePath)) {
    const packageContent = JSON.parse(fs.readFileSync(packagePath, 'utf8'));
    const packageChecks = [
      { name: 'electron-pack script', test: packageContent.scripts && packageContent.scripts['electron-pack'] },
      { name: 'electron-builder config', test: packageContent.build },
      { name: 'extraResources config', test: packageContent.build && packageContent.build.extraResources }
    ];
    
    packageChecks.forEach(check => {
      if (check.test) {
        console.log(`  ✅ ${check.name}`);
      } else {
        console.log(`  ❌ ${check.name}`);
        allPassed = false;
      }
    });
  }
  
  console.log(allPassed ? '  🎉 Electron config test passed\n' : '  💥 Electron config test failed\n');
  return allPassed;
}

// Test 6: Check packaged app (if exists)
function testPackagedApp() {
  console.log('📦 Test 5: Checking packaged app...');
  
  const distPath = path.join(__dirname, 'dist');
  if (!fs.existsSync(distPath)) {
    console.log('  ℹ️  No dist folder found (run npm run electron-pack to create)');
    console.log('  ⏭️  Skipping packaged app test\n');
    return true;
  }
  
  const files = fs.readdirSync(distPath);
  const dmgFiles = files.filter(f => f.endsWith('.dmg'));
  
  if (dmgFiles.length > 0) {
    console.log(`  ✅ Found packaged app: ${dmgFiles[0]}`);
    const dmgStats = fs.statSync(path.join(distPath, dmgFiles[0]));
    console.log(`  📊 Size: ${(dmgStats.size / 1024 / 1024).toFixed(2)} MB`);
    console.log('  🎉 Packaged app test passed\n');
    return true;
  } else {
    console.log('  ❌ No .dmg file found in dist folder');
    console.log('  💥 Packaged app test failed\n');
    return false;
  }
}

// Run all tests
async function runAllTests() {
  console.log('Starting comprehensive component testing...\n');
  
  const results = {
    buildArtifacts: testBuildArtifacts(),
    reactBuild: testReactBuild(),
    electronConfig: testElectronConfig(),
    packagedApp: testPackagedApp(),
    backend: await testBackend()
  };
  
  // Summary
  console.log('📊 Test Summary:');
  console.log('================');
  Object.entries(results).forEach(([test, passed]) => {
    console.log(`${passed ? '✅' : '❌'} ${test}: ${passed ? 'PASSED' : 'FAILED'}`);
  });
  
  const allPassed = Object.values(results).every(r => r);
  console.log(`\n${allPassed ? '🎉 All tests passed!' : '💥 Some tests failed!'}`);
  console.log('\n📋 Next Steps:');
  
  if (allPassed) {
    console.log('✅ Your MMA app is ready!');
    console.log('🚀 You can run the packaged app from dist/ folder');
    console.log('🔧 Or test in development with: npm run electron-dev');
  } else {
    console.log('🔧 Fix the failing components and run tests again');
    console.log('📖 Check the logs above for specific issues');
  }
  
  process.exit(allPassed ? 0 : 1);
}

// Run the tests
runAllTests().catch(error => {
  console.error('💥 Test suite failed:', error);
  process.exit(1);
}); 