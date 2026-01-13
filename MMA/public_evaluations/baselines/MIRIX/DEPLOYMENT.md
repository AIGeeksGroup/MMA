# Mirix Desktop Deployment Guide

This guide covers how to build and distribute the Mirix desktop application with the bundled Python backend.

## Quick Deployment

### Prerequisites
- Node.js 16+
- Python 3.8+
- Git

### One-Command Build
```bash
# Clone and build everything
git clone <your-repo>
cd Mirix
./setup-frontend.bat  # Windows
# or ./setup-frontend.sh  # macOS/Linux

cd frontend
npm run electron-pack
```

## Database Location

The application stores its SQLite database in the user's home directory:

| Platform | Database Path |
|----------|---------------|
| Windows  | `C:\Users\{username}\.mirix\sqlite.db` |
| macOS    | `/Users/{username}/.mirix/sqlite.db` |
| Linux    | `/home/{username}/.mirix/sqlite.db` |

### Database Behavior
- **First Run**: Creates `~/.mirix/` directory and empty database
- **Subsequent Runs**: Uses existing database with all chat history
- **Uninstall**: Database persists (manual deletion required)
- **Upgrade**: Database is preserved across app updates

## Build Outputs

### Windows
```bash
npm run electron-pack
```
**Output**: `dist/Mirix Setup 1.0.0.exe` (~150MB)
- NSIS installer with uninstaller
- Installs to `Program Files\Mirix`
- Creates desktop shortcut
- Adds to Start Menu

### macOS
```bash
npm run electron-pack
```
**Output**: `dist/Mirix-1.0.0.dmg` (~120MB)
- Drag-and-drop installer
- Installs to `/Applications/Mirix.app`
- Code-signed (if certificates configured)

### Linux
```bash
npm run electron-pack
```
**Output**: `dist/Mirix-1.0.0.AppImage` (~130MB)
- Portable executable
- No installation required
- Runs on most Linux distributions

## Distribution Options

### Option 1: Direct Download
1. Build the app: `npm run electron-pack`
2. Upload installers to your website/GitHub releases
3. Users download and install

### Option 2: Auto-Updates (Advanced)
Configure electron-updater in `package.json`:
```json
{
  "build": {
    "publish": {
      "provider": "github",
      "owner": "your-username",
      "repo": "mirix"
    }
  }
}
```

### Option 3: App Stores
- **Microsoft Store**: Package as MSIX
- **Mac App Store**: Requires Apple Developer account
- **Snap Store**: Create snap package

## Code Signing

### Windows
```json
{
  "build": {
    "win": {
      "certificateFile": "path/to/certificate.p12",
      "certificatePassword": "password"
    }
  }
}
```

### macOS
```json
{
  "build": {
    "mac": {
      "identity": "Developer ID Application: Your Name"
    }
  }
}
```

## CI/CD Pipeline

### GitHub Actions Example
```yaml
name: Build and Release
on:
  push:
    tags: ['v*']

jobs:
  build:
    strategy:
      matrix:
        os: [windows-latest, macos-latest, ubuntu-latest]
    
    runs-on: ${{ matrix.os }}
    
    steps:
    - uses: actions/checkout@v3
    - uses: actions/setup-node@v3
      with:
        node-version: 18
    - uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        cd frontend
        npm install
        pip install pyinstaller
    
    - name: Build app
      run: |
        cd frontend
        npm run electron-pack
    
    - name: Upload artifacts
      uses: actions/upload-artifact@v3
      with:
        name: mirix-${{ matrix.os }}
        path: frontend/dist/*
```

## Troubleshooting

### Common Build Issues

1. **PyInstaller not found**
   ```bash
   pip install pyinstaller
   ```

2. **Missing Python dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Large bundle size**
   - Use `--exclude-module` in PyInstaller
   - Remove unused dependencies

4. **Backend startup fails**
   - Check Python executable permissions
   - Verify all dependencies are bundled

### Runtime Issues

1. **Database permission errors**
   - Ensure user has write access to home directory
   - Check antivirus software blocking file creation

2. **Backend connection failed**
   - Check if port is available
   - Verify backend process started successfully

3. **App won't start**
   - Check system requirements (Windows 10+, macOS 10.14+)
   - Verify all dependencies are bundled

## Security Considerations

### Code Signing
- **Required** for macOS distribution
- **Recommended** for Windows (prevents security warnings)
- Use valid certificates from trusted CAs

### Permissions
- App requests minimal permissions
- Database stored in user directory (no admin rights needed)
- Network access for AI API calls

### Updates
- Implement secure update mechanism
- Verify update signatures
- Use HTTPS for update checks

## Performance Optimization

### Bundle Size Reduction
1. **Exclude dev dependencies**
   ```bash
   npm prune --production
   ```

2. **Optimize Python bundle**
   ```bash
   pyinstaller --onefile --strip server.py
   ```

3. **Compress assets**
   - Use webpack optimization
   - Compress images and fonts

### Startup Time
1. **Lazy load modules**
2. **Preload critical components**
3. **Optimize backend startup**

## Monitoring and Analytics

### Error Reporting
```javascript
// In electron.js
const { crashReporter } = require('electron');
crashReporter.start({
  productName: 'Mirix',
  companyName: 'Your Company',
  submitURL: 'https://your-crash-server.com/submit',
  uploadToServer: true
});
```

### Usage Analytics
- Implement privacy-respecting analytics
- Track feature usage and performance
- Monitor crash rates and errors

## Support and Maintenance

### User Support
1. **Documentation**: Comprehensive user guide
2. **FAQ**: Common issues and solutions
3. **Contact**: Support email or forum

### Maintenance
1. **Regular updates**: Security patches and features
2. **Dependency updates**: Keep libraries current
3. **Testing**: Automated and manual testing

## Legal Considerations

### Licensing
- Ensure all dependencies are compatible
- Include license files in distribution
- Respect third-party license terms

### Privacy
- Clearly state data collection practices
- Implement GDPR compliance if applicable
- Provide data export/deletion options

## Release Checklist

- [ ] All tests pass
- [ ] Code signed (if applicable)
- [ ] Documentation updated
- [ ] Release notes prepared
- [ ] Backup/rollback plan ready
- [ ] Support team notified
- [ ] Distribution channels updated 