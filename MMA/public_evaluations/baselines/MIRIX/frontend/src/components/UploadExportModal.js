import React, { useState } from 'react';
import queuedFetch from '../utils/requestQueue';
import './UploadExportModal.css';

function UploadExportModal({ isOpen, onClose, settings }) {
  const [selectedMemoryTypes, setSelectedMemoryTypes] = useState({
    episodic: true,
    semantic: true,
    procedural: true,
    resource: true
  });
  const [exportPath, setExportPath] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [exportStatus, setExportStatus] = useState(null);

  const memoryTypes = [
    { key: 'episodic', label: 'Episodic', icon: '📚', description: 'Personal experiences and events' },
    { key: 'semantic', label: 'Semantic', icon: '🧠', description: 'Facts and general knowledge' },
    { key: 'procedural', label: 'Procedural', icon: '🔧', description: 'Skills and procedures' },
    { key: 'resource', label: 'Resource', icon: '📁', description: 'Files and documents' }
  ];

  const handleMemoryTypeToggle = (type) => {
    setSelectedMemoryTypes(prev => ({
      ...prev,
      [type]: !prev[type]
    }));
  };

  const handleBrowse = async () => {
    if (window.electronAPI && window.electronAPI.selectSavePath) {
      try {
        const result = await window.electronAPI.selectSavePath({
          title: 'Save Memory Export',
          defaultName: 'memories_export.xlsx'
        });
        
        if (!result.canceled && result.filePath) {
          setExportPath(result.filePath);
        }
      } catch (error) {
        console.error('Error opening file dialog:', error);
        alert('Failed to open file browser. Please enter the path manually.');
      }
    } else {
      alert('File browser not available. Please enter the path manually.');
    }
  };

  const handleUpload = () => {
    alert('Upload functionality is not implemented yet (mock feature)');
  };

  const handleExport = async () => {
    if (!exportPath.trim()) {
      alert('Please enter or browse for a file path for export');
      return;
    }

    const selectedTypes = Object.keys(selectedMemoryTypes).filter(
      type => selectedMemoryTypes[type]
    );

    if (selectedTypes.length === 0) {
      alert('Please select at least one memory type to export');
      return;
    }

    setIsLoading(true);
    setExportStatus(null);

    try {
      const response = await queuedFetch(`${settings.serverUrl}/export/memories`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({
          file_path: exportPath,
          memory_types: selectedTypes,
          include_embeddings: false
        }),
      });

      if (response.ok) {
        const result = await response.json();
        setExportStatus({
          success: true,
          message: result.message,
          counts: result.exported_counts,
          total: result.total_exported
        });
      } else {
        const errorData = await response.json();
        setExportStatus({
          success: false,
          message: errorData.detail || 'Export failed'
        });
      }
    } catch (error) {
      console.error('Export error:', error);
      setExportStatus({
        success: false,
        message: `Export failed: ${error.message}`
      });
    } finally {
      setIsLoading(false);
    }
  };

  if (!isOpen) return null;

  return (
    <div className="upload-export-modal-overlay" onClick={onClose}>
      <div className="upload-export-modal" onClick={(e) => e.stopPropagation()}>
        <div className="upload-export-modal-header">
          <h2>📤 Upload & Export</h2>
          <button 
            className="upload-export-modal-close"
            onClick={onClose}
            title="Close"
          >
            ✕
          </button>
        </div>
        
        <div className="upload-export-modal-content">
          <div className="upload-export-modal-description">
            <p>Manage your memory data - upload new data or export existing memories</p>
          </div>

          <div className="memory-types-section">
            <h3>Memory Types</h3>
            <div className="memory-types-grid">
              {memoryTypes.map(type => (
                <div 
                  key={type.key}
                  className={`memory-type-card ${selectedMemoryTypes[type.key] ? 'selected' : ''}`}
                  onClick={() => handleMemoryTypeToggle(type.key)}
                >
                  <div className="memory-type-icon">{type.icon}</div>
                  <div className="memory-type-info">
                    <div className="memory-type-label">{type.label}</div>
                    <div className="memory-type-description">{type.description}</div>
                  </div>
                  <div className="memory-type-checkbox">
                    <input 
                      type="checkbox" 
                      checked={selectedMemoryTypes[type.key]}
                      onChange={() => {}}
                      readOnly
                    />
                  </div>
                </div>
              ))}
            </div>
          </div>

          <div className="actions-section">
            <div className="upload-section">
              <h3>Upload</h3>
              <p>Import memory data from external sources</p>
              <button 
                className="upload-btn"
                onClick={handleUpload}
              >
                📤 Upload Data (Mock)
              </button>
            </div>

            <div className="export-section">
              <h3>Export</h3>
              <p>Export selected memory types to Excel with separate sheets</p>
              
              <div className="export-path-input">
                <label htmlFor="exportPath">Export File Path:</label>
                <div className="path-input-group">
                  <input
                    id="exportPath"
                    type="text"
                    value={exportPath}
                    onChange={(e) => setExportPath(e.target.value)}
                    placeholder="e.g., /Users/username/Desktop/memories_export.xlsx"
                    className="path-input"
                  />
                  <button 
                    type="button"
                    className="browse-btn"
                    onClick={handleBrowse}
                    title="Browse for save location"
                  >
                    📁 Browse
                  </button>
                </div>
              </div>

              <button 
                className="export-btn"
                onClick={handleExport}
                disabled={isLoading}
              >
                {isLoading ? '⏳ Exporting...' : '📥 Export Memories'}
              </button>

              {exportStatus && (
                <div className={`export-status ${exportStatus.success ? 'success' : 'error'}`}>
                  <div className="status-message">{exportStatus.message}</div>
                  {exportStatus.success && exportStatus.counts && (
                    <div className="export-details">
                      <div className="total-exported">Total: {exportStatus.total} memories</div>
                      <div className="counts-breakdown">
                        {Object.entries(exportStatus.counts).map(([type, count]) => (
                          <span key={type} className="count-item">
                            {type}: {count}
                          </span>
                        ))}
                      </div>  
                    </div>
                  )}
                </div>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default UploadExportModal; 