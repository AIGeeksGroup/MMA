import React from 'react';
import './ClearChatModal.css';

const ClearChatModal = ({ isOpen, onClose, onClearLocal, onClearPermanent, isClearing }) => {
  if (!isOpen) return null;

  return (
    <div className="modal-overlay" onClick={onClose}>
      <div className="modal-content" onClick={(e) => e.stopPropagation()}>
        <div className="modal-header">
          <h3>Clear Chat</h3>
          <button className="modal-close" onClick={onClose} disabled={isClearing}>
            ×
          </button>
        </div>
        
        <div className="modal-body">
          <p>Choose how you want to clear the chat:</p>
          
          <div className="clear-options">
            <div className="clear-option">
              <div className="clear-option-header">
                <h4>🗑️ Clear Current View</h4>
                <span className="option-type">Local Only</span>
              </div>
              <p>
                Clear the conversation display in this window. This only affects what you see here - 
                your conversation history with the agent remains intact and memories are preserved.
              </p>
              <button 
                className="clear-local-btn"
                onClick={onClearLocal}
                disabled={isClearing}
              >
                Clear View Only
              </button>
            </div>
            
            <div className="clear-option permanent">
              <div className="clear-option-header">
                <h4>⚠️ Clear All Conversation History</h4>
                <span className="option-type permanent">Permanent</span>
              </div>
              <p>
                <strong>Permanently delete</strong> all conversation history between you and the chat agent. 
                This cannot be undone. Your memories (episodic, semantic, etc.) will be preserved, 
                but the chat history will be lost forever.
              </p>
              <div className="warning-note">
                <span className="warning-icon">⚠️</span>
                This action is permanent and cannot be undone!
              </div>
              <button 
                className="clear-permanent-btn"
                onClick={onClearPermanent}
                disabled={isClearing}
              >
                {isClearing ? 'Clearing...' : 'Permanently Clear All'}
              </button>
            </div>
          </div>
        </div>
        
        <div className="modal-footer">
          <button className="cancel-btn" onClick={onClose} disabled={isClearing}>
            Cancel
          </button>
        </div>
      </div>
    </div>
  );
};

export default ClearChatModal; 