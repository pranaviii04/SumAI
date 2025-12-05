// ====================================
// SummAI - Frontend Application Logic
// ====================================

// API Configuration
const API_BASE_URL = window.location.origin;

// DOM Elements
const inputText = document.getElementById('inputText');
const outputContent = document.getElementById('outputContent');
const summarizeBtn = document.getElementById('summarizeBtn');
const clearBtn = document.getElementById('clearBtn');
const uploadBtn = document.getElementById('uploadBtn');
const fileInput = document.getElementById('fileInput');
const copyBtn = document.getElementById('copyBtn');
const statusBtn = document.getElementById('statusBtn');
const charCount = document.getElementById('charCount');
const metricsPanel = document.getElementById('metricsPanel');
const toast = document.getElementById('toast');
const toastMessage = document.getElementById('toastMessage');
const statusModal = document.getElementById('statusModal');
const modalOverlay = document.getElementById('modalOverlay');
const modalClose = document.getElementById('modalClose');
const statusContent = document.getElementById('statusContent');
const themeToggle = document.getElementById('themeToggle');
const loginForm = document.getElementById('loginForm');
const signupForm = document.getElementById('signupForm');



// ====================================
// Theme Management
// ====================================

// Initialize theme from localStorage or prefer-color-scheme
function initTheme() {
    const savedTheme = localStorage.getItem('theme') || 
        (window.matchMedia('(prefers-color-scheme: dark)').matches ? 'dark' : 'light');
    
    document.documentElement.setAttribute('data-theme', savedTheme);
    updateThemeUI(savedTheme);
    return savedTheme;
}

// Toggle between light and dark theme
function toggleTheme() {
    const currentTheme = document.documentElement.getAttribute('data-theme');
    const newTheme = currentTheme === 'dark' ? 'light' : 'dark';
    
    document.documentElement.setAttribute('data-theme', newTheme);
    localStorage.setItem('theme', newTheme);
    updateThemeUI(newTheme);
    return newTheme;
}

// Update UI based on theme
function updateThemeUI(theme) {
    const themeToggle = document.getElementById('themeToggle');
    if (!themeToggle) return;

    const moonIcon = themeToggle.querySelector('.moon-icon');
    const sunIcon = themeToggle.querySelector('.sun-icon');
    const themeText = themeToggle.querySelector('.theme-text');

    if (theme === 'dark') {
        if (moonIcon) moonIcon.style.display = 'none';
        if (sunIcon) sunIcon.style.display = 'block';
        if (themeText) themeText.textContent = 'Light Mode';
    } else {
        if (moonIcon) moonIcon.style.display = 'block';
        if (sunIcon) sunIcon.style.display = 'none';
        if (themeText) themeText.textContent = 'Dark Mode';
    }
}

// Initialize theme when page loads
document.addEventListener('DOMContentLoaded', () => {
    // Only initialize theme if we're not on the login page (which handles its own theme)
    if (!window.location.pathname.includes('login') && !window.location.pathname.includes('signup')) {
        initTheme();
        
        // Add click event to theme toggle if it exists
        const themeToggle = document.getElementById('themeToggle');
        if (themeToggle) {
            themeToggle.addEventListener('click', toggleTheme);
        }
    }
});

// ====================================
// Event Listeners
// ====================================



// Only attach main app listeners if elements exist (i.e., we are on the main page)
if (inputText) {
    // Character count
    inputText.addEventListener('input', () => {
        const count = inputText.value.length;
        charCount.textContent = count.toLocaleString();
    });

    // Clear button
    clearBtn.addEventListener('click', () => {
        inputText.value = '';
        charCount.textContent = '0';
        outputContent.innerHTML = `
            <div class="empty-state">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M9.663 17h4.673M12 3v1m6.364 1.636l-.707.707M21 12h-1M4 12H3m3.343-5.657l-.707-.707m2.828 9.9a5 5 0 117.072 0l-.548.547A3.374 3.374 0 0014 18.469V19a2 2 0 11-4 0v-.531c0-.895-.356-1.754-.988-2.386l-.548-.547z" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
                <p>Your summary will appear here</p>
            </div>
        `;
        metricsPanel.style.display = 'none';
        copyBtn.style.display = 'none';
        showToast('Cleared successfully');
    });

    // Upload button
    uploadBtn.addEventListener('click', () => {
        fileInput.click();
    });

    // File input change
    fileInput.addEventListener('change', handleFileUpload);

    // Copy button
    copyBtn.addEventListener('click', async () => {
        const summaryText = outputContent.querySelector('.summary-text')?.textContent;
        if (summaryText) {
            try {
                await navigator.clipboard.writeText(summaryText);
                showToast('Copied to clipboard!');
            } catch (err) {
                showToast('Failed to copy');
            }
        }
    });

    // Summarize button
    summarizeBtn.addEventListener('click', handleSummarize);

    // Status button
    statusBtn.addEventListener('click', showStatus);

    // Modal close
    modalOverlay.addEventListener('click', closeModal);
    modalClose.addEventListener('click', closeModal);
}

// ====================================
// Main Functions
// ====================================

async function handleFileUpload(event) {
    const file = event.target.files[0];
    if (!file) return;

    // Show loading state in input
    const originalPlaceholder = inputText.placeholder;
    inputText.placeholder = "Extracting text from file...";
    inputText.disabled = true;

    const formData = new FormData();
    formData.append('file', file);

    try {
        const response = await fetch(`${API_BASE_URL}/extract-text`, {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Failed to extract text');
        }

        const data = await response.json();
        inputText.value = data.text;
        charCount.textContent = data.text.length.toLocaleString();
        showToast(`Loaded text from ${data.filename}`);

    } catch (error) {
        console.error('Error:', error);
        showToast(error.message);
    } finally {
        inputText.disabled = false;
        inputText.placeholder = originalPlaceholder;
        fileInput.value = ''; // Reset input
    }
}

async function handleSummarize() {
    const text = inputText.value.trim();

    if (!text) {
        showToast('Please enter some text to summarize');
        return;
    }

    // Disable button and show loading
    summarizeBtn.disabled = true;
    summarizeBtn.innerHTML = `
        <div class="loading-spinner" style="width: 20px; height: 20px; border-width: 2px; margin: 0;"></div>
        <span>Generating...</span>
    `;

    outputContent.innerHTML = `
        <div class="empty-state">
            <div class="loading-spinner"></div>
            <p>Generating your summary...</p>
        </div>
    `;

    try {
        const response = await fetch(`${API_BASE_URL}/summarize`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                text: text,
                use_ml: true // Always use ML
            })
        });

        if (!response.ok) {
            const error = await response.json();
            throw new Error(error.detail || 'Summarization failed');
        }

        const data = await response.json();
        displaySummary(data);

    } catch (error) {
        console.error('Error:', error);
        outputContent.innerHTML = `
            <div class="empty-state">
                <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                    <path d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" stroke="currentColor" stroke-width="2" stroke-linecap="round"/>
                </svg>
                <p style="color: var(--error);">${error.message}</p>
            </div>
        `;
        showToast('Summarization failed');
    } finally {
        // Re-enable button
        summarizeBtn.disabled = false;
        summarizeBtn.innerHTML = `
            <svg viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg">
                <path d="M13 10V3L4 14h7v7l9-11h-7z" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"/>
            </svg>
            <span>Generate Summary</span>
        `;
    }
}

function displaySummary(data) {
    // Display summary
    outputContent.innerHTML = `<div class="summary-text">${data.summary}</div>`;

    // Show copy button
    copyBtn.style.display = 'flex';

    // Display metrics
    document.getElementById('originalLength').textContent = `${data.original_length.toLocaleString()} chars`;
    document.getElementById('summaryLength').textContent = `${data.summary_length.toLocaleString()} chars`;

    const compressionRatio = ((data.summary_length / data.original_length) * 100).toFixed(1);
    document.getElementById('compressionRatio').textContent = `${compressionRatio}%`;
    document.getElementById('modelUsed').textContent = data.model_used;

    metricsPanel.style.display = 'block';

    showToast('Summary generated successfully!');
}

async function showStatus() {
    statusModal.classList.add('active');
    statusContent.innerHTML = '<div class="loading-spinner"></div>';

    try {
        const response = await fetch(`${API_BASE_URL}/status`);
        const data = await response.json();

        statusContent.innerHTML = `
            <div style="display: flex; flex-direction: column; gap: 1rem;">
                <div class="metric-card">
                    <div class="metric-icon">${data.status === 'healthy' ? '✅' : '❌'}</div>
                    <div class="metric-content">
                        <div class="metric-label">API Status</div>
                        <div class="metric-value">${data.status}</div>
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-icon">${data.ml_available ? '🤖' : '⚠️'}</div>
                    <div class="metric-content">
                        <div class="metric-label">ML Available</div>
                        <div class="metric-value">${data.ml_available ? 'Yes' : 'No'}</div>
                    </div>
                </div>
                
                <div class="metric-card">
                    <div class="metric-icon">${data.ml_model_loaded ? '✅' : '❌'}</div>
                    <div class="metric-content">
                        <div class="metric-label">ML Model Loaded</div>
                        <div class="metric-value">${data.ml_model_loaded ? 'Yes' : 'No'}</div>
                    </div>
                </div>
            </div>
        `;

    } catch (error) {
        statusContent.innerHTML = `
            <div style="text-align: center; color: var(--error);">
                <p>Failed to load status</p>
                <p style="font-size: 0.875rem; margin-top: 0.5rem;">${error.message}</p>
            </div>
        `;
    }
}

function closeModal() {
    statusModal.classList.remove('active');
}

function showToast(message) {
    toastMessage.textContent = message;
    toast.classList.add('show');

    setTimeout(() => {
        toast.classList.remove('show');
    }, 3000);
}

// ====================================
// Initialization
// ====================================

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + Enter to summarize
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter') {
        e.preventDefault();
        if (summarizeBtn && !summarizeBtn.disabled) {
            handleSummarize();
        }
    }

    // Escape to close modal
    if (e.key === 'Escape') {
        closeModal();
    }
});
