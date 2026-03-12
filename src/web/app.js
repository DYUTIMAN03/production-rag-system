/**
 * Production RAG System — Frontend Application
 * Handles query submission (with streaming), response display,
 * metrics, document ingestion, user feedback, markdown rendering,
 * toast notifications, and keyboard shortcuts.
 */

const API_BASE = '';

// ===== DOM Elements =====
const queryForm = document.getElementById('queryForm');
const queryInput = document.getElementById('queryInput');
const queryBtn = document.getElementById('queryBtn');
const responseSection = document.getElementById('responseSection');
const loadingSection = document.getElementById('loadingSection');
const answerContent = document.getElementById('answerContent');
const confidenceScore = document.getElementById('confidenceScore');
const confidenceFill = document.getElementById('confidenceFill');
const citationsSection = document.getElementById('citationsSection');
const citationsList = document.getElementById('citationsList');
const statusDot = document.querySelector('.status-dot');
const statusText = document.getElementById('statusText');
const metricsBtn = document.getElementById('metricsBtn');
const metricsModal = document.getElementById('metricsModal');
const closeModal = document.getElementById('closeModal');
const metricsGrid = document.getElementById('metricsGrid');
const ingestForm = document.getElementById('ingestForm');
const ingestResult = document.getElementById('ingestResult');
const copyBtn = document.getElementById('copyBtn');
const copyBtnText = document.getElementById('copyBtnText');
const suggestionChips = document.getElementById('suggestionChips');
const toastContainer = document.getElementById('toastContainer');

// Loading step elements
const step1 = document.getElementById('step1');
const step2 = document.getElementById('step2');
const step3 = document.getElementById('step3');

// Metric display elements
const metricLatency = document.getElementById('metricLatency');
const metricTokens = document.getElementById('metricTokens');
const metricChunks = document.getElementById('metricChunks');
const metricGrounded = document.getElementById('metricGrounded');

// Store last response for feedback
let lastResponseData = null;

// Configure marked.js
if (typeof marked !== 'undefined') {
    marked.setOptions({
        breaks: true,
        gfm: true,
    });
}

// ===== Toast Notification System =====
function showToast(message, type = 'info', duration = 3000) {
    const icons = { success: '✓', error: '✗', info: 'ℹ' };
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.innerHTML = `<span class="toast-icon">${icons[type] || icons.info}</span> ${escapeHtml(message)}`;
    toastContainer.appendChild(toast);

    setTimeout(() => {
        toast.classList.add('toast-exit');
        setTimeout(() => toast.remove(), 300);
    }, duration);
}

// ===== Health Check =====
async function checkHealth() {
    try {
        const res = await fetch(`${API_BASE}/api/health`);
        const data = await res.json();
        statusDot.classList.add('active');
        statusText.textContent = `${data.vector_store_count} chunks indexed`;
    } catch {
        statusDot.classList.remove('active');
        statusText.textContent = 'Disconnected';
    }
}

// ===== Suggestion Chips =====
document.querySelectorAll('.suggestion-chip').forEach(chip => {
    chip.addEventListener('click', () => {
        const query = chip.dataset.query;
        queryInput.value = query;
        queryForm.dispatchEvent(new Event('submit', { cancelable: true }));
    });
});

function hideSuggestions() {
    if (suggestionChips) {
        suggestionChips.classList.add('hidden');
    }
}

// ===== Copy to Clipboard =====
if (copyBtn) {
    copyBtn.addEventListener('click', async () => {
        const text = lastResponseData?.answer || answerContent.textContent;
        if (!text) return;

        try {
            await navigator.clipboard.writeText(text);
            copyBtn.classList.add('copied');
            copyBtnText.textContent = 'Copied!';
            showToast('Answer copied to clipboard', 'success');

            setTimeout(() => {
                copyBtn.classList.remove('copied');
                copyBtnText.textContent = 'Copy';
            }, 2000);
        } catch {
            showToast('Failed to copy', 'error');
        }
    });
}

// ===== Keyboard Shortcuts =====
document.addEventListener('keydown', (e) => {
    // Ctrl/Cmd + K → focus search
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
        e.preventDefault();
        queryInput.focus();
    }

    // Escape → close modal
    if (e.key === 'Escape') {
        metricsModal.classList.add('hidden');
    }
});

// ===== Query Submission (Streaming) =====
queryForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const question = queryInput.value.trim();
    if (!question) return;

    // Hide suggestions on first query
    hideSuggestions();

    // Show loading, hide response
    responseSection.classList.add('hidden');
    loadingSection.classList.remove('hidden');
    queryBtn.disabled = true;

    animateLoadingSteps();

    try {
        const res = await fetch(`${API_BASE}/api/query/stream`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ question }),
        });

        if (!res.ok) {
            const err = await res.json();
            throw new Error(err.detail || 'Query failed');
        }

        // Process SSE stream
        loadingSection.classList.add('hidden');
        responseSection.classList.remove('hidden');
        answerContent.innerHTML = '';
        answerContent.className = 'answer-content';

        const reader = res.body.getReader();
        const decoder = new TextDecoder();
        let buffer = '';
        let streamedText = '';
        let metadata = null;

        // Add streaming cursor
        const cursor = document.createElement('span');
        cursor.className = 'streaming-cursor';
        cursor.textContent = '▌';
        answerContent.appendChild(cursor);

        while (true) {
            const { done, value } = await reader.read();
            if (done) break;

            buffer += decoder.decode(value, { stream: true });
            const lines = buffer.split('\n');
            buffer = lines.pop(); // Keep incomplete line in buffer

            for (const line of lines) {
                if (line.startsWith('data: ')) {
                    try {
                        const event = JSON.parse(line.slice(6));

                        if (event.type === 'metadata') {
                            metadata = event;
                            displayStreamMetadata(event);
                        } else if (event.type === 'token') {
                            streamedText += event.content;
                            // Render markdown progressively
                            renderStreamedMarkdown(streamedText, cursor);
                        } else if (event.type === 'done') {
                            // Show per-request metrics
                            if (event.total_latency_ms) {
                                metricLatency.textContent = `${Math.round(event.total_latency_ms)}ms`;
                            }
                        } else if (event.type === 'error') {
                            throw new Error(event.message);
                        }
                    } catch (parseErr) {
                        if (parseErr.message !== 'Unexpected end of JSON input') {
                            console.warn('SSE parse error:', parseErr);
                        }
                    }
                }
            }
        }

        // Remove cursor after streaming completes
        cursor.remove();

        // Final markdown render
        if (typeof marked !== 'undefined') {
            answerContent.innerHTML = marked.parse(streamedText);
        } else {
            answerContent.textContent = streamedText;
        }

        // Store for feedback
        lastResponseData = {
            question,
            answer: streamedText,
            confidence_score: metadata?.confidence_score || 0,
            is_grounded: metadata?.is_grounded || false,
            chunks_used: metadata?.chunks_used || 0,
            citations: metadata?.citations || [],
        };

        // Show feedback buttons
        showFeedbackButtons();

    } catch (err) {
        loadingSection.classList.add('hidden');
        displayError(err.message);
        showToast(err.message, 'error');
    } finally {
        queryBtn.disabled = false;
    }
});

// ===== Render Streamed Markdown =====
function renderStreamedMarkdown(text, cursor) {
    if (typeof marked !== 'undefined') {
        // Render current text as markdown, append cursor
        const html = marked.parse(text);
        answerContent.innerHTML = html;
        answerContent.appendChild(cursor);
    } else {
        answerContent.textContent = text;
        answerContent.appendChild(cursor);
    }
}

function animateLoadingSteps() {
    [step1, step2, step3].forEach(s => {
        s.classList.remove('active', 'done');
    });

    step1.classList.add('active');

    setTimeout(() => {
        step1.classList.remove('active');
        step1.classList.add('done');
        step2.classList.add('active');
    }, 800);

    setTimeout(() => {
        step2.classList.remove('active');
        step2.classList.add('done');
        step3.classList.add('active');
    }, 1600);
}

// ===== Display Streaming Metadata =====
function displayStreamMetadata(metadata) {
    // Confidence
    const score = Math.round(metadata.confidence_score * 100);
    confidenceScore.textContent = `${score}%`;
    confidenceFill.style.width = `${Math.min(score, 100)}%`;
    confidenceFill.className = 'confidence-fill';
    if (score < 30) confidenceFill.classList.add('low');
    else if (score < 70) confidenceFill.classList.add('medium');
    else confidenceFill.classList.add('high');

    // Grounded status
    if (!metadata.is_grounded) {
        answerContent.classList.add('ungrounded');
    }

    // Citations
    if (metadata.citations && metadata.citations.length > 0) {
        citationsSection.classList.remove('hidden');
        citationsList.innerHTML = metadata.citations.map((c) => `
            <div class="citation-card" onclick="this.classList.toggle('expanded')">
                <div class="citation-header">
                    <span class="citation-source">
                        ${escapeHtml(c.source)}${c.page_number ? ` · Page ${c.page_number}` : ''}
                        ${c.section_heading ? ` · ${escapeHtml(c.section_heading)}` : ''}
                    </span>
                    <span class="citation-score">score: ${c.relevance_score.toFixed(3)}</span>
                </div>
                <div class="citation-text">${escapeHtml(c.chunk_text)}</div>
            </div>
        `).join('');
    } else {
        citationsSection.classList.add('hidden');
    }

    // Rewritten query indicator
    if (metadata.rewritten_query && metadata.rewritten_query !== lastResponseData?.question) {
        const queryHint = document.getElementById('queryRewriteHint');
        if (queryHint) {
            queryHint.textContent = `🔄 Rewritten: "${metadata.rewritten_query}"`;
            queryHint.classList.remove('hidden');
        }
    }

    // Metrics
    metricChunks.textContent = metadata.chunks_used || '—';
    metricGrounded.textContent = metadata.is_grounded ? '✓ Yes' : '✗ No';
    metricGrounded.style.color = metadata.is_grounded
        ? 'var(--accent-green)'
        : 'var(--accent-amber)';

    checkHealth();

    // Update side panels
    if (typeof highlightCitedDocuments === 'function' && metadata.citations) {
        highlightCitedDocuments(metadata.citations);
    }
    if (typeof buildNeuralGraph === 'function' && metadata.citations) {
        buildNeuralGraph(metadata.citations);
    }
}

// ===== Display Error =====
function displayError(message) {
    responseSection.classList.remove('hidden');
    answerContent.textContent = `Error: ${message}`;
    answerContent.className = 'answer-content ungrounded';
    citationsSection.classList.add('hidden');
    confidenceScore.textContent = '0%';
    confidenceFill.style.width = '0%';
    confidenceFill.className = 'confidence-fill low';
}

// ===== Feedback Buttons =====
function showFeedbackButtons() {
    const existingFeedback = document.getElementById('feedbackSection');
    if (existingFeedback) existingFeedback.remove();

    const feedbackHtml = `
        <div id="feedbackSection" class="feedback-section">
            <span class="feedback-label">Was this helpful?</span>
            <button class="feedback-btn positive" onclick="submitFeedback(true)" title="Thumbs up">👍</button>
            <button class="feedback-btn negative" onclick="submitFeedback(false)" title="Thumbs down">👎</button>
            <span id="feedbackStatus" class="feedback-status"></span>
        </div>
    `;
    answerContent.insertAdjacentHTML('afterend', feedbackHtml);
}

async function submitFeedback(isPositive) {
    if (!lastResponseData) return;

    const feedbackStatus = document.getElementById('feedbackStatus');
    const buttons = document.querySelectorAll('.feedback-btn');

    // Disable buttons
    buttons.forEach(btn => btn.disabled = true);

    try {
        const res = await fetch(`${API_BASE}/api/feedback`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                question: lastResponseData.question,
                answer: lastResponseData.answer,
                is_positive: isPositive,
                confidence_score: lastResponseData.confidence_score,
                chunks_used: lastResponseData.chunks_used,
                is_grounded: lastResponseData.is_grounded,
                citations: lastResponseData.citations,
            }),
        });

        const data = await res.json();
        feedbackStatus.textContent = data.message;
        feedbackStatus.classList.add('visible');

        // Highlight selected button
        buttons.forEach(btn => btn.classList.remove('selected'));
        buttons[isPositive ? 0 : 1].classList.add('selected');

        // Show toast
        showToast(isPositive ? 'Thanks for the positive feedback!' : 'Feedback recorded. We\'ll improve!', 'success');

    } catch (err) {
        feedbackStatus.textContent = '⚠ Failed to save feedback';
        feedbackStatus.classList.add('visible');
        buttons.forEach(btn => btn.disabled = false);
        showToast('Failed to save feedback', 'error');
    }
}

// Make submitFeedback globally accessible
window.submitFeedback = submitFeedback;

// ===== Metrics Modal =====
metricsBtn.addEventListener('click', async () => {
    metricsModal.classList.remove('hidden');
    await loadMetrics();
});

closeModal.addEventListener('click', () => {
    metricsModal.classList.add('hidden');
});

document.querySelector('.modal-overlay')?.addEventListener('click', () => {
    metricsModal.classList.add('hidden');
});

async function loadMetrics() {
    try {
        const res = await fetch(`${API_BASE}/api/metrics`);
        const data = await res.json();

        metricsGrid.innerHTML = `
            <div class="metrics-card">
                <h3>Total Requests</h3>
                <div class="value">${data.total_requests}</div>
                <div class="sub-value">${data.error_count} errors (${(data.failure_rate * 100).toFixed(1)}%)</div>
            </div>
            <div class="metrics-card">
                <h3>Latency P50</h3>
                <div class="value">${data.latency.total.p50_ms.toFixed(0)}ms</div>
                <div class="sub-value">P95: ${data.latency.total.p95_ms.toFixed(0)}ms</div>
            </div>
            <div class="metrics-card">
                <h3>Retrieval P50</h3>
                <div class="value">${data.latency.retrieval.p50_ms.toFixed(0)}ms</div>
                <div class="sub-value">P95: ${data.latency.retrieval.p95_ms.toFixed(0)}ms</div>
            </div>
            <div class="metrics-card">
                <h3>Generation P50</h3>
                <div class="value">${data.latency.generation.p50_ms.toFixed(0)}ms</div>
                <div class="sub-value">P95: ${data.latency.generation.p95_ms.toFixed(0)}ms</div>
            </div>
            <div class="metrics-card">
                <h3>Citation Coverage</h3>
                <div class="value">${(data.quality.citation_coverage * 100).toFixed(1)}%</div>
                <div class="sub-value">${data.quality.grounded_responses} grounded / ${data.quality.ungrounded_responses} refused</div>
            </div>
            <div class="metrics-card">
                <h3>Cost</h3>
                <div class="value">$${data.cost.total_cost_usd.toFixed(4)}</div>
                <div class="sub-value">${data.cost.total_input_tokens + data.cost.total_output_tokens} total tokens</div>
            </div>
            <div class="metrics-card">
                <h3>Reranker Max Score</h3>
                <div class="value">${data.reranker_scores.max_score_distribution.mean.toFixed(3)}</div>
                <div class="sub-value">Range: ${data.reranker_scores.max_score_distribution.min.toFixed(3)} — ${data.reranker_scores.max_score_distribution.max.toFixed(3)}</div>
            </div>
            <div class="metrics-card">
                <h3>Failure Rate</h3>
                <div class="value">${(data.failure_rate * 100).toFixed(1)}%</div>
                <div class="sub-value">${data.successful_requests} successful</div>
            </div>
        `;
    } catch {
        metricsGrid.innerHTML = '<p class="loading-text">Failed to load metrics. Is the server running?</p>';
    }
}

// ===== Document Ingestion =====
ingestForm.addEventListener('submit', async (e) => {
    e.preventDefault();
    const path = document.getElementById('ingestPath').value.trim();
    if (!path) return;

    ingestResult.classList.remove('hidden', 'success', 'error');
    ingestResult.textContent = 'Ingesting...';

    try {
        const res = await fetch(`${API_BASE}/api/ingest`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ path }),
        });

        const data = await res.json();

        if (!res.ok) {
            throw new Error(data.detail || 'Ingestion failed');
        }

        ingestResult.classList.add('success');
        ingestResult.textContent = `✓ ${data.message}`;
        showToast(data.message, 'success');
        checkHealth();
    } catch (err) {
        ingestResult.classList.add('error');
        ingestResult.textContent = `✗ ${err.message}`;
        showToast(`Ingestion failed: ${err.message}`, 'error');
    }
});

// ===== File Upload =====
const uploadArea = document.getElementById('uploadArea');
const fileInput = document.getElementById('fileInput');
const uploadedFiles = document.getElementById('uploadedFiles');
const uploadResult = document.getElementById('uploadResult');

if (uploadArea) {
    // Drag and drop
    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        handleFiles(e.dataTransfer.files);
    });

    // File input change
    if (fileInput) {
        fileInput.addEventListener('change', () => {
            handleFiles(fileInput.files);
        });
    }
}

async function handleFiles(files) {
    if (!files || files.length === 0) return;

    const formData = new FormData();
    const names = [];
    for (const file of files) {
        formData.append('files', file);
        names.push(file.name);
    }

    // Show file names
    uploadedFiles.classList.remove('hidden');
    uploadedFiles.innerHTML = names.map(n => `<span style="font-size:0.8rem;color:var(--text-secondary);">📄 ${escapeHtml(n)}</span>`).join('');

    uploadResult.classList.remove('hidden', 'success', 'error');
    uploadResult.textContent = 'Uploading & ingesting...';

    try {
        const res = await fetch(`${API_BASE}/api/upload`, {
            method: 'POST',
            body: formData,
        });
        const data = await res.json();
        if (!res.ok) throw new Error(data.detail || 'Upload failed');

        uploadResult.classList.add('success');
        uploadResult.textContent = `✓ ${data.message}`;
        showToast(data.message, 'success');
        checkHealth();
    } catch (err) {
        uploadResult.classList.add('error');
        uploadResult.textContent = `✗ ${err.message}`;
        showToast(`Upload failed: ${err.message}`, 'error');
    }
}

// ===== Utilities =====
function escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
}

// ===== Initialize =====
checkHealth();
setInterval(checkHealth, 30000);

// ===== Oracle Particle System =====
(function initParticles() {
    const canvas = document.getElementById('particleCanvas');
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    let particles = [];
    const PARTICLE_COUNT = 60;

    function resize() {
        canvas.width = window.innerWidth;
        canvas.height = window.innerHeight;
    }

    window.addEventListener('resize', resize);
    resize();

    // Moon position (top-right quadrant)
    function getMoon() {
        return {
            x: canvas.width * 0.85,
            y: canvas.height * 0.12,
            radius: Math.min(canvas.width, canvas.height) * 0.04,
        };
    }

    function drawMoon() {
        const moon = getMoon();
        const r = moon.radius;
        const breathe = Math.sin(Date.now() * 0.0006) * 0.08 + 0.92;

        // Outer atmospheric glow (large, faint)
        const outerGlow = ctx.createRadialGradient(moon.x, moon.y, r * 0.5, moon.x, moon.y, r * 6);
        outerGlow.addColorStop(0, `rgba(16, 185, 129, ${0.04 * breathe})`);
        outerGlow.addColorStop(0.3, `rgba(16, 185, 129, ${0.02 * breathe})`);
        outerGlow.addColorStop(1, 'transparent');
        ctx.fillStyle = outerGlow;
        ctx.beginPath();
        ctx.arc(moon.x, moon.y, r * 6, 0, Math.PI * 2);
        ctx.fill();

        // Inner glow halo
        const innerGlow = ctx.createRadialGradient(moon.x, moon.y, r * 0.3, moon.x, moon.y, r * 2.5);
        innerGlow.addColorStop(0, `rgba(200, 230, 215, ${0.08 * breathe})`);
        innerGlow.addColorStop(0.5, `rgba(16, 185, 129, ${0.04 * breathe})`);
        innerGlow.addColorStop(1, 'transparent');
        ctx.fillStyle = innerGlow;
        ctx.beginPath();
        ctx.arc(moon.x, moon.y, r * 2.5, 0, Math.PI * 2);
        ctx.fill();

        // Moon body — crescent effect using two circles
        ctx.save();
        ctx.globalAlpha = 0.15 * breathe;
        ctx.fillStyle = 'rgba(200, 235, 220, 1)';
        ctx.shadowBlur = r * 1.5;
        ctx.shadowColor = 'rgba(16, 185, 129, 0.3)';
        ctx.beginPath();
        ctx.arc(moon.x, moon.y, r, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();

        // Shadow circle to create crescent shape
        ctx.save();
        ctx.globalCompositeOperation = 'destination-out';
        ctx.globalAlpha = 0.85;
        ctx.fillStyle = 'black';
        ctx.beginPath();
        ctx.arc(moon.x + r * 0.45, moon.y - r * 0.15, r * 0.85, 0, Math.PI * 2);
        ctx.fill();
        ctx.restore();
    }

    class Particle {
        constructor() {
            this.reset();
        }

        reset() {
            this.x = Math.random() * canvas.width;
            this.y = Math.random() * canvas.height;
            this.size = Math.random() * 2.2 + 0.6;
            this.speedX = (Math.random() - 0.5) * 0.2;
            this.speedY = (Math.random() - 0.5) * 0.15 - 0.05;
            this.opacity = Math.random() * 0.5 + 0.2;
            this.pulseSpeed = Math.random() * 0.006 + 0.002;
            this.pulseOffset = Math.random() * Math.PI * 2;
            this.life = 0;
        }

        update() {
            this.x += this.speedX;
            this.y += this.speedY;
            this.life += this.pulseSpeed;

            const pulse = Math.sin(this.life + this.pulseOffset) * 0.5 + 0.5;
            this.currentOpacity = this.opacity * (0.35 + pulse * 0.65);

            if (this.x < -10 || this.x > canvas.width + 10 ||
                this.y < -10 || this.y > canvas.height + 10) {
                this.reset();
            }
        }

        draw() {
            ctx.save();
            ctx.globalAlpha = this.currentOpacity;
            ctx.fillStyle = '#34d399';
            ctx.shadowBlur = this.size * 10;
            ctx.shadowColor = 'rgba(52, 211, 153, 0.6)';
            ctx.beginPath();
            ctx.arc(this.x, this.y, this.size, 0, Math.PI * 2);
            ctx.fill();
            ctx.restore();
        }
    }

    for (let i = 0; i < PARTICLE_COUNT; i++) {
        particles.push(new Particle());
    }

    let lastTime = 0;
    const FPS_INTERVAL = 1000 / 30;

    function animate(timestamp) {
        requestAnimationFrame(animate);

        const elapsed = timestamp - lastTime;
        if (elapsed < FPS_INTERVAL) return;
        lastTime = timestamp - (elapsed % FPS_INTERVAL);

        ctx.clearRect(0, 0, canvas.width, canvas.height);

        // Draw moon first (behind stars)
        drawMoon();

        for (const particle of particles) {
            particle.update();
            particle.draw();
        }
    }

    requestAnimationFrame(animate);
})();

// ===== Hide Oracle Welcome on Query =====
const originalSubmitHandler = queryForm.onsubmit;
queryForm.addEventListener('submit', () => {
    const welcome = document.getElementById('oracleWelcome');
    if (welcome) {
        welcome.style.transition = 'opacity 0.4s ease-out, transform 0.4s ease-out';
        welcome.style.opacity = '0';
        welcome.style.transform = 'translateY(-10px)';
        setTimeout(() => welcome.style.display = 'none', 400);
    }
});

// ===== Corpus Panel =====
const corpusList = document.getElementById('corpusList');
const corpusCount = document.getElementById('corpusCount');
let corpusDocuments = [];

async function loadCorpusPanel() {
    try {
        const res = await fetch(`${API_BASE}/api/documents`);
        const data = await res.json();
        corpusDocuments = data.documents || [];

        if (corpusCount) {
            corpusCount.textContent = `${data.total_chunks || 0} chunks`;
        }

        if (!corpusList || corpusDocuments.length === 0) return;

        corpusList.innerHTML = corpusDocuments.map((doc, i) => {
            const ext = doc.name.split('.').pop().toLowerCase();
            const icon = ext === 'pdf' ? '📄' : ext === 'md' || ext === 'markdown' ? '📝' : '📃';
            return `
                <div class="corpus-item" data-source="${escapeHtml(doc.name)}" id="corpus-${i}">
                    <div class="doc-name">
                        <span class="doc-icon">${icon}</span>
                        ${escapeHtml(doc.name)}
                    </div>
                    <div class="doc-chunks">${doc.chunks} chunks</div>
                    <div class="relevance-bar">
                        <div class="relevance-fill" id="relevance-${i}"></div>
                    </div>
                </div>
            `;
        }).join('');
    } catch {
        // Silently fail — panel stays with empty state
    }
}

function highlightCitedDocuments(citations) {
    if (!corpusList) return;

    // Reset all items
    document.querySelectorAll('.corpus-item').forEach(item => {
        item.classList.remove('cited');
        const fill = item.querySelector('.relevance-fill');
        if (fill) fill.style.width = '0%';
    });

    if (!citations || citations.length === 0) return;

    // Find max score for normalization
    const maxScore = Math.max(...citations.map(c => c.relevance_score || 0), 0.01);

    citations.forEach(citation => {
        const sourceName = citation.source
            ? citation.source.replace(/\\/g, '/').split('/').pop()
            : '';
        if (!sourceName) return;

        const items = document.querySelectorAll('.corpus-item');
        items.forEach(item => {
            const itemSource = item.getAttribute('data-source');
            if (itemSource && (itemSource === sourceName || sourceName.includes(itemSource) || itemSource.includes(sourceName))) {
                item.classList.add('cited');
                const fill = item.querySelector('.relevance-fill');
                if (fill) {
                    const pct = Math.round((citation.relevance_score / maxScore) * 100);
                    fill.style.width = `${Math.max(pct, 15)}%`;
                }
                // Scroll into view
                item.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
            }
        });
    });
}

// Load corpus on init
loadCorpusPanel();

// ===== Neural Retrieval Map — Force-Directed Graph =====
const neuralCanvas = document.getElementById('neuralCanvas');
const neuralStatus = document.getElementById('neuralStatus');
let neuralCtx = null;
let neuralNodes = [];
let neuralEdges = [];
let neuralParticles = [];
let neuralAnimating = false;

if (neuralCanvas) {
    neuralCtx = neuralCanvas.getContext('2d');

    function resizeNeuralCanvas() {
        const container = neuralCanvas.parentElement;
        if (container) {
            neuralCanvas.width = container.clientWidth;
            neuralCanvas.height = container.clientHeight - 50; // Account for legend
        }
    }

    window.addEventListener('resize', resizeNeuralCanvas);
    resizeNeuralCanvas();

    // Initialize with idle state
    drawIdleState();
}

function drawIdleState() {
    if (!neuralCtx || !neuralCanvas) return;
    const w = neuralCanvas.width;
    const h = neuralCanvas.height;
    neuralCtx.clearRect(0, 0, w, h);

    // Draw faint grid
    neuralCtx.strokeStyle = 'rgba(16, 185, 129, 0.03)';
    neuralCtx.lineWidth = 0.5;
    for (let x = 0; x < w; x += 30) {
        neuralCtx.beginPath();
        neuralCtx.moveTo(x, 0);
        neuralCtx.lineTo(x, h);
        neuralCtx.stroke();
    }
    for (let y = 0; y < h; y += 30) {
        neuralCtx.beginPath();
        neuralCtx.moveTo(0, y);
        neuralCtx.lineTo(w, y);
        neuralCtx.stroke();
    }

    // Draw "IDLE" text
    neuralCtx.fillStyle = 'rgba(16, 185, 129, 0.1)';
    neuralCtx.font = '10px "JetBrains Mono", monospace';
    neuralCtx.textAlign = 'center';
    neuralCtx.fillText('AWAITING QUERY', w / 2, h / 2);
}

function buildNeuralGraph(citations) {
    if (!neuralCtx || !neuralCanvas) return;

    const w = neuralCanvas.width;
    const h = neuralCanvas.height;

    neuralNodes = [];
    neuralEdges = [];
    neuralParticles = [];

    if (!citations || citations.length === 0) {
        drawIdleState();
        if (neuralStatus) neuralStatus.textContent = 'IDLE';
        return;
    }

    if (neuralStatus) neuralStatus.textContent = 'ACTIVE';

    // Create query node (center)
    const queryNode = {
        id: 'query',
        label: 'QUERY',
        x: w / 2,
        y: h * 0.4,
        vx: 0, vy: 0,
        radius: 18,
        color: '#06b6d4',
        glow: 'rgba(6, 182, 212, 0.5)',
        fixed: true,
        type: 'query',
    };
    neuralNodes.push(queryNode);

    // Get unique sources from citations
    const sourceMap = new Map();
    citations.forEach(c => {
        const name = c.source ? c.source.replace(/\\/g, '/').split('/').pop() : 'unknown';
        if (!sourceMap.has(name)) {
            sourceMap.set(name, { name, maxScore: c.relevance_score, citations: [c] });
        } else {
            const existing = sourceMap.get(name);
            existing.maxScore = Math.max(existing.maxScore, c.relevance_score);
            existing.citations.push(c);
        }
    });

    const maxScore = Math.max(...[...sourceMap.values()].map(s => s.maxScore), 0.01);
    const sources = [...sourceMap.values()];

    // Place document nodes — use width-based radius to fill narrow panel
    const graphRadius = Math.min(w * 0.38, h * 0.32);
    sources.forEach((src, i) => {
        const angle = (i / sources.length) * Math.PI * 2 - Math.PI / 2;
        const strength = src.maxScore / maxScore;
        const dist = graphRadius * (1.05 - strength * 0.2);

        // Truncate name for display — show more characters
        const shortName = src.name.replace(/\.(md|pdf|txt)$/i, '');
        const displayName = shortName.length > 12 ? shortName.substring(0, 11) + '...' : shortName;

        const node = {
            id: `doc-${i}`,
            label: displayName,
            fullName: src.name,
            x: w / 2 + Math.cos(angle) * dist,
            y: h * 0.4 + Math.sin(angle) * dist,
            vx: 0, vy: 0,
            radius: 8 + strength * 10,
            color: `rgba(16, 185, 129, ${0.4 + strength * 0.6})`,
            glow: `rgba(16, 185, 129, ${0.15 + strength * 0.35})`,
            strength,
            fixed: false,
            type: 'document',
        };
        neuralNodes.push(node);

        // Create edge from query to this document
        neuralEdges.push({
            from: queryNode,
            to: node,
            strength,
            color: `rgba(16, 185, 129, ${0.08 + strength * 0.25})`,
        });
    });

    // Also add some inactive "ghost" nodes for un-cited documents
    if (corpusDocuments.length > sources.length) {
        const uncited = corpusDocuments.filter(d =>
            !sources.some(s => s.name === d.name)
        ).slice(0, 6); // Max 6 ghost nodes

        uncited.forEach((doc, i) => {
            const angle = Math.random() * Math.PI * 2;
            const dist = graphRadius * (1.2 + Math.random() * 0.3);
            neuralNodes.push({
                id: `ghost-${i}`,
                label: '',
                fullName: doc.name,
                x: w / 2 + Math.cos(angle) * dist,
                y: h * 0.4 + Math.sin(angle) * dist,
                vx: 0, vy: 0,
                radius: 2.5,
                color: 'rgba(255, 255, 255, 0.1)',
                glow: 'rgba(255, 255, 255, 0.03)',
                strength: 0,
                fixed: false,
                type: 'ghost',
            });
        });
    }

    // Create edge particles
    neuralEdges.forEach(edge => {
        const particleCount = Math.ceil(edge.strength * 4) + 1;
        for (let i = 0; i < particleCount; i++) {
            neuralParticles.push({
                edge,
                t: Math.random(), // position along edge (0-1)
                speed: 0.003 + edge.strength * 0.008,
                size: 1 + edge.strength * 2,
                opacity: 0.3 + edge.strength * 0.5,
            });
        }
    });

    // Start animation
    if (!neuralAnimating) {
        neuralAnimating = true;
        animateNeuralGraph();
    }
}

function animateNeuralGraph() {
    if (!neuralCtx || !neuralCanvas || neuralNodes.length === 0) {
        neuralAnimating = false;
        return;
    }

    const w = neuralCanvas.width;
    const h = neuralCanvas.height;

    neuralCtx.clearRect(0, 0, w, h);

    // Draw faint grid
    neuralCtx.strokeStyle = 'rgba(16, 185, 129, 0.02)';
    neuralCtx.lineWidth = 0.5;
    for (let x = 0; x < w; x += 30) {
        neuralCtx.beginPath();
        neuralCtx.moveTo(x, 0);
        neuralCtx.lineTo(x, h);
        neuralCtx.stroke();
    }
    for (let y = 0; y < h; y += 30) {
        neuralCtx.beginPath();
        neuralCtx.moveTo(0, y);
        neuralCtx.lineTo(w, y);
        neuralCtx.stroke();
    }

    // Draw edges
    neuralEdges.forEach(edge => {
        neuralCtx.beginPath();
        neuralCtx.moveTo(edge.from.x, edge.from.y);
        neuralCtx.lineTo(edge.to.x, edge.to.y);
        neuralCtx.strokeStyle = edge.color;
        neuralCtx.lineWidth = 0.5 + edge.strength * 1.5;
        neuralCtx.stroke();
    });

    // Draw and update particles along edges
    neuralParticles.forEach(p => {
        p.t += p.speed;
        if (p.t > 1) p.t -= 1;

        const x = p.edge.from.x + (p.edge.to.x - p.edge.from.x) * p.t;
        const y = p.edge.from.y + (p.edge.to.y - p.edge.from.y) * p.t;

        neuralCtx.save();
        neuralCtx.globalAlpha = p.opacity;
        neuralCtx.fillStyle = '#10b981';
        neuralCtx.shadowBlur = p.size * 4;
        neuralCtx.shadowColor = 'rgba(16, 185, 129, 0.6)';
        neuralCtx.beginPath();
        neuralCtx.arc(x, y, p.size, 0, Math.PI * 2);
        neuralCtx.fill();
        neuralCtx.restore();
    });

    // Draw nodes
    neuralNodes.forEach(node => {
        // Glow
        neuralCtx.save();
        neuralCtx.globalAlpha = 1;
        neuralCtx.fillStyle = node.glow;
        neuralCtx.shadowBlur = node.radius * 3;
        neuralCtx.shadowColor = node.glow;
        neuralCtx.beginPath();
        neuralCtx.arc(node.x, node.y, node.radius + 4, 0, Math.PI * 2);
        neuralCtx.fill();
        neuralCtx.restore();

        // Node body
        neuralCtx.beginPath();
        neuralCtx.arc(node.x, node.y, node.radius, 0, Math.PI * 2);
        neuralCtx.fillStyle = node.color;
        neuralCtx.fill();

        // Border
        if (node.type === 'query') {
            neuralCtx.strokeStyle = '#22d3ee';
            neuralCtx.lineWidth = 2;
            neuralCtx.stroke();
        } else if (node.type === 'document') {
            neuralCtx.strokeStyle = `rgba(16, 185, 129, ${0.3 + node.strength * 0.5})`;
            neuralCtx.lineWidth = 1;
            neuralCtx.stroke();
        }

        // Label
        if (node.label) {
            neuralCtx.fillStyle = node.type === 'query' ? '#ffffff' : 'rgba(255, 255, 255, 0.8)';
            neuralCtx.font = node.type === 'query'
                ? 'bold 10px "JetBrains Mono", monospace'
                : '7px "JetBrains Mono", monospace';
            neuralCtx.textAlign = 'center';
            neuralCtx.textBaseline = 'middle';

            if (node.type === 'query') {
                neuralCtx.fillText(node.label, node.x, node.y);
            } else {
                neuralCtx.fillText(node.label, node.x, node.y + node.radius + 10);
            }
        }

        // Gentle floating animation for non-fixed nodes
        if (!node.fixed) {
            node.x += Math.sin(Date.now() * 0.001 + node.x * 0.1) * 0.15;
            node.y += Math.cos(Date.now() * 0.0008 + node.y * 0.1) * 0.1;
        }

        // Query node pulse
        if (node.type === 'query') {
            const pulse = Math.sin(Date.now() * 0.003) * 0.15 + 0.85;
            neuralCtx.save();
            neuralCtx.globalAlpha = 0.15 * pulse;
            neuralCtx.beginPath();
            neuralCtx.arc(node.x, node.y, node.radius + 10 + pulse * 5, 0, Math.PI * 2);
            neuralCtx.strokeStyle = '#06b6d4';
            neuralCtx.lineWidth = 1;
            neuralCtx.stroke();
            neuralCtx.restore();
        }
    });

    requestAnimationFrame(animateNeuralGraph);
}

// Reset panels on new query
queryForm.addEventListener('submit', () => {
    // Reset corpus highlights
    document.querySelectorAll('.corpus-item').forEach(item => {
        item.classList.remove('cited');
        const fill = item.querySelector('.relevance-fill');
        if (fill) fill.style.width = '0%';
    });

    // Reset neural graph
    neuralNodes = [];
    neuralEdges = [];
    neuralParticles = [];
    neuralAnimating = false;
    if (neuralStatus) neuralStatus.textContent = 'SCANNING...';
    drawIdleState();

    // Reload corpus in case new docs were ingested
    loadCorpusPanel();
});
