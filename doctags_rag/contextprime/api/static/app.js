const API_BASE = '/api';
const DEFAULT_CHUNK_HINT =
  'Structure aware preserves headings. Semantic uses embeddings when configured.';

const state = {
  theme: 'light',
  accessToken: '',
  processing: false,
  documents: [],
  currentDocumentId: null,
  semanticChunkingAvailable: false,
  semanticChunkingMessage: null,
  latestSearchResults: [],
  latestSearchMetadata: {},
  latestSearchRerank: false,
  searchFeedback: {
    queryId: null,
    helpful: null,
    resultLabels: {},
    comment: '',
  },
};

document.addEventListener('DOMContentLoaded', () => {
  initialiseTheme();
  initialiseAccessToken();
  bindEvents();
  bootstrap();
});

async function bootstrap() {
  await refreshStatus();
  await refreshNeo4jConnectivity();
  await refreshDocuments();
  await refreshTrendSummary();
}

// ---------------------------------------------------------------------------
// Theme handling
// ---------------------------------------------------------------------------

function initialiseTheme() {
  const stored = localStorage.getItem('wwc-theme');
  const prefersDark = window.matchMedia('(prefers-color-scheme: dark)').matches;
  state.theme = stored || (prefersDark ? 'dark' : 'light');
  applyTheme(state.theme);
}

function toggleTheme() {
  state.theme = state.theme === 'dark' ? 'light' : 'dark';
  applyTheme(state.theme);
  localStorage.setItem('wwc-theme', state.theme);
}

function applyTheme(theme) {
  document.body.dataset.theme = theme;
}

// ---------------------------------------------------------------------------
// Event binding
// ---------------------------------------------------------------------------

function bindEvents() {
  const accessTokenInput = document.getElementById('accessTokenInput');
  if (accessTokenInput) {
    accessTokenInput.addEventListener('input', (event) => {
      persistAccessToken(event.target.value || '');
    });
  }

  const clearAccessToken = document.getElementById('clearAccessToken');
  if (clearAccessToken) {
    clearAccessToken.addEventListener('click', () => {
      persistAccessToken('');
      const tokenInput = document.getElementById('accessTokenInput');
      if (tokenInput) {
        tokenInput.value = '';
      }
      showToast('Access token cleared', 'success');
    });
  }

  const uploadForm = document.getElementById('uploadForm');
  if (uploadForm) {
    uploadForm.addEventListener('submit', handleUpload);
  }

  const searchForm = document.getElementById('searchForm');
  if (searchForm) {
    searchForm.addEventListener('submit', handleSearch);
  }

  const agenticForm = document.getElementById('agenticForm');
  if (agenticForm) {
    agenticForm.addEventListener('submit', handleAgentic);
  }

  const themeToggle = document.getElementById('themeToggle');
  if (themeToggle) {
    themeToggle.addEventListener('click', toggleTheme);
  }

  const trendRefresh = document.getElementById('trendRefresh');
  if (trendRefresh) {
    trendRefresh.addEventListener('click', () => {
      refreshTrendSummary();
    });
  }

  const helpfulButton = document.getElementById('searchFeedbackHelpful');
  if (helpfulButton) {
    helpfulButton.addEventListener('click', () => {
      setOverallFeedback(true);
    });
  }

  const needsWorkButton = document.getElementById('searchFeedbackNeedsWork');
  if (needsWorkButton) {
    needsWorkButton.addEventListener('click', () => {
      setOverallFeedback(false);
    });
  }

  const feedbackComment = document.getElementById('searchFeedbackComment');
  if (feedbackComment) {
    feedbackComment.addEventListener('input', (event) => {
      state.searchFeedback.comment = event.target.value || '';
    });
  }

  const feedbackSubmit = document.getElementById('searchFeedbackSubmit');
  if (feedbackSubmit) {
    feedbackSubmit.addEventListener('click', submitSearchFeedback);
  }

  const neo4jRecoveryForm = document.getElementById('neo4jRecoveryForm');
  if (neo4jRecoveryForm) {
    neo4jRecoveryForm.addEventListener('submit', handleNeo4jRecovery);
  }

  const neo4jRecoveryRefresh = document.getElementById('neo4jRecoveryRefresh');
  if (neo4jRecoveryRefresh) {
    neo4jRecoveryRefresh.addEventListener('click', () => {
      refreshNeo4jConnectivity();
    });
  }
}

function initialiseAccessToken() {
  state.accessToken = '';
  const tokenInput = document.getElementById('accessTokenInput');
  if (tokenInput) {
    tokenInput.value = state.accessToken;
  }
}

function persistAccessToken(token) {
  state.accessToken = (token || '').trim();
}

function buildRequestHeaders(initialHeaders) {
  const headers = new Headers(initialHeaders || {});
  if (state.accessToken) {
    headers.set('Authorization', `Bearer ${state.accessToken}`);
  }
  return headers;
}

async function apiFetch(path, options = {}) {
  const requestOptions = { ...options };
  requestOptions.headers = buildRequestHeaders(options.headers);
  return fetch(`${API_BASE}${path}`, requestOptions);
}

// ---------------------------------------------------------------------------
// Status and configuration helpers
// ---------------------------------------------------------------------------

async function refreshStatus() {
  try {
    const response = await apiFetch('/status');
    if (!response.ok) {
      throw new Error('Failed to fetch status');
    }
    const data = await response.json();
    const semantic = data?.services?.processing?.details?.semantic_chunking || {};
    state.semanticChunkingAvailable = Boolean(semantic.available);
    state.semanticChunkingMessage = semantic.reason || null;
    renderServiceStatus(data);
  } catch (error) {
    console.warn('Status refresh failed', error);
    state.semanticChunkingAvailable = false;
    renderServiceStatus(null);
  } finally {
    updateSemanticOption();
  }
}

async function refreshNeo4jConnectivity() {
  const statusEl = document.getElementById('neo4jRecoveryStatus');
  if (!statusEl) return;

  setHintStatus(statusEl, 'Checking Neo4j password status…');
  try {
    const response = await apiFetch('/admin/neo4j/connectivity');
    const data = await response.json();
    if (!response.ok) {
      const message = data?.detail || data?.message || 'Neo4j connectivity check failed';
      throw new Error(message);
    }

    const connected = Boolean(data.connected);
    setHintStatus(
      statusEl,
      data?.message || (connected ? 'Neo4j connection is healthy.' : 'Neo4j connection is not healthy.'),
      connected ? 'success' : 'error'
    );
  } catch (error) {
    setHintStatus(statusEl, error.message || 'Unable to check Neo4j connectivity.', 'error');
  }
}

async function handleNeo4jRecovery(event) {
  event.preventDefault();
  const passwordInput = document.getElementById('neo4jRecoveryPassword');
  const persistInput = document.getElementById('neo4jPersistEnv');
  const statusEl = document.getElementById('neo4jRecoveryStatus');

  const password = passwordInput?.value?.trim() || '';
  if (!password) {
    setHintStatus(statusEl, 'Enter a Neo4j password first.', 'error');
    showToast('Enter a Neo4j password first.', 'error');
    return;
  }

  setHintStatus(statusEl, 'Verifying Neo4j password…');
  try {
    const response = await apiFetch('/admin/neo4j/recover-password', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        password,
        persist_to_env: persistInput?.checked ?? true,
      }),
    });
    const data = await response.json();
    if (!response.ok) {
      const message = data?.detail || data?.message || 'Neo4j password verification failed';
      throw new Error(message);
    }

    if (passwordInput) {
      passwordInput.value = '';
    }
    setHintStatus(statusEl, data?.message || 'Neo4j password verified.', 'success');
    showToast('Neo4j password verified.', 'success');
    await refreshNeo4jConnectivity();
  } catch (error) {
    setHintStatus(statusEl, error.message || 'Unable to verify Neo4j password.', 'error');
    showToast(error.message || 'Unable to verify Neo4j password.', 'error');
  }
}

// ---------------------------------------------------------------------------
// Upload and document management
// ---------------------------------------------------------------------------

async function handleUpload(event) {
  event.preventDefault();
  if (state.processing) {
    return;
  }

  const fileInput = document.getElementById('fileInput');
  const chunkSizeEl = document.getElementById('chunkSize');
  const chunkOverlapEl = document.getElementById('chunkOverlap');
  const chunkMethodEl = document.getElementById('chunkMethod');
  const enableOcrEl = document.getElementById('enableOcr');
  const extractEntitiesEl = document.getElementById('extractEntities');
  const buildRaptorEl = document.getElementById('buildRaptor');

  const file = fileInput?.files?.[0];
  if (!file) {
    showToast('Please choose a document to upload.', 'error');
    return;
  }

  const chunkSize = Number(chunkSizeEl?.value) || 1000;
  const chunkOverlap = Number(chunkOverlapEl?.value) || 200;

  if (chunkOverlap >= chunkSize) {
    showToast('Chunk overlap must be smaller than chunk size.', 'error');
    return;
  }

  const settings = {
    enable_ocr: enableOcrEl?.checked ?? true,
    chunk_size: chunkSize,
    chunk_overlap: chunkOverlap,
    chunking_method: chunkMethodEl?.value || 'structure',
    extract_entities: extractEntitiesEl?.checked ?? false,
    build_raptor: buildRaptorEl?.checked ?? false,
  };

  const formData = new FormData();
  formData.append('file', file);
  formData.append('settings', JSON.stringify(settings));

  setProcessing(true, `Processing ${file.name}…`);

  try {
    const response = await apiFetch('/documents', {
      method: 'POST',
      body: formData,
    });

    const payload = await response.json();
    if (!response.ok) {
      const message = payload?.detail || payload?.message || 'Upload failed';
      throw new Error(message);
    }

    showToast(payload.message || 'Document processed', 'success');
    state.currentDocumentId = payload.document?.id || null;
    await refreshDocuments(state.currentDocumentId);
    await refreshStatus();
    event.target.reset();
    document.getElementById('chunkSize').value = 1000;
    document.getElementById('chunkOverlap').value = 200;
    document.getElementById('enableOcr').checked = true;
    document.getElementById('extractEntities').checked = false;
    document.getElementById('buildRaptor').checked = false;
    updateSemanticOption();
  } catch (error) {
    showToast(error.message || 'Unable to process document', 'error');
  } finally {
    setProcessing(false);
  }
}

async function refreshDocuments(preselectId = null) {
  try {
    const response = await apiFetch('/documents');
    if (!response.ok) {
      throw new Error('Failed to load documents');
    }
    const data = await response.json();
    state.documents = data.documents || [];
    renderDocumentList(preselectId);
  } catch (error) {
    showToast(error.message || 'Unable to fetch documents', 'error');
  }
}

function renderDocumentList(preselectId) {
  const container = document.getElementById('documentContainer');
  const detailSection = document.getElementById('detailContainer');
  const documents = state.documents;

  if (!container) return;

  if (!documents.length) {
    container.innerHTML = '<div class="document-empty">No documents processed yet.</div>';
    if (detailSection) detailSection.hidden = true;
    state.currentDocumentId = null;
    return;
  }

  const list = document.createElement('div');
  list.className = 'document-list';

  documents.forEach((doc, index) => {
    const item = document.createElement('div');
    item.className = 'document-item';
    const shouldSelect =
      (preselectId && doc.id === preselectId) ||
      (!preselectId && !state.currentDocumentId && index === 0) ||
      state.currentDocumentId === doc.id;
    if (shouldSelect) {
      item.classList.add('active');
      state.currentDocumentId = doc.id;
    }

    const infoLine = formatDocumentMeta(doc);
    item.innerHTML = `<h3>${escapeHtml(doc.filename || doc.id)}</h3><span class="meta">${escapeHtml(
      infoLine
    )}</span>`;
    item.addEventListener('click', () => {
      state.currentDocumentId = doc.id;
      renderDocumentList(doc.id);
    });

    list.appendChild(item);
  });

  container.innerHTML = '';
  container.appendChild(list);
  if (detailSection) detailSection.hidden = false;

  if (state.currentDocumentId) {
    loadDocumentDetail(state.currentDocumentId);
  }
}

function formatDocumentMeta(doc) {
  const chunks = doc.num_chunks ?? 0;
  const time = doc.processing_time ? `${doc.processing_time.toFixed(2)}s` : 'n/a';
  const chunkMethod = doc.metadata?.chunking_method || 'structure';
  return `${chunks} chunk${chunks === 1 ? '' : 's'} • ${time} • ${chunkMethod}`;
}

async function loadDocumentDetail(documentId) {
  try {
    const response = await apiFetch(`/documents/${encodeURIComponent(documentId)}`);
    if (!response.ok) {
      throw new Error('Failed to load document details');
    }
    const data = await response.json();
    renderDocumentDetail(data);
  } catch (error) {
    showToast(error.message || 'Unable to fetch document detail', 'error');
  }
}

function renderDocumentDetail(detail) {
  const metaGrid = document.getElementById('documentMeta');
  const markdownPreview = document.getElementById('markdownPreview');
  if (!metaGrid || !markdownPreview) return;

  const doc = detail.document;
  const uploaded = doc.uploaded_at ? new Date(doc.uploaded_at).toLocaleString() : 'n/a';
  const tagCounts = doc.metadata?.tag_counts || {};
  const topTags = Object.entries(tagCounts)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 4)
    .map(([tag, count]) => `${tag}: ${count}`)
    .join(', ');

  const metaItems = [
    { label: 'Filename', value: doc.filename || doc.id },
    { label: 'Document ID', value: doc.id },
    { label: 'Processing Time', value: doc.processing_time ? `${doc.processing_time.toFixed(2)} s` : 'n/a' },
    { label: 'Chunks', value: doc.num_chunks ?? 0 },
    { label: 'Chunk Method', value: doc.metadata?.chunking_method || 'structure' },
    { label: 'Uploaded', value: uploaded },
    { label: 'Top Tags', value: topTags || 'n/a' },
  ];

  metaGrid.innerHTML = '';
  metaItems.forEach((entry) => {
    const card = document.createElement('div');
    card.className = 'meta-card';
    card.innerHTML = `<span class="label">${escapeHtml(entry.label)}</span><span class="value">${escapeHtml(
      String(entry.value)
    )}</span>`;
    metaGrid.appendChild(card);
  });

  markdownPreview.textContent = detail.markdown || 'No markdown reconstruction available.';
}

// ---------------------------------------------------------------------------
// Hybrid search
// ---------------------------------------------------------------------------

async function handleSearch(event) {
  event.preventDefault();

  const query = document.getElementById('searchQuery')?.value?.trim();
  if (!query) {
    showToast('Enter a query to search.', 'error');
    return;
  }

  const topK = Number(document.getElementById('searchTopK')?.value) || 5;
  const strategy = document.getElementById('searchStrategy')?.value || 'hybrid';
  const graphPolicy = document.getElementById('searchGraphPolicy')?.value || 'standard';
  const useRerank = document.getElementById('searchRerank')?.checked ?? true;
  const includeContext = document.getElementById('searchIncludeContext')?.checked ?? true;
  const useQueryExpansion = document.getElementById('searchExpandQuery')?.checked ?? false;

  let vectorWeight = 0.7;
  let graphWeight = 0.3;
  if (strategy === 'vector') {
    vectorWeight = 1.0;
    graphWeight = 0.0;
  } else if (strategy === 'graph') {
    vectorWeight = 0.0;
    graphWeight = 1.0;
  }

  const payload = {
    query,
    top_k: topK,
    strategy,
    vector_weight: vectorWeight,
    graph_weight: graphWeight,
    use_reranking: useRerank,
    use_query_expansion: useQueryExpansion,
    include_graph_context: includeContext,
    graph_policy: graphPolicy,
  };

  const metricsPanel = document.getElementById('searchMetrics');
  const resultsContainer = document.getElementById('searchResults');
  if (metricsPanel) metricsPanel.classList.add('hidden');
  if (resultsContainer) resultsContainer.innerHTML = '<p class="lead">Searching…</p>';

  try {
    const response = await apiFetch('/search/hybrid', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });

    const data = await response.json();
    if (!response.ok) {
      const message = data?.detail || data?.message || 'Search failed';
      throw new Error(message);
    }

    state.latestSearchResults = data.results || [];
    state.latestSearchMetadata = data.metadata || {};
    state.latestSearchRerank = Boolean(state.latestSearchMetadata?.rerank_applied);

    initialiseSearchFeedback(state.latestSearchMetadata, state.latestSearchResults);
    renderSearchMetrics(state.latestSearchMetadata);
    renderSearchResults(state.latestSearchResults, state.latestSearchRerank);
    showToast('Hybrid search completed', 'success');
  } catch (error) {
    state.latestSearchResults = [];
    state.latestSearchMetadata = {};
    state.latestSearchRerank = false;
    if (resultsContainer) {
      resultsContainer.innerHTML = `<p class="lead">${escapeHtml(error.message || 'Search failed')}</p>`;
    }
    if (metricsPanel) metricsPanel.classList.add('hidden');
    initialiseSearchFeedback({}, []);
    showToast(error.message || 'Unable to execute search', 'error');
  }
}

function renderSearchMetrics(metadata) {
  const panel = document.getElementById('searchMetrics');
  if (!panel) return;

  const entries = [
    { label: 'Query Type', value: metadata.query_type || 'n/a' },
    { label: 'Strategy', value: metadata.strategy || 'n/a' },
    { label: 'Graph Policy', value: metadata?.services?.graph_policy || 'default' },
    { label: 'Vector Results', value: metadata.vector_results ?? '0' },
    { label: 'Graph Results', value: metadata.graph_results ?? '0' },
    { label: 'Combined Results', value: metadata.combined_results ?? '0' },
    { label: 'Total Time', value: metadata.total_time_ms ? `${metadata.total_time_ms.toFixed(2)} ms` : 'n/a' },
    { label: 'Rerank Applied', value: metadata.rerank_applied ? 'Yes' : 'No' },
    { label: 'Cache Hit', value: metadata.cache_hit ? 'Yes' : 'No' },
  ];

  if (typeof metadata.rerank_time_ms === 'number') {
    entries.splice(entries.length - 2, 0, {
      label: 'Rerank Time',
      value: `${Number(metadata.rerank_time_ms).toFixed(2)} ms`,
    });
  }

  panel.innerHTML = entries
    .map(
      (entry) => `
        <div class="metric-card">
          <span class="label">${escapeHtml(entry.label)}</span>
          <span class="value">${escapeHtml(String(entry.value))}</span>
        </div>
      `
    )
    .join('');

  if (metadata.services && typeof metadata.services === 'object') {
    const serviceChips = Object.entries(metadata.services)
      .map(([name, status]) => {
        if (typeof status === 'boolean') {
          const statusClass = status ? 'ok' : 'warn';
          const label = status ? 'online' : 'offline';
          return `<span class="service-chip ${statusClass}">${escapeHtml(name)} • ${label}</span>`;
        }
        return `<span class="service-chip">${escapeHtml(name)} • ${escapeHtml(String(status))}</span>`;
      })
      .join('');

    panel.insertAdjacentHTML(
      'beforeend',
      `
        <div class="metric-card service-status">
          <span class="label">Services</span>
          <div class="service-chip-row">${serviceChips || '<span class="hint">—</span>'}</div>
        </div>
      `
    );
  }

  panel.classList.remove('hidden');
}

function renderSearchResults(results, rerankApplied) {
  const container = document.getElementById('searchResults');
  if (!container) return;

  const previousHint = container.previousElementSibling;
  if (previousHint && previousHint.classList?.contains('search-hint')) {
    previousHint.remove();
  }

  if (!results.length) {
    container.innerHTML = '<p class="lead">No results returned for this query.</p>';
    return;
  }

  container.innerHTML = '';
  results.forEach((item, index) => {
    const card = document.createElement('article');
    card.className = 'result-item';
    const context = item.graph_context
      ? `<details><summary>Graph context</summary><pre>${escapeHtml(
          JSON.stringify(item.graph_context, null, 2)
        )}</pre></details>`
      : '';
    const resultId = String(item.id || '');
    const currentLabel = state.searchFeedback.resultLabels[resultId];
    card.innerHTML = `
      <header>
        <h4>${escapeHtml(item.metadata?.doc_title || item.metadata?.title || `Result ${index + 1}`)}</h4>
        <span class="score-chip">${escapeHtml(item.source)} • ${item.score.toFixed(4)}</span>
      </header>
      <p>${escapeHtml(item.content)}</p>
      ${context}
    `;

    if (state.searchFeedback.queryId && resultId) {
      const feedbackRow = document.createElement('div');
      feedbackRow.className = 'result-feedback-row';
      feedbackRow.appendChild(
        createResultFeedbackButton(
          resultId,
          1,
          'Mark relevant',
          currentLabel === 1 ? 'active-positive' : ''
        )
      );
      feedbackRow.appendChild(
        createResultFeedbackButton(
          resultId,
          0,
          'Mark not relevant',
          currentLabel === 0 ? 'active-negative' : ''
        )
      );
      card.appendChild(feedbackRow);
    }
    container.appendChild(card);
  });

  if (rerankApplied) {
    container.insertAdjacentHTML(
      'beforebegin',
      '<p class="hint search-hint">MonoT5 reranking applied — results sorted by cross-encoder confidence.</p>'
    );
  }
}

function createResultFeedbackButton(resultId, label, text, activeClass = '') {
  const button = document.createElement('button');
  button.type = 'button';
  button.className = `secondary ${activeClass}`.trim();
  button.textContent = text;
  button.addEventListener('click', () => {
    setResultFeedback(resultId, label);
  });
  return button;
}

function setResultFeedback(resultId, label) {
  if (!resultId) return;
  state.searchFeedback.resultLabels[resultId] = label;
  renderSearchResults(state.latestSearchResults, state.latestSearchRerank);
  renderSearchFeedbackPanel();
}

function initialiseSearchFeedback(metadata, results) {
  const queryId = metadata?.query_id || null;
  state.searchFeedback = {
    queryId,
    helpful: null,
    resultLabels: {},
    comment: '',
  };

  // Keep first result marked relevant as a starting point for fast feedback loops.
  if (queryId && results?.length) {
    const firstId = String(results[0]?.id || '');
    if (firstId) {
      state.searchFeedback.resultLabels[firstId] = 1;
    }
  }
  renderSearchFeedbackPanel();
}

function setOverallFeedback(helpful) {
  if (!state.searchFeedback.queryId) return;
  state.searchFeedback.helpful = helpful;
  renderSearchFeedbackPanel();
}

function renderSearchFeedbackPanel() {
  const panel = document.getElementById('searchFeedbackPanel');
  const queryIdEl = document.getElementById('feedbackQueryId');
  const helpfulButton = document.getElementById('searchFeedbackHelpful');
  const needsWorkButton = document.getElementById('searchFeedbackNeedsWork');
  const commentInput = document.getElementById('searchFeedbackComment');
  const statusEl = document.getElementById('searchFeedbackStatus');

  if (!panel || !queryIdEl || !helpfulButton || !needsWorkButton || !commentInput || !statusEl) {
    return;
  }

  if (!state.searchFeedback.queryId) {
    panel.classList.add('hidden');
    queryIdEl.textContent = '';
    statusEl.textContent = '';
    return;
  }

  panel.classList.remove('hidden');
  queryIdEl.textContent = `Query id: ${state.searchFeedback.queryId}`;

  helpfulButton.className = `secondary ${
    state.searchFeedback.helpful === true ? 'active-positive' : ''
  }`.trim();
  needsWorkButton.className = `secondary ${
    state.searchFeedback.helpful === false ? 'active-negative' : ''
  }`.trim();
  commentInput.value = state.searchFeedback.comment || '';

  const labels = Object.values(state.searchFeedback.resultLabels);
  const positiveCount = labels.filter((label) => label === 1).length;
  const negativeCount = labels.filter((label) => label === 0).length;
  statusEl.textContent = `Marked relevant: ${positiveCount}, marked not relevant: ${negativeCount}`;
}

async function submitSearchFeedback() {
  const queryId = state.searchFeedback.queryId;
  if (!queryId) {
    showToast('No query identifier available for feedback.', 'error');
    return;
  }

  const statusEl = document.getElementById('searchFeedbackStatus');
  if (statusEl) {
    statusEl.textContent = 'Submitting feedback...';
  }

  const resultLabels = Object.entries(state.searchFeedback.resultLabels).map(([resultId, label]) => ({
    result_id: resultId,
    label,
  }));
  const selectedResultIds = resultLabels
    .filter((item) => Number(item.label) === 1)
    .map((item) => item.result_id);

  const payload = {
    query_id: queryId,
    helpful: state.searchFeedback.helpful,
    selected_result_ids: selectedResultIds,
    result_labels: resultLabels,
    comment: state.searchFeedback.comment || '',
    metadata: {
      source: 'web_interface',
      graph_policy: state.latestSearchMetadata?.services?.graph_policy || null,
      strategy: state.latestSearchMetadata?.strategy || null,
    },
  };

  try {
    const response = await apiFetch('/feedback/retrieval', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      const message = data?.detail || data?.message || 'Feedback submission failed';
      throw new Error(message);
    }
    if (statusEl) {
      statusEl.textContent = `Feedback stored (${data.feedback_id})`;
    }
    showToast('Feedback submitted', 'success');
  } catch (error) {
    if (statusEl) {
      statusEl.textContent = 'Feedback submission failed';
    }
    showToast(error.message || 'Unable to submit feedback', 'error');
  }
}

async function refreshTrendSummary() {
  const panel = document.getElementById('searchTrendPanel');
  if (!panel) return;

  try {
    const response = await apiFetch('/feedback/trends');
    const data = await response.json();
    if (!response.ok) {
      throw new Error(data?.detail || 'Unable to load trend summary');
    }
    renderTrendSummary(data);
  } catch (error) {
    renderTrendSummary({
      available: false,
      message: error.message || 'Unable to load trend summary',
    });
  }
}

function renderTrendSummary(data) {
  const panel = document.getElementById('searchTrendPanel');
  const text = document.getElementById('trendSummaryText');
  const links = document.getElementById('trendLinks');
  if (!panel || !text || !links) return;

  panel.classList.remove('hidden');
  if (!data?.available) {
    text.textContent =
      data?.message ||
      'No published trend summary found yet. Run policy benchmark with trend publishing enabled.';
    links.innerHTML = '';
    return;
  }

  const historyCount = Number(data.history_records || 0);
  const updated = data.last_updated || 'n/a';
  text.textContent = `Trend records: ${historyCount}. Last updated: ${updated}`;

  const markdownEndpoint = data.markdown_endpoint || '/api/feedback/trends/markdown';
  links.innerHTML = `
    <a class="trend-link" href="${escapeHtml(markdownEndpoint)}" target="_blank" rel="noopener noreferrer">
      Open trend summary
    </a>
  `;
}

// ---------------------------------------------------------------------------
// Agentic pipeline
// ---------------------------------------------------------------------------

async function handleAgentic(event) {
  event.preventDefault();

  const query = document.getElementById('agenticQuery')?.value?.trim();
  if (!query) {
    showToast('Enter a question for the agentic pipeline.', 'error');
    return;
  }

  const iterations = Number(document.getElementById('agenticIterations')?.value) || 3;
  const returnReasoning = document.getElementById('agenticReasoningToggle')?.checked ?? true;

  const payload = {
    query,
    max_iterations: iterations,
    use_evaluation: true,
    return_reasoning: returnReasoning,
  };

  const output = document.getElementById('agenticOutput');
  if (output) {
    output.classList.add('hidden');
  }

  try {
    const response = await apiFetch('/agentic', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
    });
    const data = await response.json();
    if (!response.ok) {
      const message = data?.detail || data?.message || 'Agentic query failed';
      throw new Error(message);
    }

    renderAgenticResponse(data);
    showToast('Agentic pipeline completed', 'success');
  } catch (error) {
    showToast(error.message || 'Unable to complete agentic query', 'error');
  }
}

function renderAgenticResponse(data) {
  const output = document.getElementById('agenticOutput');
  const answerEl = document.getElementById('agenticAnswer');
  const scoreEl = document.getElementById('agenticScore');
  const confidenceEl = document.getElementById('agenticConfidence');
  const reasoningContainer = document.getElementById('agenticReasoning');
  const reasoningList = document.getElementById('agenticReasoningList');

  if (!output || !answerEl || !scoreEl || !confidenceEl || !reasoningContainer || !reasoningList) {
    return;
  }

  answerEl.textContent = data.answer || 'No answer produced.';
  scoreEl.textContent = `Score: ${(data.metadata?.assessment?.overall_score ?? data.confidence ?? 0).toFixed(3)}`;
  confidenceEl.textContent = `Confidence: ${(data.confidence ?? 0).toFixed(3)}`;

  if (data.reasoning_steps?.length) {
    reasoningList.innerHTML = '';
    data.reasoning_steps.forEach((step) => {
      const li = document.createElement('li');
      const status = step.success === true ? '✅' : step.success === false ? '⚠️' : 'ℹ️';
      li.innerHTML = `
        <strong>${status} ${escapeHtml(step.description || step.step_id)}</strong>
        <div class="hint">${escapeHtml(step.type || '')}</div>
        ${step.results && step.results.length
          ? `<pre>${escapeHtml(JSON.stringify(step.results.slice(0, 2), null, 2))}</pre>`
          : ''}
      `;
      reasoningList.appendChild(li);
    });
    reasoningContainer.classList.remove('hidden');
  } else {
    reasoningContainer.classList.add('hidden');
  }

  output.classList.remove('hidden');
}

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function renderServiceStatus(statusData) {
  const panel = document.getElementById('serviceStatus');
  if (!panel) return;

  if (!statusData) {
    panel.innerHTML = `
      <div class="status-card warn">
        <span class="status-label">Status</span>
        <span class="status-value">Unavailable</span>
      </div>
    `;
    return;
  }

  const uptime = formatDuration(statusData.uptime_seconds || 0);
  const serviceCards = Object.values(statusData.services || {})
    .map((info) => {
      const status = (info.status || 'unknown').toLowerCase();
      const message = info.message || '';
      const details = info.details || {};
      const docCount = details.documents_processed ?? details.total ?? null;
      return `
        <div class="status-card ${status}">
          <span class="status-label">${escapeHtml(info.name || 'Service')}</span>
          <span class="status-value">${escapeHtml(info.status || 'Unknown')}</span>
          ${docCount !== null ? `<span class="status-hint">Docs: ${escapeHtml(String(docCount))}</span>` : ''}
          ${message ? `<span class="status-hint">${escapeHtml(message)}</span>` : ''}
        </div>
      `;
    })
    .join('');

  panel.innerHTML = `
    <div class="status-card neutral">
      <span class="status-label">Uptime</span>
      <span class="status-value">${escapeHtml(uptime)}</span>
    </div>
    ${serviceCards}
  `;
}

function formatDuration(totalSeconds) {
  const seconds = Math.max(0, Number(totalSeconds) || 0);
  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  if (hours > 0) {
    return `${hours}h ${minutes}m`;
  }
  if (minutes > 0) {
    return `${minutes}m`;
  }
  return `${Math.floor(seconds)}s`;
}

function setProcessing(active, message = 'Processing…') {
  state.processing = active;
  const banner = document.getElementById('statusBanner');
  const statusText = document.getElementById('statusText');
  if (!banner || !statusText) return;

  if (active) {
    banner.classList.remove('hidden');
    statusText.textContent = message;
  } else {
    banner.classList.add('hidden');
    statusText.textContent = '';
  }
}

let toastTimeout = null;
function showToast(message, type = 'info') {
  const toast = document.getElementById('toast');
  if (!toast) return;

  toast.textContent = message;
  toast.className = `toast visible ${type === 'error' ? 'error' : type === 'success' ? 'success' : ''}`;

  if (toastTimeout) {
    clearTimeout(toastTimeout);
  }

  toastTimeout = setTimeout(() => {
    toast.classList.remove('visible');
  }, 3600);
}

function setHintStatus(element, message, tone = '') {
  if (!element) return;
  const normalizedTone = tone === 'success' || tone === 'error' ? tone : '';
  element.className = ['hint', normalizedTone].filter(Boolean).join(' ');
  element.textContent = message;
}

function escapeHtml(value) {
  const text = value === undefined || value === null ? '' : String(value);
  return text
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#x27;');
}

function updateSemanticOption() {
  const chunkMethodEl = document.getElementById('chunkMethod');
  const hintEl = document.getElementById('chunkMethodHint');
  if (!chunkMethodEl) {
    return;
  }

  const semanticOption = Array.from(chunkMethodEl.options).find((option) => option.value === 'semantic');

  if (!semanticOption) {
    return;
  }

  if (state.semanticChunkingAvailable) {
    semanticOption.disabled = false;
    if (hintEl) {
      hintEl.textContent = DEFAULT_CHUNK_HINT;
    }
  } else {
    semanticOption.disabled = true;
    if (chunkMethodEl.value === 'semantic') {
      chunkMethodEl.value = 'structure';
    }
    if (hintEl) {
      const reason = state.semanticChunkingMessage
        ? `${state.semanticChunkingMessage} Using structure-aware chunking.`
        : 'Semantic chunking requires an embeddings model. Using structure-aware chunking.';
      hintEl.textContent = reason;
    }
  }
}
