const form = document.getElementById('search-form');
const input = document.getElementById('image-input');
const preview = document.getElementById('query-preview');
const statusBox = document.getElementById('status');
const button = document.getElementById('search-button');
const resultsSection = document.getElementById('results-section');
const category = document.getElementById('category');
const results = document.getElementById('results');
let previewUrl = null;

function showStatus(message, code = '') {
  statusBox.innerHTML = code ? `<span class="error-code">${escapeHtml(code)}</span> — ${escapeHtml(message)}` : escapeHtml(message);
  statusBox.classList.remove('hidden');
}
function clearStatus() { statusBox.classList.add('hidden'); statusBox.textContent = ''; }
function escapeHtml(value) {
  return String(value).replace(/[&<>'"]/g, ch => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[ch]));
}

input.addEventListener('change', () => {
  const file = input.files?.[0];
  resultsSection.classList.add('hidden');
  results.replaceChildren();
  clearStatus();
  if (!file) { preview.classList.add('hidden'); return; }
  if (!file.type.startsWith('image/')) {
    preview.classList.add('hidden');
    showStatus('The uploaded file is not a supported image.', 'INVALID_IMAGE');
    input.value = '';
    return;
  }
  if (previewUrl) URL.revokeObjectURL(previewUrl);
  previewUrl = URL.createObjectURL(file);
  preview.src = previewUrl;
  preview.classList.remove('hidden');
});

form.addEventListener('submit', async event => {
  event.preventDefault();
  clearStatus();
  resultsSection.classList.add('hidden');
  results.replaceChildren();
  const file = input.files?.[0];
  if (!file) { showStatus('Please select an image first.', 'MISSING_FILE'); return; }

  const k = Number(document.getElementById('k-input').value || 5);
  const data = new FormData();
  data.append('file', file, file.name);
  button.disabled = true;
  button.textContent = 'Searching...';

  try {
    const response = await fetch(`/search?k=${encodeURIComponent(k)}`, { method: 'POST', body: data });
    const body = await response.json().catch(() => null);
    if (!response.ok) {
      const error = body?.error;
      if (error) {
        showStatus(error.message, error.code);
      } else if (Array.isArray(body?.detail)) {
        showStatus(body.detail.map(item => item.msg || 'Invalid request.').join('; '), `HTTP_${response.status}`);
      } else {
        showStatus(`Request failed with status ${response.status}.`, 'HTTP_ERROR');
      }
      return;
    }
    if (!body || !Array.isArray(body.results)) {
      showStatus('The API returned an invalid search response.', 'INTERNAL_ERROR');
      return;
    }
    category.textContent = `Category: ${body.category}`;
    body.results.forEach(item => results.appendChild(createResultCard(item)));
    resultsSection.classList.remove('hidden');
  } catch (error) {
    showStatus('The search service could not be reached. Make sure the API is running.', 'NETWORK_ERROR');
  } finally {
    button.disabled = false;
    button.textContent = 'Search';
  }
});

function createResultCard(item) {
  const card = document.createElement('article');
  card.className = 'result-card';
  const image = document.createElement('img');
  image.alt = `Product ${item.product_id}`;
  image.src = `/result-image/${encodeURIComponent(item.product_id)}`;
  image.onerror = () => { image.style.display = 'none'; };
  const info = document.createElement('div');
  info.className = 'result-info';
  info.innerHTML = `<strong>Product: ${escapeHtml(item.product_id)}</strong><p>Similarity: ${Number(item.similarity).toFixed(1)}%</p>`;
  card.append(image, info);
  return card;
}
