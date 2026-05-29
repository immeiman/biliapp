/**
 * main.js
 * Bilirubin Detection — Tauri Frontend
 * Screen navigation + API calls + camera preview
 */

const API = 'http://127.0.0.1:7878';
const DEFAULT_PREVIEW_POLL_MS = 33;
const DEFAULT_PREVIEW_STATUS_MS = 500;
const CAMERA_CONTROLS_SPACE = 118;
const RISK_BANDS = [
  { min: 17, className: 'sev-err', label: 'TINGGI - perlu evaluasi klinis' },
  { min: 12, className: 'sev-warn', label: 'MENINGKAT - perlu konfirmasi' },
  { min: 0, className: 'sev-ok', label: 'RENDAH - interpretasikan sesuai usia bayi' },
];

// ── State ─────────────────────────────────────────────────────────────────
const state = {
  currentScreen: 'screen-splash',
  cameraTimer: null,
  cameraStatusTimer: null,
  modelMode: 'stage2',
  useStage2: true,
  activeModelId: null,
  activeModelName: null,
  availableModels: [],
  lastImageB64: null,
  lastImageMeta: null,
  lastPrediction: null,
  previewPollMs: DEFAULT_PREVIEW_POLL_MS,
  previewStatusMs: DEFAULT_PREVIEW_STATUS_MS,
  backendStatus: null,
  isCapturing: false,
  lastFocusOk: null,
  lastFocusScore: null,
  screenMetrics: null,
  nativeDisplayMetrics: null,
  gpioAvailable: false,  // true when RPi GPIO is active
  gpioReady: true,       // false = waiting for limit switch to return HIGH
  babies: [],
  activeBaby: null,
  activeBabyId: null,
  syncStatus: null,
  networkStatus: null,
  networkScan: [],
  networkModeDraft: 'hotspot',
};

// ── Runtime viewport measurement ─────────────────────────────────────────
function getScreenMetrics() {
  const vv = window.visualViewport;
  const dpr = window.devicePixelRatio || 1;
  const native = state.nativeDisplayMetrics;
  const cssWidth = Math.round(native?.css_width || vv?.width || window.innerWidth || document.documentElement.clientWidth || screen.width);
  const cssHeight = Math.round(native?.css_height || vv?.height || window.innerHeight || document.documentElement.clientHeight || screen.height);
  const offsetLeft = Math.round(vv?.offsetLeft || 0);
  const offsetTop = Math.round(vv?.offsetTop || 0);
  return {
    css_width: cssWidth,
    css_height: cssHeight,
    physical_width: Math.round(native?.monitor_width || cssWidth * dpr),
    physical_height: Math.round(native?.monitor_height || cssHeight * dpr),
    screen_width: screen.width,
    screen_height: screen.height,
    device_pixel_ratio: native?.scale_factor || dpr,
    offset_left: offsetLeft,
    offset_top: offsetTop,
    orientation: cssWidth >= cssHeight ? 'landscape' : 'portrait',
    native,
  };
}

function applyScreenMetrics() {
  const metrics = getScreenMetrics();
  state.screenMetrics = metrics;

  const root = document.documentElement;
  const controlsSpace = metrics.orientation === 'landscape'
    ? Math.min(CAMERA_CONTROLS_SPACE, Math.max(84, Math.round(metrics.css_height * 0.24)))
    : CAMERA_CONTROLS_SPACE;
  const controlsTop = Math.max(0, metrics.offset_top + metrics.css_height - controlsSpace);

  root.style.setProperty('--app-width', `${metrics.css_width}px`);
  root.style.setProperty('--app-height', `${metrics.css_height}px`);
  root.style.setProperty('--viewport-offset-left', `${metrics.offset_left}px`);
  root.style.setProperty('--viewport-offset-top', `${metrics.offset_top}px`);
  root.style.setProperty('--camera-controls-space', `${controlsSpace}px`);
  root.style.setProperty('--camera-controls-top', `${controlsTop}px`);

  return metrics;
}

function installScreenMetricsWatcher() {
  const update = () => requestAnimationFrame(applyScreenMetrics);
  applyScreenMetrics();
  window.addEventListener('resize', update);
  window.addEventListener('orientationchange', update);
  window.visualViewport?.addEventListener('resize', update);
  window.visualViewport?.addEventListener('scroll', update);
}

async function syncNativeDisplayMetrics() {
  const invoke = window.__TAURI__?.core?.invoke;
  if (!invoke) {
    return null;
  }

  try {
    const metrics = await invoke('sync_display_metrics');
    state.nativeDisplayMetrics = metrics;
    applyScreenMetrics();
    return metrics;
  } catch (err) {
    console.warn('Failed to sync native display metrics:', err);
    return null;
  }
}

// ── API helpers ───────────────────────────────────────────────────────────
async function apiFetch(method, path, body) {
  const opts = { method, headers: { 'Content-Type': 'application/json' } };
  if (body !== undefined) opts.body = JSON.stringify(body);
  const r = await fetch(API + path, opts);
  return r.json();
}
const apiGet  = (path)       => apiFetch('GET',  path);
const apiPost = (path, body) => apiFetch('POST', path, body);
const apiPut  = (path, body) => apiFetch('PUT',  path, body);

async function getBackendStartStatus() {
  try {
    return await window.__TAURI__?.core?.invoke?.('get_backend_status');
  } catch {
    return null;
  }
}

// ── Screen navigation ─────────────────────────────────────────────────────
function showScreen(id, onEnter) {
  applyScreenMetrics();
  document.querySelectorAll('.screen').forEach(s => s.classList.remove('active'));
  state.currentScreen = id;
  document.getElementById(id).classList.add('active');
  if (onEnter) onEnter();
}

// ── Toast ─────────────────────────────────────────────────────────────────
let toastTimer = null;
function toast(msg, duration = 2500) {
  const el = document.getElementById('toast');
  el.textContent = msg;
  el.classList.add('show');
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove('show'), duration);
}

// ── Camera preview ────────────────────────────────────────────────────────
function setFocusState(focusOk, focusScore) {
  const cls = focusOk === true ? 'focus-ok' : focusOk === false ? 'focus-warn' : 'focus-idle';
  state.lastFocusOk = focusOk === true ? true : focusOk === false ? false : null;
  state.lastFocusScore = typeof focusScore === 'number' ? focusScore : null;

  [document.getElementById('camera-wrap'), document.getElementById('focus-reticle')]
    .filter(Boolean)
    .forEach(el => {
      el.classList.remove('focus-ok', 'focus-warn', 'focus-idle');
      el.classList.add(cls);
    });
}

function setCameraStatus(status) {
  const el = document.getElementById('camera-status');
  if (!el) return;

  el.classList.remove('is-visible', 'is-warn', 'is-idle');
  el.textContent = '';

  if (!status) return;

  if (status.busy) {
    el.textContent = 'Kamera sedang capture';
    el.classList.add('is-visible', 'is-idle');
    return;
  }

  if (status.available === false) {
    el.textContent = 'Menunggu kamera...';
    el.classList.add('is-visible', 'is-idle');
    return;
  }

  if (status.fps_ok === false && typeof status.fps === 'number') {
    const fps = Number(status.fps).toFixed(1);
    const minFps = status.min_fps ?? 30;
    el.textContent = `Preview ${fps} FPS, target ${minFps} FPS`;
    el.classList.add('is-visible', 'is-warn');
  }
}

function updateLastThumb() {
  const img = document.getElementById('last-thumb-img');
  const empty = document.getElementById('last-thumb-empty');
  if (!img || !empty) return;

  if (state.lastImageB64) {
    img.src = 'data:image/jpeg;base64,' + state.lastImageB64;
    img.style.display = 'block';
    empty.style.display = 'none';
  } else {
    img.removeAttribute('src');
    img.style.display = 'none';
    empty.style.display = 'block';
  }
}

function setLastImageFromPayload(payload) {
  if (!payload?.image_b64) return false;
  state.lastImageB64 = payload.image_b64;
  state.lastImageMeta = {
    filename: payload.filename ?? payload.image_path?.split(/[\\/]/).pop() ?? null,
    image_path: payload.image_path ?? null,
    width: payload.width ?? null,
    height: payload.height ?? null,
    modified_at: payload.modified_at ?? payload.timestamp ?? null,
  };
  updateLastThumb();
  return true;
}

async function refreshLastImageFromDisk() {
  try {
    const payload = await apiGet('/api/images/latest');
    if (payload?.success && setLastImageFromPayload(payload)) {
      return true;
    }
  } catch {
    // Keep the in-memory image if the backend is temporarily unavailable.
  }
  updateLastThumb();
  return false;
}

function setBabyState(payload) {
  if (!payload) return;
  if (Array.isArray(payload.babies)) {
    state.babies = payload.babies;
  }
  if (Object.prototype.hasOwnProperty.call(payload, 'active_baby')) {
    state.activeBaby = payload.active_baby;
  } else if (Object.prototype.hasOwnProperty.call(payload, 'activeBaby')) {
    state.activeBaby = payload.activeBaby;
  }
  if (Object.prototype.hasOwnProperty.call(payload, 'active_baby_id')) {
    state.activeBabyId = payload.active_baby_id;
  } else {
    state.activeBabyId = state.activeBaby?.baby_id ?? null;
  }
  if (!state.activeBaby && state.activeBabyId != null) {
    state.activeBaby = state.babies.find(b => String(b.baby_id) === String(state.activeBabyId)) ?? null;
  }
  updateBabyUi();
}

function setSyncState(payload) {
  if (!payload) return;
  state.syncStatus = payload;
  updateSyncUi();
}

function updateBabyUi() {
  const nameEl = document.getElementById('active-baby-name');
  const chip = document.getElementById('baby-select-btn');
  const name = state.activeBaby?.baby_name;
  if (nameEl) nameEl.textContent = name || 'Pilih profil bayi';
  if (chip) chip.classList.toggle('is-missing', !state.activeBaby || Number(state.activeBaby?.is_archived || 0) === 1);
  updateCaptureButton();
}

function syncStatusLabel(status) {
  if (!status?.configured) return 'Offline';
  if (status?.skipped && status?.skip_reason === 'internet_unavailable') return 'Tertunda (offline)';
  if (status.failed_count > 0 || status.last_error) return 'Sync error';
  if ((status.pending ?? 0) > 0) return `Pending ${status.pending}`;
  return 'Synced';
}

function updateSyncUi() {
  const el = document.getElementById('sync-chip');
  if (!el) return;
  const label = syncStatusLabel(state.syncStatus);
  el.textContent = label;
  el.classList.remove('sync-offline', 'sync-pending', 'sync-error', 'sync-ok');
  if (label === 'Synced') el.classList.add('sync-ok');
  else if (label.startsWith('Pending')) el.classList.add('sync-pending');
  else if (label === 'Sync error') el.classList.add('sync-error');
  else el.classList.add('sync-offline');
}

function networkModeLabel(mode) {
  const normalized = String(mode || '').trim().toLowerCase();
  if (['wifi', 'wifi_client', 'client', 'internet'].includes(normalized)) return 'WiFi Client';
  if (['hotspot', 'ap'].includes(normalized)) return 'Hotspot';
  return 'Tidak dikenal';
}

function networkModeFromState(status) {
  const value = String(status?.saved_mode || status?.mode || state.networkModeDraft || 'hotspot').toLowerCase();
  return ['wifi', 'wifi_client', 'client', 'internet'].includes(value) ? 'wifi' : 'hotspot';
}

function renderNetworkScanCards(networks) {
  if (!Array.isArray(networks) || networks.length === 0) {
    return `<div class="info-panel">Belum ada hasil scan WiFi.</div>`;
  }

  return networks.map(network => {
    const security = network.security ? esc(network.security) : 'Open';
    const signal = Number.isFinite(Number(network.signal)) ? `${Number(network.signal)}%` : '-';
    const inUse = network.in_use ? '<span class="scan-tag scan-tag-active">Sedang dipakai</span>' : '';
    return `
      <button class="menu-card network-scan-card" type="button" onclick='App.pickNetworkSsid(${JSON.stringify(network.ssid)})'>
        <div class="menu-card-bar"></div>
        <div class="menu-card-body">
          <div class="menu-card-title">${esc(network.ssid)}</div>
          <div class="menu-card-sub">${security} • Signal ${esc(signal)}</div>
        </div>
        <div class="menu-card-arrow">${inUse || '❯'}</div>
      </button>
    `;
  }).join('');
}

function renderNetworkScreen(status, networks) {
  const mode = networkModeFromState(status);
  state.networkModeDraft = mode;
  const hotspotSsid = status?.hotspot_ssid || 'BiliApp-Local';
  const activeSsid = status?.active_ssid || '-';
  const activeConnection = status?.active_connection || '-';
  const ipAddress = status?.ip_address || '-';
  const apiUrl = status?.api_url || '-';
  const statusLabel = status?.status_label || (status?.internet ? 'Online' : networkModeLabel(status?.mode || mode));
  const internet = status?.internet ? 'Tersedia' : 'Tidak tersedia';
  const connectivity = status?.connectivity || 'unknown';
  const lastError = status?.last_error || status?.network_last_error || '-';

  return `
    <div class="card">
      ${infoRow('Status jaringan', statusLabel)}
      ${infoRow('Mode aktif', networkModeLabel(status?.mode || status?.saved_mode || mode))}
      ${infoRow('Internet', internet)}
      ${infoRow('Konektivitas', connectivity)}
      ${infoRow('Koneksi aktif', activeConnection)}
      ${infoRow('SSID aktif', activeSsid)}
      ${infoRow('Hotspot SSID', hotspotSsid)}
      ${infoRow('Alamat IP', ipAddress)}
      ${infoRow('Alamat API', apiUrl)}
      ${infoRow('Terakhir error', lastError)}
    </div>

    <div class="card network-form-card">
      <label class="field-row">
        <span>Mode jaringan</span>
        <select id="network-mode-select" onchange="App.updateNetworkModeView()">
          <option value="hotspot"${mode === 'hotspot' ? ' selected' : ''}>Hotspot</option>
          <option value="wifi"${mode === 'wifi' ? ' selected' : ''}>WiFi Client</option>
        </select>
      </label>

      <div id="network-hotspot-fields" class="network-mode-section">
        <div class="info-panel">SSID hotspot tetap dipakai sebagai UUID lokal. Password bisa diisi ulang jika Anda membuat profile baru.</div>
        <label class="field-row">
          <span>Password hotspot</span>
          <input id="network-hotspot-password" type="password" placeholder="Password hotspot">
        </label>
        <button class="btn btn-primary" style="width:100%; margin-top:8px" onclick="App.applyNetworkMode()">
          Aktifkan Hotspot
        </button>
      </div>

      <div id="network-wifi-fields" class="network-mode-section">
        <label class="field-row">
          <span>SSID WiFi</span>
          <input id="network-wifi-ssid" type="text" placeholder="Pilih dari hasil scan atau ketik manual">
        </label>
        <label class="field-row">
          <span>Password</span>
          <input id="network-wifi-password" type="password" placeholder="Password WiFi">
        </label>
        <div class="button-stack" style="margin-top:8px">
          <button class="btn btn-primary" onclick="App.applyNetworkMode()">Hubungkan WiFi</button>
          <button class="btn btn-secondary" onclick="App.refreshNetworkScan()">Scan Ulang SSID</button>
        </div>
      </div>
    </div>

    <div class="card">
      <div class="sync-panel" style="margin-bottom:8px">
        <div>
          <div class="sync-panel-title">Hasil Scan WiFi</div>
          <div class="sync-panel-sub">Tap SSID untuk mengisi form WiFi</div>
        </div>
        <div class="sync-panel-actions">
          <button class="btn btn-soft" style="height:38px; padding:0 14px" onclick="App.refreshNetworkScan()">Scan</button>
        </div>
      </div>
      <div class="network-scan-list">
        ${renderNetworkScanCards(networks)}
      </div>
    </div>
  `;
}

async function loadBabies() {
  try {
    const payload = await apiGet('/api/babies');
    if (payload?.success) setBabyState(payload);
    return payload;
  } catch {
    updateBabyUi();
    return null;
  }
}

async function loadSyncStatus() {
  try {
    const payload = await apiGet('/api/sync/status');
    setSyncState(payload);
    setBabyState(payload);
    return payload;
  } catch {
    setSyncState({ configured: false, pending: 0, last_error: 'backend offline' });
    return null;
  }
}

function resolutionValue(res) {
  if (!res) return '';
  const width = Array.isArray(res) ? res[0] : res.width;
  const height = Array.isArray(res) ? res[1] : res.height;
  return `${width}x${height}`;
}

function parseResolutionValue(value) {
  const [width, height] = String(value).split('x').map(v => parseInt(v, 10));
  return { width, height };
}

function optionHtml(value, label, selectedValue) {
  const selected = String(value) === String(selectedValue) ? ' selected' : '';
  return `<option value="${value}"${selected}>${label}</option>`;
}

function renderResolutionOptions(selectedValue, presets) {
  const values = presets.includes(selectedValue) ? presets : [selectedValue, ...presets].filter(Boolean);
  return values.map(value => optionHtml(value, value, selectedValue)).join('');
}

function normalizeModelMode(mode) {
  const value = String(mode || '').trim().toLowerCase();
  if (value === 'stage1' || value === 'stage1_only') return 'stage1';
  if (value === 'stage2' || value === 'stage2_only') return 'stage2';
  if (['stage1_stage2_average', 'stage1_stage2', 'stage1+stage2', 'stage1+2', 'stage12', '1+2', 'average', 'ensemble'].includes(value)) {
    return 'stage1_stage2_average';
  }
  return 'stage2';
}

function modelModeLabel(mode) {
  const normalized = normalizeModelMode(mode);
  if (normalized === 'stage1') return 'Stage 1 saja';
  if (normalized === 'stage2') return 'Stage 2 saja';
  return 'Stage 1 + Stage 2';
}

function escapeHtml(value) {
  return String(value ?? '')
    .replaceAll('&', '&amp;')
    .replaceAll('<', '&lt;')
    .replaceAll('>', '&gt;')
    .replaceAll('"', '&quot;')
    .replaceAll("'", '&#39;');
}

function modelRuntimeLabel(modelInfo = {}, fallbackMode = null) {
  return modelInfo.active_model_name
    || modelInfo.active_model_id
    || modelInfo.name
    || modelInfo.id
    || modelModeLabel(fallbackMode);
}

function formatModelMeta(model) {
  const parts = [];
  if (model?.filename) parts.push(model.filename);
  if (model?.format) parts.push(String(model.format).toUpperCase());
  if (model?.size_mb != null) parts.push(`${Number(model.size_mb).toFixed(1)} MB`);
  return parts.join(' · ');
}

function renderModelOptions(models, activeModelId) {
  const list = document.getElementById('model-list');
  if (!list) return;
  if (!models?.length) {
    list.innerHTML = '<div class="empty-state">Tidak ada model regresi di folder models.</div>';
    return;
  }

  list.innerHTML = models.map((model, index) => {
    const checked = model.id === activeModelId || (!activeModelId && index === 0);
    const meta = formatModelMeta(model);
    return `
      <label class="radio-opt card" style="display:flex; gap:0; margin-top:${index === 0 ? 0 : 8}px">
        <div style="width:5px; background:var(--accent-lt); border-radius:10px 0 0 10px; flex-shrink:0"></div>
        <div style="display:flex; align-items:center; gap:12px; padding:14px; min-width:0; flex:1">
          <input type="radio" name="model-id" value="${escapeHtml(model.id)}" ${checked ? 'checked' : ''} style="accent-color:var(--accent); width:18px; height:18px; flex-shrink:0">
          <div style="min-width:0">
            <div style="font-size:14px; color:var(--text); overflow-wrap:anywhere">${escapeHtml(model.name || model.id)}</div>
            <div style="font-size:12px; color:var(--text-sub); margin-top:3px; overflow-wrap:anywhere">${escapeHtml(meta)}</div>
          </div>
        </div>
      </label>`;
  }).join('');
}

function saveModelMode(mode) {
  try {
    localStorage.setItem('bilirubin.modelMode', normalizeModelMode(mode));
  } catch { /* localStorage may be unavailable */ }
}

function loadSavedModelMode() {
  try {
    return localStorage.getItem('bilirubin.modelMode');
  } catch {
    return null;
  }
}

function applyModelModeState(mode) {
  state.modelMode = normalizeModelMode(mode);
  state.useStage2 = state.modelMode !== 'stage1';
}

function updateCaptureButton() {
  const btn = document.getElementById('btn-capture');
  if (!btn) return;
  const gpioBlocked = state.gpioAvailable && !state.gpioReady;
  const babyUnavailable = !state.activeBaby || Number(state.activeBaby?.is_archived || 0) === 1;
  btn.disabled = gpioBlocked || state.isCapturing || babyUnavailable;
  btn.title = babyUnavailable ? 'Pilih profil bayi aktif terlebih dahulu' : '';
}

function startCamera() {
  stopCamera();
  updateLastThumb();
  setFocusState(null, null);
  setCameraStatus(null);
  const img = document.getElementById('cam-img');
  const ph  = document.getElementById('cam-placeholder');
  let streamStarted = false;

  const openPreviewStream = () => {
    if (!img || streamStarted || state.isCapturing) return;
    streamStarted = true;
    img.src = `${API}/api/camera/stream?ts=${Date.now()}`;
  };

  if (img) {
    img.onload = () => {
      img.style.display = 'block';
      ph.style.display = 'none';
    };
    img.onerror = () => {
      img.style.display = 'none';
      ph.style.display = 'flex';
      setCameraStatus({ available: false });
    };
  }

  if (state.isCapturing) {
    setCameraStatus({ busy: true });
  } else {
    openPreviewStream();
  }

  async function tickStatus() {
    if (state.currentScreen !== 'screen-home') return;
    if (state.isCapturing) {
      setCameraStatus({ busy: true });
      state.cameraStatusTimer = setTimeout(tickStatus, state.previewStatusMs);
      return;
    }
    openPreviewStream();
    try {
      const [d, gpioData] = await Promise.all([
        apiGet('/api/camera/preview/status'),
        apiGet('/api/gpio/status').catch(() => null),
      ]);

      // Update GPIO state and handle auto-trigger from limit switch
      if (gpioData) {
        state.gpioAvailable = !!gpioData.available;
        state.gpioReady = gpioData.capture_ready !== false;
        if (gpioData.capture_triggered && state.gpioReady && !state.isCapturing) {
          state.cameraStatusTimer = setTimeout(tickStatus, state.previewStatusMs);
          App.startCapture();
          return;
        }
      }
      updateCaptureButton();

      if (d.available) {
        img.style.display = 'block';
        ph.style.display  = 'none';
        setFocusState(typeof d.focus_ok === 'boolean' ? d.focus_ok : null, d.focus_score);
        setCameraStatus(d);
      } else {
        img.style.display = 'none';
        ph.style.display  = 'flex';
        setFocusState(null, null);
        setCameraStatus(d);
      }

      // GPIO waiting message overrides camera status
      if (state.gpioAvailable && !state.gpioReady) {
        const el = document.getElementById('camera-status');
        if (el) {
          el.textContent = 'Menunggu sensor — lepaskan limit switch (GPIO 8)';
          el.classList.remove('is-warn');
          el.classList.add('is-visible', 'is-idle');
        }
      }
    } catch {
      setFocusState(null, null);
      setCameraStatus({ available: false });
    }
    state.cameraStatusTimer = setTimeout(tickStatus, state.previewStatusMs);
  }
  tickStatus();
}

function stopCamera() {
  clearTimeout(state.cameraTimer);
  clearTimeout(state.cameraStatusTimer);
  state.cameraTimer = null;
  state.cameraStatusTimer = null;
  const img = document.getElementById('cam-img');
  if (img) {
    img.onload = null;
    img.onerror = null;
    img.removeAttribute('src');
    img.style.display = 'none';
  }
  setCameraStatus(null);
}

// ── App public API ────────────────────────────────────────────────────────
const App = {

  // ── Home ────────────────────────────────────────────────────────────────
  goHome() {
    showScreen('screen-home', () => {
      updateLastThumb();
      updateBabyUi();
      loadBabies();
      loadSyncStatus();
      refreshLastImageFromDisk();
      startCamera();
    });
  },

  // ── Menu ────────────────────────────────────────────────────────────────
  goMenu() {
    stopCamera();
    showScreen('screen-menu');
  },

  // ── Capture ─────────────────────────────────────────────────────────────
  async startCapture() {
    if (state.isCapturing) return;
    if (!state.activeBaby || Number(state.activeBaby?.is_archived || 0) === 1) {
      toast('Pilih profil bayi aktif terlebih dahulu');
      this.goBabies();
      return;
    }
    if (state.gpioAvailable && !state.gpioReady) {
      toast('Sensor belum siap — tunggu limit switch kembali ke posisi awal');
      return;
    }
    state.isCapturing = true;
    stopCamera();
    updateCaptureButton();

    // Show capture screen with loading indicator
    showScreen('screen-capture');
    const content = document.getElementById('capture-content');
    content.innerHTML = `
      <div class="capture-loading">
        <div class="mini-spinner"></div>
        Mengambil gambar dan menganalisis…
      </div>`;

    try {
      const result = await apiPost('/api/capture');
      renderCaptureResult(result);
      loadSyncStatus();
    } catch (e) {
      content.innerHTML = `
        <div class="result-card sev-err" style="padding:20px">
          <div style="font-size:18px; font-weight:700; margin-bottom:8px">Koneksi Gagal</div>
          <div style="font-size:13px">${e.message}</div>
        </div>`;
    } finally {
      state.isCapturing = false;
      updateCaptureButton();
      if (state.currentScreen === 'screen-home') {
        startCamera();
      }
    }
  },

  // ── Baby profiles ───────────────────────────────────────────────────────
  async goBabies() {
    stopCamera();
    showScreen('screen-babies', async () => {
      await loadBabies();
      await loadSyncStatus();
      renderBabiesScreen();
    });
  },

  async refreshBabies() {
    const content = document.getElementById('babies-content');
    if (content) content.innerHTML = `<div style="padding:20px; text-align:center; color:var(--text-sub)">Memuat profil dari Supabase...</div>`;
    try {
      const payload = await apiPost('/api/babies/refresh');
      setBabyState(payload);
      setSyncState(await apiGet('/api/sync/status'));
      renderBabiesScreen();
      toast(payload?.success ? 'Profil bayi diperbarui' : (payload?.error || 'Refresh gagal'));
    } catch {
      await loadBabies();
      renderBabiesScreen();
      toast('Refresh gagal, memakai cache lokal');
    }
  },

 async selectBaby(babyId) {
    try {
      const payload = await apiPut('/api/babies/active', { baby_id: String(babyId) });
      if (payload?.success) {
        setBabyState(payload);
        renderBabiesScreen();
        toast(`Bayi aktif: ${payload.active_baby?.baby_name ?? babyId}`);
      } else {
        let errorMsg = payload?.error || 'Gagal memilih bayi';
        // Cegat array error dari FastAPI agar tidak jadi [object Object]
        if (Array.isArray(payload?.detail)) {
            errorMsg = payload.detail[0].msg; 
        } else if (payload?.detail) {
            errorMsg = payload.detail;
        }
        toast(errorMsg);
      }
    } catch (err) {
      toast(err?.message || 'Gagal memilih bayi');
    }
  },

  async runSync() {
    setSyncState({ ...(state.syncStatus || {}), configured: state.syncStatus?.configured, pending: state.syncStatus?.pending ?? 0, last_error: null });
    try {
      const payload = await apiPost('/api/sync/run');
      setSyncState(payload);
      setBabyState(payload);
      await loadBabies();
      if (state.currentScreen === 'screen-babies') renderBabiesScreen();
      toast(payload?.success ? syncStatusLabel(payload) : (payload?.error || 'Sync gagal'));
    } catch {
      setSyncState({ configured: false, pending: state.syncStatus?.pending ?? 0, last_error: 'backend offline' });
      toast('Sync gagal');
    }
  },

  // ── History ──────────────────────────────────────────────────────────────
  async goHistory() {
    showScreen('screen-history', async () => {
      await loadBabies();
      const active = state.activeBaby;
      const query = active ? `baby_id=${encodeURIComponent(active.baby_id)}` : 'all=true';
      const [histRes, statsRes] = await Promise.all([
        apiGet(`/api/history?limit=10&${query}`),
        apiGet(`/api/stats?${query}`),
      ]);

      const filterEl = document.getElementById('history-filter');
      if (filterEl) {
        filterEl.textContent = active
          ? `Profil aktif: ${active.baby_name} (ID ${active.baby_id})`
          : 'Semua data lokal - belum ada bayi aktif';
      }

      // Stats bar
      const stats = statsRes || {};
      const meanStr = (stats.mean_bilirubin != null)
        ? parseFloat(stats.mean_bilirubin).toFixed(2) + ' mg/dL'
        : 'N/A';
      document.getElementById('history-stats').innerHTML = `
        <div class="stat-col"><div class="stat-label">Total</div><div class="stat-value">${stats.total_predictions ?? 0}</div></div>
        <div class="stat-col"><div class="stat-label">Berhasil</div><div class="stat-value">${stats.successful ?? 0}</div></div>
        <div class="stat-col"><div class="stat-label">Gagal</div><div class="stat-value">${stats.failed ?? 0}</div></div>
        <div class="stat-col"><div class="stat-label">Rata-rata</div><div class="stat-value">${meanStr}</div></div>`;

      // Table
      const records = histRes?.records ?? [];
      if (!records.length) {
        document.getElementById('history-table').textContent = 'Belum ada data prediksi.';
        return;
      }
      const header = `${'#'.padEnd(3)} ${'Waktu'.padEnd(19)} ${'mg/dL'.padEnd(7)} ${'Kualitas'.padEnd(8)} Mode\n${'─'.repeat(56)}\n`;
      const rows = records.slice(0, 10).map((r, i) => {
        const ts   = String(r.timestamp ?? 'N/A').slice(0, 19).replace('T', ' ');
        const bili = r.bilirubin_prediction != null ? parseFloat(r.bilirubin_prediction).toFixed(2) : 'N/A';
        const q    = String(r.quality_label ?? 'N/A');
        const m    = String(r.preprocessing_mode ?? 'N/A');
        return `${String(i + 1).padEnd(3)} ${ts.padEnd(19)} ${bili.padEnd(7)} ${q.padEnd(8)} ${m}`;
      }).join('\n');
      document.getElementById('history-table').textContent = header + rows;
    });
  },

  // ── Last image ────────────────────────────────────────────────────────────
  goLastImage() {
    stopCamera();
    showScreen('screen-image', async () => {
      const content = document.getElementById('image-content');
      content.innerHTML = `<div style="padding:20px; text-align:center; color:var(--text-sub)">Memuat foto terakhir...</div>`;
      await refreshLastImageFromDisk();
      if (!state.lastImageB64) {
        content.innerHTML = `<div style="text-align:center; padding:40px; color:var(--text-sub); font-size:14px">
          Belum ada foto.<br>Lakukan Capture terlebih dahulu.</div>`;
        return;
      }
      const meta = state.lastImageMeta ?? {};
      const dims = meta.width && meta.height ? `${meta.width}x${meta.height}` : '-';
      const filename = meta.filename ? esc(meta.filename) : 'Foto Terakhir';
      content.innerHTML = `
        <div class="last-image-view">
          <img src="data:image/jpeg;base64,${state.lastImageB64}" class="image-preview" alt="Last capture" />
        </div>
        <div class="last-image-caption">
          <span>${filename}</span>
          <span>${dims}</span>
        </div>`;
    });
  },

  // ── System info ───────────────────────────────────────────────────────────
  async goSysInfo() {
    showScreen('screen-sysinfo', async () => {
      const content = document.getElementById('sysinfo-content');
      content.innerHTML = `<div style="padding:20px; text-align:center; color:var(--text-sub)">Memuat…</div>`;
      try {
        const s = await apiGet('/api/status');
        const cam = s.camera ?? {};
        const mdl = s.models ?? {};
        const runtime = s.runtime_config ?? {};
        content.innerHTML = buildInfoSections([
          {
            title: 'KAMERA',
            rows: [
              ['Status',    cam.status ?? '?'],
              ['Tipe',      cam.camera_type ?? '?'],
              ['Resolusi',  cam.frame_size ? JSON.stringify(cam.frame_size) : '?'],
              ['FPS',       cam.fps != null ? String(parseFloat(cam.fps).toFixed(0)) : '?'],
            ],
          },
          {
            title: 'MODEL',
            rows: [
              ['Backend',   mdl.model_backend ?? runtime.model_backend ?? '?'],
              ['File',      mdl.active_model_path ? String(mdl.active_model_path).split(/[\\/]/).pop() : '-'],
              ['Preprocess', mdl.preprocess_profile ?? '-'],
              ['Digunakan', modelRuntimeLabel(mdl, runtime.model_mode)],
              ['Latency',   mdl.last_inference_time_ms != null ? `${mdl.last_inference_time_ms} ms` : '-'],
            ],
          },
          {
            title: 'RUNTIME',
            rows: [
              ['Device',    runtime.device_profile ?? 'desktop'],
              ['Preview',   runtime.preview_fps != null ? (runtime.preview_fps === 0 ? 'Auto' : `${runtime.preview_fps} FPS`) : '?'],
              ['Server',    s.initialized ? 'Aktif' : 'Tidak aktif'],
            ],
          },
          {
            title: 'PENYIMPANAN',
            rows: [
              ['Dir. Log',    String(s.logs_directory  ?? '?')],
              ['Dir. Gambar', String(s.images_directory ?? '?')],
              ['Total Foto',  String(s.total_captures  ?? 0)],
            ],
          },
        ]);
      } catch {
        content.innerHTML = `<div style="padding:20px; color:var(--err)">Gagal memuat status sistem.</div>`;
      }
    });
  },

  // ── Settings ──────────────────────────────────────────────────────────────
  goSettings() {
    stopCamera();
    showScreen('screen-settings');
  },

  async goNetworkSettings() {
    showScreen('screen-network', async () => {
      await this.loadNetworkSettings();
    });
  },

  async loadNetworkSettings() {
    const content = document.getElementById('network-content');
    if (!content) return;
    content.innerHTML = `<div style="padding:20px; text-align:center; color:var(--text-sub)">Memuat…</div>`;
    try {
      const [statusResult, scanResult] = await Promise.allSettled([
        apiGet('/api/network/status'),
        apiGet('/api/network/scan'),
      ]);
      const status = statusResult.status === 'fulfilled' && statusResult.value ? statusResult.value : { available: false, mode: 'unknown' };
      const networks = scanResult.status === 'fulfilled' && scanResult.value?.networks ? scanResult.value.networks : [];
      state.networkStatus = status;
      state.networkScan = networks;
      content.innerHTML = renderNetworkScreen(status, networks);
      this.updateNetworkModeView();
    } catch (err) {
      content.innerHTML = `<div style="padding:20px; color:var(--err)">Gagal memuat info jaringan: ${esc(err?.message || err || 'error tidak diketahui')}</div>`;
    }
  },

  updateNetworkModeView() {
    const modeSelect = document.getElementById('network-mode-select');
    const mode = modeSelect?.value || state.networkModeDraft || 'hotspot';
    state.networkModeDraft = mode;

    const hotspotFields = document.getElementById('network-hotspot-fields');
    const wifiFields = document.getElementById('network-wifi-fields');
    if (hotspotFields) hotspotFields.classList.toggle('is-hidden', mode !== 'hotspot');
    if (wifiFields) wifiFields.classList.toggle('is-hidden', mode !== 'wifi');
  },

  pickNetworkSsid(ssid) {
    const modeSelect = document.getElementById('network-mode-select');
    const wifiSsid = document.getElementById('network-wifi-ssid');
    if (modeSelect) modeSelect.value = 'wifi';
    if (wifiSsid) wifiSsid.value = ssid;
    this.updateNetworkModeView();
  },

  async refreshNetworkScan() {
    try {
      const payload = await apiGet('/api/network/scan');
      if (payload?.success) {
        state.networkScan = payload.networks || [];
        if (state.currentScreen === 'screen-network') {
          const content = document.getElementById('network-content');
          if (content) {
            content.innerHTML = renderNetworkScreen(state.networkStatus || {}, state.networkScan);
            this.updateNetworkModeView();
          }
        }
      } else {
        toast(payload?.error || 'Scan jaringan gagal');
      }
    } catch {
      toast('Gagal menghubungi server');
    }
  },

  async applyNetworkMode() {
    const modeSelect = document.getElementById('network-mode-select');
    const mode = modeSelect?.value || state.networkModeDraft || 'hotspot';
    const hotspotPassword = document.getElementById('network-hotspot-password')?.value || '';
    const wifiSsid = document.getElementById('network-wifi-ssid')?.value || '';
    const wifiPassword = document.getElementById('network-wifi-password')?.value || '';
    const hotspotSsid = state.networkStatus?.hotspot_ssid || 'BiliApp-Local';

    if (mode === 'wifi' && !wifiSsid.trim()) {
      toast('Isi SSID WiFi terlebih dahulu');
      return;
    }

    try {
      const payload = mode === 'hotspot'
        ? { mode: 'hotspot', ssid: hotspotSsid, password: hotspotPassword }
        : { mode: 'wifi', ssid: wifiSsid.trim(), password: wifiPassword };
      const resp = await apiPost('/api/network/apply', payload);
      if (resp?.success) {
        state.networkStatus = resp;
        state.networkModeDraft = resp.mode === 'wifi' ? 'wifi' : 'hotspot';
        toast(`✓ Mode jaringan: ${networkModeLabel(resp.mode || mode)}`);
        await this.loadNetworkSettings();
      } else {
        toast(resp?.error || 'Gagal menerapkan mode jaringan');
      }
    } catch (err) {
      toast(err?.message || 'Gagal menghubungi server');
    }
  },

  // ── Camera config ─────────────────────────────────────────────────────────
  async goCameraConfig() {
    showScreen('screen-camera-config', async () => {
      const content = document.getElementById('camera-config-content');
      content.innerHTML = `<div style="padding:20px; text-align:center; color:var(--text-sub)">Memuat…</div>`;
      try {
        const [configResult, devicesResult, statusResult] = await Promise.allSettled([
          apiGet('/api/camera/config'),
          apiGet('/api/camera/devices'),
          apiGet('/api/status'),
        ]);
        if (configResult.status !== 'fulfilled') {
          throw configResult.reason;
        }
        const configResp = configResult.value;
        const devicesResp = devicesResult.status === 'fulfilled' ? devicesResult.value : { devices: [], error: devicesResult.reason?.message };
        const statusResp = statusResult.status === 'fulfilled' ? statusResult.value : { camera: {} };
        const settings = configResp.settings ?? {};
        const devices = devicesResp.devices ?? [];
        const cam = statusResp.camera ?? {};
        const scanWarning = devicesResp.success === false || devicesResult.status !== 'fulfilled'
          ? `<div class="info-panel" style="margin-top:10px">Scan kamera tidak lengkap: ${esc(devicesResp.error || 'kamera sedang dipakai')}</div>`
          : '';
        const previewValue = resolutionValue(settings.preview_resolution);
        const captureValue = resolutionValue(settings.capture_resolution);
        const fpsValue = settings.fps ?? 0;
        const cameraIndex = settings.camera_index ?? 0;
        const deviceOptions = devices.length
          ? devices.map(d => {
              const details = d.width && d.height ? ` - ${d.width}x${d.height}${d.fps ? ` @ ${d.fps} FPS` : ''}` : '';
              return optionHtml(d.index, `${d.name ?? `Camera ${d.index}`}${details}`, cameraIndex);
            }).join('')
          : optionHtml(cameraIndex, `Camera ${cameraIndex}`, cameraIndex);
        content.innerHTML = `
          <div class="card">
            ${infoRow('Status',   cam.status   ?? '?')}
            ${infoRow('Tipe',     cam.camera_type ?? '?')}
            ${infoRow('Resolusi', cam.frame_size ? JSON.stringify(cam.frame_size) : '?')}
            ${infoRow('FPS',      cam.fps != null ? String(parseFloat(cam.fps).toFixed(0)) : '?')}
          </div>
          <div class="card camera-form">
            <label class="field-row">
              <span>Kamera</span>
              <select id="camera-index-select">${deviceOptions}</select>
            </label>
            <label class="field-row">
              <span>Resolusi preview</span>
              <select id="preview-resolution-select">
                ${renderResolutionOptions(previewValue, ['320x240', '640x480', '1280x720'])}
              </select>
            </label>
            <label class="field-row">
              <span>Resolusi capture</span>
              <select id="capture-resolution-select">
                ${renderResolutionOptions(captureValue, ['1280x720', '1920x1080', '3840x2160'])}
              </select>
            </label>
            <label class="field-row">
              <span>FPS</span>
              <select id="camera-fps-select">
                ${optionHtml(0, 'Auto', fpsValue)}
                ${optionHtml(15, '15 FPS', fpsValue)}
                ${optionHtml(24, '24 FPS', fpsValue)}
                ${optionHtml(30, '30 FPS', fpsValue)}
                ${optionHtml(60, '60 FPS', fpsValue)}
              </select>
            </label>
          </div>
          <div class="button-stack">
            <button class="btn btn-primary" onclick="App.saveCameraConfig()">Simpan & Terapkan</button>
            <button class="btn btn-secondary" onclick="App.goCameraConfig()">Scan Ulang Kamera</button>
          </div>
          <button class="btn btn-primary" style="width:100%; margin-top:4px" onclick="App.reconnectCamera()">
            🔄 Sambung Ulang Kamera
          </button>
          <div class="info-panel" style="margin-top:10px">
            Setting hanya tersimpan sementara di memori dan akan hilang saat server di-restart. Untuk perubahan permanen, edit config.py lalu restart server.
          </div>
          ${scanWarning}`;
      } catch (err) {
        content.innerHTML = `<div style="padding:20px; color:var(--err)">Gagal memuat info kamera: ${esc(err?.message || err || 'error tidak diketahui')}</div>`;
      }
    });
  },

  async saveCameraConfig() {
    const cameraIndexEl = document.getElementById('camera-index-select');
    const previewEl = document.getElementById('preview-resolution-select');
    const captureEl = document.getElementById('capture-resolution-select');
    const fpsEl = document.getElementById('camera-fps-select');
    if (!cameraIndexEl || !previewEl || !captureEl || !fpsEl) return;

    const payload = {
      camera_index: parseInt(cameraIndexEl.value, 10),
      preview_resolution: parseResolutionValue(previewEl.value),
      capture_resolution: parseResolutionValue(captureEl.value),
      fps: parseInt(fpsEl.value, 10),
    };

    try {
      const r = await apiPut('/api/camera/config', payload);
      if (r.success) {
        toast('Setting kamera diterapkan');
        await this.goCameraConfig();
      } else {
        toast(r.error || r.detail || 'Gagal menerapkan setting kamera');
      }
    } catch {
      toast('Gagal menghubungi server');
    }
  },

  async reconnectCamera() {
    try {
      const r = await apiPost('/api/camera/reconnect');
      toast(r.success ? '✓ Kamera disambungkan ulang' : '✗ Kamera tidak ditemukan');
    } catch {
      toast('✗ Gagal menghubungi server');
    }
  },

  // ── Model select ──────────────────────────────────────────────────────────
  goModelSelect() {
    showScreen('screen-model', async () => {
      const list = document.getElementById('model-list');
      const msg = document.getElementById('model-msg');
      msg.style.color = 'var(--ok)';
      msg.textContent = '';
      if (list) list.innerHTML = '<div class="empty-state">Memuat daftar model...</div>';
      try {
        const r = await apiGet('/api/settings/model-type');
        if (!r?.success) throw new Error(r?.error || 'Gagal memuat model');
        state.availableModels = r.available || [];
        state.activeModelId = r.active_model_id || r.active_model?.id || null;
        state.activeModelName = r.active_model?.name || state.activeModelName;
        renderModelOptions(state.availableModels, state.activeModelId);
      } catch (err) {
        if (list) list.innerHTML = '<div class="empty-state">Gagal memuat daftar model.</div>';
        msg.style.color = 'var(--err)';
        msg.textContent = err?.message || 'Gagal memuat daftar model';
      }
    });
  },

  async applyModelSettings() {
    const selected = document.querySelector('input[name="model-id"]:checked');
    const modelId = selected?.value;
    const msg = document.getElementById('model-msg');
    if (!modelId) {
      msg.style.color = 'var(--err)';
      msg.textContent = 'Pilih model terlebih dahulu';
      return;
    }
    try {
      msg.style.color = 'var(--text-sub)';
      msg.textContent = 'Memuat model...';
      const r = await apiPost('/api/settings/model-type', { model_id: modelId });
      if (r.success) {
        state.activeModelId = r.active_model_id || modelId;
        state.activeModelName = r.model?.name || state.activeModelName;
        msg.style.color = 'var(--ok)';
        msg.textContent = `Model aktif: ${state.activeModelName || state.activeModelId}`;
      } else {
        msg.style.color = 'var(--err)';
        msg.textContent = r.error || 'Gagal menerapkan pengaturan';
      }
    } catch {
      msg.style.color = 'var(--err)';
      msg.textContent = 'Gagal menghubungi server';
    }
  },

  // ── Logging prefs ─────────────────────────────────────────────────────────
  async goLoggingPrefs() {
    showScreen('screen-logging', async () => {
      const content = document.getElementById('logging-content');
      content.innerHTML = `<div style="padding:20px; text-align:center; color:var(--text-sub)">Memuat…</div>`;
      try {
        const s = await apiGet('/api/status');
        content.innerHTML = `
          <div class="card">
            ${infoRow('Dir. Log',    String(s.logs_directory   ?? '?'))}
            ${infoRow('Dir. Gambar', String(s.images_directory ?? '?'))}
            ${infoRow('Total Foto',  String(s.total_captures  ?? 0))}
            ${infoRow('Format',      'CSV (.csv)')}
          </div>
          <div class="divider" style="margin:12px 0"></div>
          <button class="btn btn-soft" style="width:100%;text-align:left;padding-left:16px" onclick="App.cleanupImages()">
            🗑 Bersihkan Gambar Lama (&gt; 7 hari)
          </button>`;
      } catch {
        content.innerHTML = `<div style="padding:20px; color:var(--err)">Gagal memuat info logging.</div>`;
      }
    });
  },

  // ── Cleanup images ────────────────────────────────────────────────────────
  async cleanupImages() {
    try {
      const r = await apiPost('/api/images/cleanup');
      toast(r.success ? `✓ ${r.deleted} gambar lama dihapus` : '✗ Gagal membersihkan gambar');
    } catch {
      toast('✗ Gagal menghubungi server');
    }
  },

  // ── Exit ──────────────────────────────────────────────────────────────────
  async exitApp() {
    if (window.__TAURI__?.core) {
      try {
        await window.__TAURI__.core.invoke('exit_app');
      } catch {
        window.close();
      }
    } else {
      window.close();
    }
  },
};

// ── Build helpers ─────────────────────────────────────────────────────────

function infoRow(label, value) {
  return `<div class="info-row"><div class="info-label">${label}</div><div class="info-value">${esc(value)}</div></div>`;
}

function buildInfoSections(sections) {
  return sections.map(sec => `
    <div class="section-hdr">${sec.title}</div>
    <div class="card" style="border-radius:0 0 10px 10px; margin-bottom:0">
      ${sec.rows.map(([l, v]) => infoRow(l, v)).join('')}
    </div>`).join('');
}

function esc(str) {
  if (str === null || str === undefined) return '';
  return String(str)
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function classifyBilirubin(value) {
  if (!Number.isFinite(value)) {
    return { className: 'sev-err', label: 'HASIL TIDAK VALID' };
  }
  return RISK_BANDS.find(band => value >= band.min) ?? RISK_BANDS[RISK_BANDS.length - 1];
}

function capturedImageHtml(imageB64) {
  if (!imageB64) {
    return `<div class="capture-image-empty">Foto tidak tersedia</div>`;
  }
  return `<img src="data:image/jpeg;base64,${imageB64}" class="capture-result-image" alt="Foto hasil capture" />`;
}

function compactInfoRows(rows) {
  return rows
    .filter(([, value]) => value !== undefined && value !== null && value !== '')
    .map(([label, value]) => infoRow(label, value))
    .join('');
}

function compactGateList(errors) {
  const visible = errors.slice(0, 3);
  const hiddenCount = Math.max(0, errors.length - visible.length);
  const extra = hiddenCount ? `<li>+${hiddenCount} catatan lain</li>` : '';
  return `<ul class="gate-list compact">${visible.map(e => `<li>${esc(e)}</li>`).join('')}${extra}</ul>`;
}

function formatBabyDob(value) {
  if (!value) return '-';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value).slice(0, 16).replace('T', ' ');
  return d.toLocaleString('id-ID', { dateStyle: 'medium', timeStyle: 'short' });
}

function renderBabiesScreen() {
  const content = document.getElementById('babies-content');
  if (!content) return;
  const babies = state.babies || [];
  const sync = state.syncStatus || {};
  const activeId = state.activeBaby?.baby_id ?? state.activeBabyId;
  const syncLine = `${syncStatusLabel(sync)}${sync.last_error ? ` - ${esc(sync.last_error)}` : ''}`;

  const rows = babies.map(baby => {
    // Pastikan UUID dibandingkan sebagai teks yang sama persis
    const isActive = String(baby.baby_id) === String(activeId); 
    const archived = Number(baby.is_archived || 0) === 1;
    
    // UUID wajib dibungkus tanda kutip satu pada onclick
    return `
      <button class="baby-row ${isActive ? 'is-active' : ''}" ${archived ? 'disabled' : ''} onclick="App.selectBaby('${baby.baby_id}')">
        <div class="baby-row-main">
          <div class="baby-row-name">${esc(baby.baby_name)}</div>
          <div class="baby-row-meta">ID ${esc(baby.baby_id)} · Lahir ${esc(formatBabyDob(baby.baby_dob))}</div>
        </div>
        <div class="baby-row-status">${archived ? 'Archived' : isActive ? 'Aktif' : 'Pilih'}</div>
      </button>`;
  }).join('');

  content.innerHTML = `
    <div class="sync-panel">
      <div>
        <div class="sync-panel-title">Status Sync</div>
        <div class="sync-panel-sub">${syncLine}</div>
      </div>
      <div class="sync-panel-actions">
        <button class="btn btn-secondary" onclick="App.refreshBabies()">Refresh</button>
        <button class="btn btn-primary" onclick="App.runSync()">Sync</button>
      </div>
    </div>
    <div class="baby-list">
      ${rows || '<div class="empty-state">Belum ada profil bayi di cache lokal. Tekan Refresh saat Raspi online.</div>'}
    </div>`;
}

function renderCaptureResult(result) {
  const content = document.getElementById('capture-content');
  if (result?.image_b64) {
    setLastImageFromPayload(result);
  }
  const imageB64 = result?.image_b64 || state.lastImageB64;
  const ts = result?.timestamp ? result.timestamp.slice(0, 19).replace('T', '  ') : '-';
  const qual = result?.quality_label != null
    ? `${String(result.quality_label).toUpperCase()} (${result.quality_score ?? 0}/100)`
    : '-';
  const mode = result?.preprocessing_mode ?? '-';
  const palette = result?.palette_detected ? 'Terdeteksi' : 'Tidak terdeteksi';
  const inference = `${result?.model_backend ?? '?'} / ${result?.active_model_name || result?.active_model_id || modelModeLabel(result?.model_mode || result?.model_used)}`;
  const latency = result?.inference_time_ms != null ? `${Number(result.inference_time_ms).toFixed(1)} ms` : '-';
  const attempt = result?.capture_attempts
    ? `${result.capture_attempt ?? 1}/${result.capture_attempts}`
    : null;
  const babyName = result?.baby_name || state.activeBaby?.baby_name || null;
  const ageHours = result?.age_hours != null ? `${Number(result.age_hours).toFixed(1)} jam` : null;
  const syncInfo = result?.sync_status ? String(result.sync_status) : null;

  if (!result || !result.success) {
    const errMsg = result?.error ?? 'Error tidak diketahui';
    const gateErrors = result?.gatecheck_errors ?? [];
    const gateWarnings = result?.gatecheck_warnings ?? [];
    const title = result?.gatecheck_passed === false ? 'Foto Ditolak' : 'Prediksi Gagal';
    const helper = result?.gatecheck_passed === false
      ? 'Pastikan kartu kalibrasi, color palette, dan area kulit terlihat jelas.'
      : 'Periksa status kamera dan model, lalu coba capture ulang.';
    const detail = gateErrors.length
      ? compactGateList(gateErrors)
      : `<div class="result-compact-text">${esc(errMsg)}</div>`;
    const warnings = gateWarnings.length
      ? `<div class="gate-warn">${gateWarnings.slice(0, 2).map(esc).join('<br>')}</div>`
      : '';
    content.innerHTML = `
      <div class="capture-result-layout">
        <div class="capture-image-pane">${capturedImageHtml(imageB64)}</div>
        <div class="capture-summary-pane">
          <div class="result-card compact sev-err">
            <div class="result-status">Gagal</div>
            <div class="result-title">${title}</div>
            ${detail}
            ${warnings}
            <div class="result-helper">${helper}</div>
          </div>
          <div class="card result-detail-card">
            ${compactInfoRows([
              ['Bayi', babyName],
              ['Waktu', ts],
              ['Usia', ageHours],
              ['Kualitas', qual],
              ['Palette', palette],
              ['Mode', mode],
              ['Percobaan', attempt],
              ['Sync', syncInfo],
            ])}
          </div>
        </div>
      </div>`;
    return;
  }

  const bili = Number.parseFloat(result.bilirubin_prediction);
  if (!Number.isFinite(bili)) {
    content.innerHTML = `
      <div class="capture-result-layout">
        <div class="capture-image-pane">${capturedImageHtml(imageB64)}</div>
        <div class="capture-summary-pane">
          <div class="result-card compact sev-err">
            <div class="result-status">Gagal</div>
            <div class="result-title">Prediksi Gagal</div>
            <div class="result-compact-text">Nilai bilirubin dari server tidak valid.</div>
          </div>
        </div>
      </div>`;
    return;
  }
  const risk = classifyBilirubin(bili);
  const sevClass = risk.className;
  const level = risk.label;

  const rawAlignedReason = result.palette_detected
    ? 'mode raw_aligned - koreksi warna tidak diterapkan karena kualitas kalibrasi belum cukup stabil'
    : 'palette tidak terdeteksi - mode raw_aligned';
  const rawAlignedBanner = mode === 'raw_aligned'
    ? `<div class="result-mode-warn">Peringatan: ${rawAlignedReason}, akurasi prediksi lebih rendah</div>`
    : '';
  const logWarnBanner = result.log_warning
    ? `<div class="result-mode-warn">Peringatan: ${esc(result.log_warning)}</div>`
    : '';
  const offlineWarnBanner = result.offline_warning
    ? `<div class="result-mode-warn">Peringatan: ${esc(result.offline_warning)}</div>`
    : '';

  content.innerHTML = `
    <div class="capture-result-layout">
      <div class="capture-image-pane">${capturedImageHtml(imageB64)}</div>
      <div class="capture-summary-pane">
        <div class="result-card compact ${sevClass}">
          <div class="result-status">Berhasil</div>
          <div class="result-num">${bili.toFixed(2)}</div>
          <div class="result-unit">mg/dL</div>
          <div class="result-level">${level}</div>
        </div>
        ${rawAlignedBanner}
        ${logWarnBanner}
        ${offlineWarnBanner}
        <div class="card result-detail-card">
          ${compactInfoRows([
            ['Bayi', babyName],
            ['Waktu', ts],
            ['Usia', ageHours],
            ['Kualitas', qual],
            ['Palette', palette],
            ['Mode', mode],
            ['Inferensi', inference],
            ['Latency', latency],
            ['Sync', syncInfo],
          ])}
        </div>
      </div>
    </div>`;

  state.lastPrediction = bili;
}

// ── Startup — wait for server ─────────────────────────────────────────────
async function waitForServer() {
  const statusEl = document.getElementById('splash-status');
  let attempts = 0;
  const maxAttempts = 60; // 30 seconds

  while (attempts < maxAttempts) {
    attempts++;
    statusEl.textContent = `Menghubungkan ke server… (${attempts})`;
    try {
      const s = await apiGet('/api/status');
      if (s) {
        const runtime = s.runtime_config ?? {};
        const models = s.models ?? {};
        state.previewPollMs = runtime.preview_poll_ms ?? DEFAULT_PREVIEW_POLL_MS;
        state.previewStatusMs = Math.max(DEFAULT_PREVIEW_STATUS_MS, runtime.preview_poll_ms ?? DEFAULT_PREVIEW_STATUS_MS);
        applyModelModeState(models.model_mode ?? runtime.model_mode ?? (runtime.use_stage2 ? 'stage2' : 'stage1'));

        state.activeModelId = models.active_model_id ?? runtime.active_model_id ?? null;
        state.activeModelName = models.active_model_name ?? runtime.active_model_name ?? null;
        statusEl.textContent = '✓ Terhubung!';
        await loadBabies();
        await loadSyncStatus();
        await new Promise(r => setTimeout(r, 400));
        App.goHome();
        return;
      }
    } catch { /* server not ready */ }
    await new Promise(r => setTimeout(r, 500));
  }

  state.backendStatus = await getBackendStartStatus();
  if (state.backendStatus?.error) {
    statusEl.textContent = `Server gagal start: ${state.backendStatus.error}`;
  } else {
    statusEl.textContent = '✗ Server tidak merespons. Coba restart aplikasi.';
  }
}

// ── Init ──────────────────────────────────────────────────────────────────
window.App = App;
window.getScreenMetrics = getScreenMetrics;
window.syncNativeDisplayMetrics = syncNativeDisplayMetrics;

// Append toast element
const toastEl = document.createElement('div');
toastEl.id = 'toast';
document.body.appendChild(toastEl);

// Keyboard shortcuts (F11 / Escape for fullscreen toggle via Tauri)
document.addEventListener('keydown', e => {
  if (e.key === 'F11') e.preventDefault();
  if (e.key === 'Escape' && state.currentScreen === 'screen-home') {
    // do nothing on home
  }
});

installScreenMetricsWatcher();
syncNativeDisplayMetrics().finally(waitForServer);
