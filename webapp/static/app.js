const sliders = Array.from(document.querySelectorAll('input[type="range"]'));
const verdictLabel = document.getElementById('verdict_label');
const verdictScore = document.getElementById('verdict-score');
const meterFill = document.getElementById('meter-fill');
const mlVotes = document.getElementById('ml-votes');
const topFlagCount = document.getElementById('top-flag-count');
const flags = document.getElementById('flags');
const models = document.getElementById('models');

function readPayload() {
  return sliders.reduce((payload, slider) => {
    payload[slider.id] = Number(slider.value);
    return payload;
  }, {});
}

function syncOutputs() {
  sliders.forEach((slider) => {
    const out = document.getElementById(`out_${slider.id}`);
    if (out) out.textContent = slider.value;
  });
}

function applyBootstrapDefaults() {
  const defaults = window.CATFISH_BOOTSTRAP?.defaultProfile || {};
  sliders.forEach((slider) => {
    if (Object.prototype.hasOwnProperty.call(defaults, slider.id)) {
      slider.value = String(defaults[slider.id]);
    }
  });
}

function renderFlags(topFlags) {
  if (!topFlags.length) {
    flags.innerHTML = '<div class="bar-item">No strong red flags detected.</div>';
    return;
  }
  flags.innerHTML = topFlags.map((flag) => {
    const label = flag.name ?? flag[0];
    const value = Number(flag.value ?? flag[1] ?? 0);
    return `
    <div class="bar-item">
      <div class="bar-item-header">
        <strong>${label}</strong>
        <span style="color: var(--neon-pink);">${value.toFixed(1)} pts</span>
      </div>
      <div class="bar-track"><div class="bar-fill flag-fill" style="width:${Math.min(value * 4, 100)}%"></div></div>
    </div>
  `;
  }).join('');
}

function displayModelName(name) {
  return name === 'KMeans + PCA' ? 'KMeans' : name;
}

function renderModels(modelProbs, thresholds = {}, modelDetails = null) {
  if (modelDetails && modelDetails.length) {
    models.innerHTML = modelDetails.map((row) => {
      const isAbove = row.flagged;
      return `
        <div class="bar-item" style="border-color: ${isAbove ? 'rgba(239,68,68,0.55)' : 'var(--glass-border)'}">
          <div class="bar-item-header">
            <strong>${row.name}</strong>
            <span style="color: ${isAbove ? 'var(--danger)' : 'var(--success)'};">${row.probability_pct.toFixed(1)}%</span>
          </div>
          <div class="bar-track">
            <div class="bar-fill" style="width:${Math.max(row.probability * 100, 4)}%; background: ${isAbove ? 'var(--danger)' : 'var(--success)'};"></div>
          </div>
          <div style="font-size: 0.85rem; color: var(--text-main); margin-top: 8px; display:flex; justify-content:space-between;">
            <span>Threshold: <b>${row.threshold.toFixed(3)}</b></span>
            <span style="display:inline-block; padding:4px 10px; border-radius:12px; font-size:11px; color:white; font-weight:800; background: ${isAbove ? 'var(--danger)' : 'var(--success)'}; box-shadow:0 1px 2px rgba(0,0,0,0.2);">${isAbove ? '🚨 CATFISH' : '✅ GENUINE'}</span>
          </div>
        </div>
      `;
    }).join('');
    return;
  }

  models.innerHTML = Object.entries(modelProbs).map(([name, probability]) => {
    const label = displayModelName(name);
    const threshold = thresholds[name] ?? thresholds[label] ?? 0.5;
    const isAbove = probability >= threshold;
    return `
      <div class="bar-item" style="border-color: ${isAbove ? 'rgba(239,68,68,0.55)' : 'var(--glass-border)'}">
        <div class="bar-item-header">
          <strong>${label}</strong>
          <span style="color: ${isAbove ? 'var(--danger)' : 'var(--success)'};">${(probability * 100).toFixed(1)}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${Math.max(probability * 100, 4)}%; background: ${isAbove ? 'var(--danger)' : 'var(--success)'};"></div>
        </div>
        <div style="font-size: 0.85rem; color: var(--text-main); margin-top: 8px; display:flex; justify-content:space-between;">
          <span>Threshold: <b>${threshold.toFixed(3)}</b></span>
          <span style="display:inline-block; padding:4px 10px; border-radius:12px; font-size:11px; color:white; font-weight:800; background: ${isAbove ? 'var(--danger)' : 'var(--success)'}; box-shadow:0 1px 2px rgba(0,0,0,0.2);">${isAbove ? '🚨 CATFISH' : '✅ GENUINE'}</span>
        </div>
      </div>
    `;
  }).join('');
}

function renderScanReport(result) {
  const host = document.getElementById('scan-report');
  if (!host) return;
  if (result.html_report) {
    host.innerHTML = result.html_report;
    return;
  }
  host.innerHTML = '<p style="color:var(--text-main);padding:12px;">Report unavailable — re-run model training export.</p>';
}

let lastVoiceTime = 0;
// Sound Design — lazy-init AudioContext only after first user gesture to comply with browser policy
let audioCtx = null;
function getAudioCtx() {
  if (!audioCtx) {
    try {
      audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    } catch(e) { return null; }
  }
  return audioCtx;
}
function playSound(type) {
  const ctx = getAudioCtx();
  if (!ctx) return;
  if(ctx.state === 'suspended') ctx.resume();
  const osc = ctx.createOscillator();
  const gain = ctx.createGain();
  osc.connect(gain);
  gain.connect(ctx.destination);
  
  if (type === 'hover') {
    osc.type = 'sine'; osc.frequency.setValueAtTime(800, ctx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(1200, ctx.currentTime + 0.1);
    gain.gain.setValueAtTime(0.05, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, ctx.currentTime + 0.1);
    osc.start(); osc.stop(ctx.currentTime + 0.1);
  } else if (type === 'alert') {
    osc.type = 'sawtooth'; osc.frequency.setValueAtTime(150, ctx.currentTime);
    gain.gain.setValueAtTime(0.2, ctx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, ctx.currentTime + 0.5);
    osc.start(); osc.stop(ctx.currentTime + 0.5);
  }
}

// Add hover sounds to buttons
document.addEventListener('mouseover', (e) => {
  if (e.target.tagName === 'BUTTON' || e.target.classList.contains('nav-link')) playSound('hover');
});

function speakWarning() {
  playSound('alert');
  const now = Date.now();
  if (now - lastVoiceTime < 10000) return; // Prevent spamming voice
  lastVoiceTime = now;
  
  if ('speechSynthesis' in window) {
    const msg = new SpeechSynthesisUtterance("Warning. High catfish probability detected.");
    msg.rate = 0.9;
    msg.pitch = 0.8;
    const voices = window.speechSynthesis.getVoices();
    const synthVoice = voices.find(v => v.name.includes("Google") || v.name.includes("English")) || voices[0];
    if (synthVoice) msg.voice = synthVoice;
    window.speechSynthesis.speak(msg);
  }
}

async function scan(cinematic = false) {
  if (!verdictLabel) return;
  syncOutputs();
  
  // Start Cinematic Sequence if Overlay exists
  const overlay = document.getElementById('analysis-overlay');
  const resultCont = document.getElementById('result-container');
  const analysisText = document.getElementById('analysis-text');
  
  if (cinematic && overlay && resultCont) {
    resultCont.style.display = 'none';
    overlay.style.display = 'flex';
    
    // Rotating text
    const phrases = ["ANALYZING INTERACTION PATTERNS...", "DETECTING BEHAVIORAL ANOMALIES...", "CROSS-VALIDATING AUTHENTICITY SIGNALS...", "COMPILING THREAT REPORT..."];
    let step = 0;
    const interval = setInterval(() => {
      step++;
      if (step < phrases.length) analysisText.textContent = phrases[step];
    }, 800);

    // Wait 3 seconds for the cinematic effect
    await new Promise(r => setTimeout(r, 3200));
    clearInterval(interval);
    analysisText.textContent = "INITIALIZING AI SEQUENCE...";
    overlay.style.display = 'none';
    resultCont.style.display = 'block';
  }

  const response = await fetch('/api/scan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(readPayload()),
  });
  
  const result = await response.json();
  
  verdictLabel.textContent = result.verdict_label;
  const threatBar = document.getElementById('threat-bar-fill');
  const score = result.behavioral_score || 0;
  
  if (result.verdict_label && result.verdict_label.includes('CATFISH')) {
    verdictLabel.className = 'verdict-main verdict-catfish';
    if (threatBar) {
      // Use actual behavioral score for accurate threat bar
      threatBar.style.width = `${Math.min(score, 98)}%`;
      threatBar.style.background = score > 70 ? '#ef4444' : '#f59e0b';
    }
    if (score > 60) speakWarning();
  } else {
    verdictLabel.className = 'verdict-main verdict-genuine';
    if (threatBar) {
      threatBar.style.width = `${Math.min(score, 45)}%`;
      threatBar.style.background = score > 30 ? '#f59e0b' : '#10b981';
    }
  }
  
  verdictScore.textContent = `${result.behavioral_score.toFixed(1)}`;
  meterFill.style.width = `${result.behavioral_score}%`;
  
  mlVotes.textContent = `${result.ml_votes}/${Object.keys(result.model_probs).length}`;
  topFlagCount.textContent = String(result.top_flags.length);
  
  renderScanReport(result);
  renderFlags(result.top_flags);
  renderModels(
    result.model_probs,
    result.thresholds || window.CATFISH_BOOTSTRAP?.thresholds || {},
    result.model_details || null,
  );
  
  if (!cinematic) {
    updateScanHistory(result);
  }
}

// --- RADAR CHART LOGIC ---
let radarChartInstance = null;
const featureKeys = [
  'app_usage_time_min', 'swipe_right_ratio', 'bio_length', 
  'message_sent_count', 'profile_pics_count', 'likes_received', 'mutual_matches'
];
const featureLabels = [
  'App Usage', 'Swipe Right %', 'Bio Length', 
  'Messages', 'Photos', 'Likes', 'Matches'
];

function normalize(val, key, stats) {
  const min = stats[key].min || 0;
  const max = stats[key].max || 1;
  return Math.max(0, Math.min(1, (val - min) / (max - min)));
}

function updateRadarChart(payload) {
  const ctx = document.getElementById('radarChart');
  if (!ctx) return;
  
  const bs = window.CATFISH_BOOTSTRAP || {};
  const stats = bs.popStats || {};
  const catfish = bs.catfishProfile || {};
  
  const currentUserNorm = featureKeys.map(k => normalize(payload[k] || 0, k, stats));
  const catfishNorm = featureKeys.map(k => normalize(catfish[k] || 0, k, stats));
  const popAvgNorm = featureKeys.map(k => normalize(stats[k]?.mean || 0, k, stats));

  if (radarChartInstance) {
    radarChartInstance.data.datasets[0].data = popAvgNorm;
    radarChartInstance.data.datasets[1].data = catfishNorm;
    radarChartInstance.data.datasets[2].data = currentUserNorm;
    radarChartInstance.update();
    return;
  }

  radarChartInstance = new Chart(ctx, {
    type: 'radar',
    data: {
      labels: featureLabels,
      datasets: [
        {
          label: 'Population Avg',
          data: popAvgNorm,
          borderColor: 'rgba(0, 240, 255, 0.8)',
          backgroundColor: 'rgba(0, 240, 255, 0.1)',
          borderWidth: 2,
          pointRadius: 0
        },
        {
          label: 'Catfish Median',
          data: catfishNorm,
          borderColor: 'rgba(239, 68, 68, 0.8)',
          borderDash: [5, 5],
          backgroundColor: 'rgba(239, 68, 68, 0.1)',
          borderWidth: 2,
          pointRadius: 0
        },
        {
          label: 'Current Input',
          data: currentUserNorm,
          borderColor: 'rgba(16, 185, 129, 1)',
          backgroundColor: 'rgba(16, 185, 129, 0.4)',
          borderWidth: 3,
          pointBackgroundColor: 'rgba(16, 185, 129, 1)'
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      scales: {
        r: {
          min: 0, max: 1,
          ticks: { display: false },
          grid: { color: 'rgba(255, 255, 255, 0.1)' },
          angleLines: { color: 'rgba(255, 255, 255, 0.1)' },
          pointLabels: { color: 'rgba(255, 255, 255, 0.7)', font: { family: 'Outfit', size: 11 } }
        }
      },
      plugins: {
        legend: { labels: { color: '#fff', font: { family: 'Outfit' } }, position: 'bottom' }
      }
    }
  });
}

// --- SCAN HISTORY LOGIC ---
const scanHistory = [];
function updateScanHistory(result) {
  const tbody = document.getElementById('scan-history-body');
  if (!tbody) return;
  
  const topFlag = result.top_flags && result.top_flags.length > 0 
    ? (result.top_flags[0].name || result.top_flags[0][0]) 
    : 'None';
  
  const isCatfish = result.verdict_label && result.verdict_label.includes('CATFISH');
  const color = isCatfish ? 'rgba(239, 68, 68, 0.1)' : 'rgba(16, 185, 129, 0.1)';
  const textColor = isCatfish ? 'var(--danger)' : 'var(--success)';
  const timeStr = new Date().toLocaleTimeString([], {hour: '2-digit', minute:'2-digit', second:'2-digit'});
  
  const tr = document.createElement('tr');
  tr.style.background = color;
  tr.style.borderBottom = '1px solid var(--glass-border)';
  tr.innerHTML = `
    <td style="padding: 10px; color: var(--text-muted);">${timeStr}</td>
    <td style="padding: 10px; color: ${textColor}; font-weight: bold;">${isCatfish ? 'CATFISH' : 'GENUINE'}</td>
    <td style="padding: 10px; color: #fff;">
      <div style="display:flex; align-items:center; gap:8px;">
        <div style="width: 50px; height: 6px; background: rgba(255,255,255,0.1); border-radius:3px;">
          <div style="height:100%; width:${Math.min(result.behavioral_score, 100)}%; background:${textColor}; border-radius:3px;"></div>
        </div>
        <span>${result.behavioral_score.toFixed(1)}%</span>
      </div>
    </td>
    <td style="padding: 10px; color: var(--text-muted); font-size: 0.8rem;">${topFlag}</td>
  `;
  
  tbody.insertBefore(tr, tbody.firstChild);
  
  // Keep only last 5 scans
  while (tbody.children.length > 5) {
    tbody.removeChild(tbody.lastChild);
  }
}

const clearHistoryBtn = document.getElementById('clear-history-btn');
if (clearHistoryBtn) {
  clearHistoryBtn.addEventListener('click', () => {
    const tbody = document.getElementById('scan-history-body');
    if (tbody) tbody.innerHTML = '';
  });
}

// Ensure radar is drawn
updateRadarChart(readPayload());

if (sliders.length) {
  applyBootstrapDefaults();
  let debounceTimer;
  sliders.forEach((slider) => {
    slider.addEventListener('input', () => {
      syncOutputs();
      updateRadarChart(readPayload());
      clearTimeout(debounceTimer);
      debounceTimer = setTimeout(() => scan(false), 200);
    });
  });
  syncOutputs();
  scan(false);
}

const simBtn = document.getElementById('simulate-btn');
if (simBtn) {
  simBtn.addEventListener('click', () => {
      const catfish = window.CATFISH_BOOTSTRAP?.catfishProfile || {};
      document.getElementById('app_usage_time_min').value = catfish.app_usage_time_min || 800;
      document.getElementById('swipe_right_ratio').value = catfish.swipe_right_ratio || 0.95;
      document.getElementById('bio_length').value = catfish.bio_length || 10;
      document.getElementById('message_sent_count').value = catfish.message_sent_count || 1200;
      document.getElementById('profile_pics_count').value = catfish.profile_pics_count || 1;
      document.getElementById('likes_received').value = catfish.likes_received || 3000;
      document.getElementById('mutual_matches').value = catfish.mutual_matches || 2;
      syncOutputs();
      updateRadarChart(readPayload());
      scan(true);
      const dg = document.querySelector('.dashboard-grid');
      if (dg) { dg.style.boxShadow = '0 0 50px rgba(239, 68, 68, 0.8)'; setTimeout(() => { dg.style.boxShadow = 'none'; }, 2000); }
  });
}

const genBtn = document.getElementById('genuine-btn');
if (genBtn) {
  genBtn.addEventListener('click', () => {
    const genuine = window.CATFISH_BOOTSTRAP?.defaultProfile || {};
    document.getElementById('app_usage_time_min').value = genuine.app_usage_time_min || 60;
    document.getElementById('swipe_right_ratio').value = genuine.swipe_right_ratio || 0.4;
    document.getElementById('bio_length').value = genuine.bio_length || 150;
    document.getElementById('message_sent_count').value = genuine.message_sent_count || 50;
    document.getElementById('profile_pics_count').value = genuine.profile_pics_count || 4;
    document.getElementById('likes_received').value = genuine.likes_received || 50;
    document.getElementById('mutual_matches').value = genuine.mutual_matches || 50;
    syncOutputs();
    updateRadarChart(readPayload());
    scan(true);
    const dg = document.querySelector('.dashboard-grid');
    if (dg) { dg.style.boxShadow = '0 0 50px rgba(16, 185, 129, 0.8)'; setTimeout(() => { dg.style.boxShadow = 'none'; }, 2000); }
  });
}