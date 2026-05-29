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
  flags.innerHTML = topFlags.map((flag) => `
    <div class="bar-item">
      <div class="bar-item-header">
        <strong>${flag.name}</strong>
        <span style="color: var(--neon-pink);">${flag.value.toFixed(1)} pts</span>
      </div>
      <div class="bar-track"><div class="bar-fill flag-fill" style="width:${Math.min(flag.value * 4, 100)}%"></div></div>
    </div>
  `).join('');
}

function renderModels(modelProbs, thresholds = {}) {
  models.innerHTML = Object.entries(modelProbs).map(([name, probability]) => {
    const threshold = thresholds[name] ?? 0.5;
    const isAbove = probability >= threshold;
    return `
      <div class="bar-item" style="border-color: ${isAbove ? 'rgba(239,68,68,0.4)' : 'var(--glass-border)'}">
        <div class="bar-item-header">
          <strong>${name}</strong>
          <span style="color: ${isAbove ? 'var(--danger)' : 'var(--success)'};">${(probability * 100).toFixed(1)}%</span>
        </div>
        <div class="bar-track">
          <div class="bar-fill" style="width:${Math.max(probability * 100, 4)}%; background: ${isAbove ? 'var(--danger)' : 'var(--success)'}; box-shadow: 0 0 10px ${isAbove ? 'var(--danger)' : 'var(--success)'};"></div>
        </div>
        <div style="font-size: 0.8rem; color: var(--text-muted); margin-top: 6px; text-align: right;">Threshold: ${threshold.toFixed(3)}</div>
      </div>
    `;
  }).join('');
}

let lastVoiceTime = 0;
// Sound Design
const audioCtx = new (window.AudioContext || window.webkitAudioContext)();
function playSound(type) {
  if(audioCtx.state === 'suspended') audioCtx.resume();
  const osc = audioCtx.createOscillator();
  const gain = audioCtx.createGain();
  osc.connect(gain);
  gain.connect(audioCtx.destination);
  
  if (type === 'hover') {
    osc.type = 'sine'; osc.frequency.setValueAtTime(800, audioCtx.currentTime);
    osc.frequency.exponentialRampToValueAtTime(1200, audioCtx.currentTime + 0.1);
    gain.gain.setValueAtTime(0.05, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + 0.1);
    osc.start(); osc.stop(audioCtx.currentTime + 0.1);
  } else if (type === 'alert') {
    osc.type = 'sawtooth'; osc.frequency.setValueAtTime(150, audioCtx.currentTime);
    gain.gain.setValueAtTime(0.2, audioCtx.currentTime);
    gain.gain.exponentialRampToValueAtTime(0.01, audioCtx.currentTime + 0.5);
    osc.start(); osc.stop(audioCtx.currentTime + 0.5);
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

async function scan() {
  if (!verdictLabel) return;
  syncOutputs();
  
  // Start Cinematic Sequence if Overlay exists
  const overlay = document.getElementById('analysis-overlay');
  const resultCont = document.getElementById('result-container');
  const analysisText = document.getElementById('analysis-text');
  
  if (overlay && resultCont) {
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
  
  if (result.verdict_label.includes('CATFISH')) {
    verdictLabel.className = 'verdict-main verdict-catfish';
    if (threatBar) { threatBar.style.width = '90%'; threatBar.style.background = '#ef4444'; }
    if (result.behavioral_score > 80) speakWarning();
  } else {
    verdictLabel.className = 'verdict-main verdict-genuine';
    if (threatBar) {
      if (result.behavioral_score > 40) { threatBar.style.width = '50%'; threatBar.style.background = '#f59e0b'; }
      else { threatBar.style.width = '15%'; threatBar.style.background = '#10b981'; }
    }
  }
  
  verdictScore.textContent = `${result.behavioral_score.toFixed(1)}`;
  meterFill.style.width = `${result.behavioral_score}%`;
  
  mlVotes.textContent = `${result.ml_votes}/${Object.keys(result.model_probs).length}`;
  topFlagCount.textContent = String(result.top_flags.length);
  
  renderFlags(result.top_flags);
  renderModels(result.model_probs, result.thresholds || window.CATFISH_BOOTSTRAP?.thresholds || {});
}

if (sliders.length) {
  applyBootstrapDefaults();
  sliders.forEach((slider) => slider.addEventListener('input', scan));
  syncOutputs();
  scan();
}

const simBtn = document.getElementById('simulate-btn');
if (simBtn) {
  simBtn.addEventListener('click', () => {
      // Example of a realistic but clearly manipulative Catfish profile
      document.getElementById('app_usage_time_min').value = 750; // 12.5 hours
      document.getElementById('swipe_right_ratio').value = 0.99; // Swiping right on almost everyone
      document.getElementById('bio_length').value = 5; // Extremely short bio
      document.getElementById('message_sent_count').value = 850; // Spamming messages
      document.getElementById('profile_pics_count').value = 1; // Only 1 picture
      document.getElementById('likes_received').value = 4500; // Getting a ton of likes
      document.getElementById('mutual_matches').value = 2; // But hardly anyone matches back
      scan();
    document.querySelector('.dashboard-grid').style.boxShadow = '0 0 50px rgba(239, 68, 68, 0.8)';
    setTimeout(() => { document.querySelector('.dashboard-grid').style.boxShadow = 'none'; }, 2000);
  });
}

const genBtn = document.getElementById('genuine-btn');
if (genBtn) {
  genBtn.addEventListener('click', () => {
    // Example of a completely normal, healthy Genuine profile based on dataset medians
    const profile = window.CATFISH_BOOTSTRAP.defaultProfile;
    document.getElementById('app_usage_time_min').value = profile.app_usage_time_min;
    document.getElementById('swipe_right_ratio').value = profile.swipe_right_ratio;
    document.getElementById('bio_length').value = profile.bio_length;
    document.getElementById('message_sent_count').value = profile.message_sent_count;
    document.getElementById('profile_pics_count').value = profile.profile_pics_count;
    document.getElementById('likes_received').value = profile.likes_received;
    document.getElementById('mutual_matches').value = profile.mutual_matches;
    scan();
    document.querySelector('.dashboard-grid').style.boxShadow = '0 0 50px rgba(16, 185, 129, 0.8)';
    setTimeout(() => { document.querySelector('.dashboard-grid').style.boxShadow = 'none'; }, 2000);
  });
}