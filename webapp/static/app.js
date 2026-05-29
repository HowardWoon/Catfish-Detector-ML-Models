const sliders = Array.from(document.querySelectorAll('input[type="range"]'));
const verdictLabel = document.getElementById('verdict-label');
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

async function scan() {
  if (!verdictLabel) return;
  syncOutputs();
  
  const response = await fetch('/api/scan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(readPayload()),
  });
  
  const result = await response.json();
  
  verdictLabel.textContent = result.verdict_label;
  if (result.verdict_label.includes('CATFISH')) {
    verdictLabel.className = 'verdict-main verdict-catfish';
  } else {
    verdictLabel.className = 'verdict-main verdict-genuine';
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