const sliders = Array.from(document.querySelectorAll('input[type="range"]'));
const outputs = new Map(sliders.map((slider) => [slider.id, slider.parentElement.querySelector('output')]));
const verdictLabel = document.getElementById('verdict-label');
const verdictScore = document.getElementById('verdict-score');
const meterFill = document.getElementById('meter-fill');
const mlVotes = document.getElementById('ml-votes');
const topFlagCount = document.getElementById('top-flag-count');
const flags = document.getElementById('flags');
const models = document.getElementById('models');
const validationBox = document.getElementById('validation');
const validationButton = document.getElementById('run-validation');
const tabButtons = Array.from(document.querySelectorAll('.tab'));
const cells = Array.from(document.querySelectorAll('.cell'));

function readPayload() {
  return sliders.reduce((payload, slider) => {
    payload[slider.id] = Number(slider.value);
    return payload;
  }, {});
}

function syncOutputs() {
  sliders.forEach((slider) => {
    const output = outputs.get(slider.id);
    if (output) {
      output.textContent = slider.value;
    }
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
    flags.innerHTML = '<div class="validation-item">No strong red flags detected.</div>';
    return;
  }
  flags.innerHTML = topFlags.map((flag) => `
    <div class="flag-item">
      <div>
        <strong>${flag.name}</strong>
        <div class="flag-bar"><div style="width:${Math.min(flag.value * 4, 100)}%"></div></div>
      </div>
      <strong>${flag.value.toFixed(1)} pts</strong>
    </div>
  `).join('');
}

function renderModels(modelProbs, thresholds = {}) {
  models.innerHTML = Object.entries(modelProbs).map(([name, probability]) => {
    const threshold = thresholds[name] ?? 0.5;
    const thresholdText = probability >= threshold ? 'above threshold' : 'below threshold';
    return `
      <div class="model-item">
        <div class="model-meta"><strong>${name}</strong><span>${thresholdText}</span></div>
        <div class="meter"><div style="width:${Math.max(probability * 100, 4)}%; background: linear-gradient(90deg, var(--accent), var(--accent-2));"></div></div>
        <div class="model-meta"><span>Threshold ${threshold.toFixed(3)}</span><strong>${(probability * 100).toFixed(1)}%</strong></div>
      </div>
    `;
  }).join('');
}

async function scan() {
  if (!verdictLabel || !meterFill || !mlVotes || !topFlagCount || !flags || !models) {
    return;
  }
  syncOutputs();
  const response = await fetch('/api/scan', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(readPayload()),
  });
  const result = await response.json();
  verdictLabel.textContent = result.verdict_label;
  verdictScore.textContent = `Risk score: ${result.behavioral_score.toFixed(1)}%`;
  meterFill.style.width = `${result.behavioral_score}%`;
  // update ARIA and confidence
  const progress = meterFill.parentElement;
  if (progress) {
    progress.setAttribute('aria-valuenow', String(Math.round(result.behavioral_score)));
  }
  const confidenceEl = document.getElementById('risk-confidence');
  if (confidenceEl) {
    confidenceEl.textContent = `${(result.confidence || result.behavioral_score || 0).toFixed(1)}%`;
  }
  mlVotes.textContent = `${result.ml_votes}/${Object.keys(result.model_probs).length}`;
  topFlagCount.textContent = String(result.top_flags.length);
  renderFlags(result.top_flags);
  renderModels(result.model_probs, result.thresholds || window.CATFISH_BOOTSTRAP?.thresholds || {});
}

async function runValidation() {
  if (!validationBox) { return; }
  validationBox.textContent = 'Running validation suite...';
  const response = await fetch('/api/check');
  const items = await response.json();
  validationBox.innerHTML = items.map((item) => `
    <div class="validation-item" style="margin-bottom:10px; border-color:${item.passed ? 'rgba(16,185,129,0.45)' : 'rgba(239,68,68,0.45)'}">
      <strong>${item.passed ? 'PASS' : 'FAIL'} • ${item.name}</strong>
      <div style="color:var(--muted); margin-top:6px;">${item.details}</div>
    </div>
  `).join('');
}

tabButtons.forEach((button) => {
  button.addEventListener('click', () => {
    tabButtons.forEach((other) => other.classList.toggle('active', other === button));
    const mode = button.dataset.tab;
    cells.forEach((cell) => {
      const isCode = cell.dataset.kind === 'code';
      cell.style.display = mode === 'all' || (mode === 'code' && isCode) ? 'block' : 'none';
    });
  });
});

if (sliders.length) {
  applyBootstrapDefaults();
  sliders.forEach((slider) => slider.addEventListener('input', scan));
  syncOutputs();
  scan();
}

if (validationButton) {
  validationButton.addEventListener('click', runValidation);
}

// Normalize UI text and slider ranges in case an older template is served
function normalizeUI() {
  const heading = document.querySelector('.panel-head h2');
  if (heading && /Live Scanner/i.test(heading.textContent)) {
    heading.textContent = 'Profile Risk Assessment';
  }
  const sub = document.querySelector('.panel-head span');
  if (sub && /Interactive input testing/i.test(sub.textContent)) {
    sub.textContent = 'Interactive profile risk assessment and validation';
  }
  // ensure sliders have professional ranges
  const overrides = {
    app_usage_time_min: [0, 10000, 1],
    bio_length: [0, 10000, 1],
    message_sent_count: [0, 10000, 1],
    likes_received: [0, 10000, 1],
    mutual_matches: [0, 10000, 1],
    profile_pics_count: [0, 100, 1],
    swipe_right_ratio: [0, 1, 0.01],
  };
  Object.entries(overrides).forEach(([id, spec]) => {
    const el = document.getElementById(id);
    if (el) {
      el.min = spec[0]; el.max = spec[1]; el.step = spec[2];
    }
  });
}

normalizeUI();