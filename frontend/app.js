const state = {
  activeTicker: null,
  lastPrediction: null,
  loading: false,
};

const els = {};

document.addEventListener("DOMContentLoaded", () => {
  cacheElements();
  bindEvents();
  checkApi();
  loadProviders();
  setProviderChip();
});

function cacheElements() {
  Object.assign(els, {
    apiStatus: document.querySelector("#apiStatus"),
    stockForm: document.querySelector("#stockForm"),
    tickerInput: document.querySelector("#tickerInput"),
    forceRetrain: document.querySelector("#forceRetrain"),
    analyzeButton: document.querySelector("#analyzeButton"),
    insightButton: document.querySelector("#insightButton"),
    notice: document.querySelector("#notice"),
    directionCard: document.querySelector("#directionCard"),
    directionValue: document.querySelector("#directionValue"),
    directionMeta: document.querySelector("#directionMeta"),
    priceValue: document.querySelector("#priceValue"),
    priceMeta: document.querySelector("#priceMeta"),
    avgUpValue: document.querySelector("#avgUpValue"),
    avgUpMeta: document.querySelector("#avgUpMeta"),
    voteValue: document.querySelector("#voteValue"),
    voteMeta: document.querySelector("#voteMeta"),
    chartTitle: document.querySelector("#chartTitle"),
    chartMeta: document.querySelector("#chartMeta"),
    priceChart: document.querySelector("#priceChart"),
    providerChip: document.querySelector("#providerChip"),
    insightCopy: document.querySelector("#insightCopy"),
    modelList: document.querySelector("#modelList"),
    cacheChip: document.querySelector("#cacheChip"),
    probGauge: document.querySelector("#probGauge"),
    gaugeValue: document.querySelector("#gaugeValue"),
    providerInputs: [...document.querySelectorAll("input[name='provider']")],
    statusSteps: [...document.querySelectorAll(".status-step")],
    tickerButtons: [...document.querySelectorAll("[data-ticker]")],
  });
}

function bindEvents() {
  els.stockForm.addEventListener("submit", (event) => {
    event.preventDefault();
    runAnalysis(els.tickerInput.value, els.forceRetrain.checked);
  });

  els.insightButton.addEventListener("click", () => {
    runInsight();
  });

  els.providerInputs.forEach((input) => {
    input.addEventListener("change", setProviderChip);
  });

  els.tickerButtons.forEach((button) => {
    button.addEventListener("click", () => {
      els.tickerInput.value = button.dataset.ticker;
      runAnalysis(button.dataset.ticker, false);
    });
  });
}

async function checkApi() {
  try {
    await fetchJson("/health");
    els.apiStatus.textContent = "API ready";
    els.apiStatus.classList.add("ready");
    els.apiStatus.classList.remove("error");
  } catch (error) {
    els.apiStatus.textContent = "API offline";
    els.apiStatus.classList.add("error");
    els.apiStatus.classList.remove("ready");
  }
}

async function loadProviders() {
  try {
    const data = await fetchJson("/providers");
    const configured = Object.entries(data.available)
      .filter(([, isReady]) => isReady)
      .map(([name]) => providerLabel(name));

    if (configured.length > 0) {
      showNotice(`AI ready: ${configured.join(", ")}`, "success", 2600);
    }
  } catch (error) {
    showNotice("Provider status is unavailable right now.", "error");
  }
}

async function runAnalysis(rawTicker, forceRetrain) {
  const ticker = normalizeTicker(rawTicker);
  if (!ticker) {
    showNotice("Enter a ticker symbol first.", "error");
    return;
  }

  state.activeTicker = ticker;
  state.lastPrediction = null;
  setLoading(true);
  resetSteps();
  setStep("history", true);
  renderInsightPlaceholder();
  els.insightButton.disabled = true;
  showNotice(`Analyzing ${ticker}...`, "");

  const encodedTicker = encodeURIComponent(ticker);
  const historyRequest = fetchJson(`/market/${encodedTicker}/history?days=180`);
  const predictRequest = fetchJson(
    `/predict/${encodedTicker}?force_retrain=${forceRetrain ? "true" : "false"}`
  );

  const [historyResult, predictionResult] = await Promise.allSettled([
    historyRequest,
    predictRequest,
  ]);

  if (historyResult.status === "fulfilled") {
    renderHistory(historyResult.value);
  } else {
    renderChartError(historyResult.reason.message);
  }

  setStep("ensemble", true);

  if (predictionResult.status === "fulfilled") {
    renderPrediction(predictionResult.value);
    state.lastPrediction = predictionResult.value;
    els.insightButton.disabled = false;
    showNotice(`${ticker} analysis is ready.`, "success", 2800);
  } else {
    showNotice(predictionResult.reason.message, "error");
    renderPredictionError();
  }

  setLoading(false);
}

async function runInsight() {
  if (!state.activeTicker || !state.lastPrediction) {
    showNotice("Run an analysis before requesting AI insight.", "error");
    return;
  }

  const provider = getSelectedProvider();
  const encodedTicker = encodeURIComponent(state.activeTicker);
  els.insightButton.disabled = true;
  setStep("insight", true);
  els.insightCopy.innerHTML = `<p>Generating ${providerLabel(provider)} insight...</p>`;

  try {
    const data = await fetchJson(`/insight/${encodedTicker}?provider=${provider}`);
    renderInsight(data);
    showNotice(`${providerLabel(provider)} insight is ready.`, "success", 2600);
  } catch (error) {
    els.insightCopy.innerHTML = `<p>${escapeHtml(error.message)}</p>`;
    showNotice(error.message, "error");
  } finally {
    els.insightButton.disabled = false;
  }
}

function renderHistory(data) {
  const series = data.series || [];
  if (series.length < 2) {
    renderChartError("Not enough history to draw a chart.");
    return;
  }

  const width = 820;
  const height = 330;
  const padX = 38;
  const padTop = 24;
  const padBottom = 48;
  const chartHeight = height - padTop - padBottom;
  const closes = series.map((point) => Number(point.close));
  const volumes = series.map((point) => Number(point.volume || 0));
  const min = Math.min(...closes);
  const max = Math.max(...closes);
  const maxVolume = Math.max(...volumes, 1);
  const spread = max - min || 1;
  const last = series[series.length - 1];
  const first = series[0];
  const isUp = Number(last.close) >= Number(first.close);
  const stroke = isUp ? "#137a63" : "#b7443e";
  const fill = isUp ? "rgba(19, 122, 99, 0.14)" : "rgba(183, 68, 62, 0.13)";

  const pointCoords = series.map((point, index) => {
    const x = padX + (index / (series.length - 1)) * (width - padX * 2);
    const y = padTop + ((max - Number(point.close)) / spread) * chartHeight;
    return { x, y };
  });

  const line = pointCoords.map((point) => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(" ");
  const area = `${padX},${height - padBottom} ${line} ${width - padX},${height - padBottom}`;
  const barWidth = Math.max(2, (width - padX * 2) / series.length - 1);
  const bars = series
    .map((point, index) => {
      const x = padX + (index / (series.length - 1)) * (width - padX * 2);
      const barHeight = (Number(point.volume || 0) / maxVolume) * 34;
      const y = height - 16 - barHeight;
      return `<rect x="${(x - barWidth / 2).toFixed(2)}" y="${y.toFixed(2)}" width="${barWidth.toFixed(2)}" height="${barHeight.toFixed(2)}" rx="1.5"></rect>`;
    })
    .join("");

  els.chartTitle.textContent = `${data.ticker} Close`;
  els.chartMeta.textContent = `${series.length} sessions`;
  els.priceChart.innerHTML = `
    <svg class="price-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${escapeHtml(data.ticker)} close chart">
      <rect x="0" y="0" width="${width}" height="${height}" rx="8" fill="#fbfdfb"></rect>
      <g opacity="0.85" fill="#d7dfd7">${bars}</g>
      <path d="M ${area}" fill="${fill}"></path>
      <polyline points="${line}" fill="none" stroke="${stroke}" stroke-width="4" stroke-linecap="round" stroke-linejoin="round"></polyline>
      <circle cx="${pointCoords[pointCoords.length - 1].x.toFixed(2)}" cy="${pointCoords[pointCoords.length - 1].y.toFixed(2)}" r="5.5" fill="${stroke}"></circle>
      <line x1="${padX}" x2="${width - padX}" y1="${height - padBottom}" y2="${height - padBottom}" stroke="#dce5dc" stroke-width="1"></line>
      <text class="chart-axis" x="${padX}" y="18">${formatNumber(max)}</text>
      <text class="chart-axis" x="${padX}" y="${height - padBottom - 6}">${formatNumber(min)}</text>
      <text class="chart-axis" x="${padX}" y="${height - 5}">${escapeHtml(first.date)}</text>
      <text class="chart-axis" x="${width - padX}" y="${height - 5}" text-anchor="end">${escapeHtml(last.date)}</text>
    </svg>
  `;
}

function renderPrediction(data) {
  const direction = data.majority_direction || "UNKNOWN";
  const directionClass = direction === "UP" ? "up" : "down";
  const change = Number(data.change_pct || 0);
  const avgUp = Number(data.avg_up_probability || 0);

  els.directionCard.classList.remove("up", "down");
  els.directionCard.classList.add(directionClass);
  els.directionValue.textContent = direction;
  els.directionMeta.textContent = `${formatPct(data.majority_confidence)} majority confidence`;

  els.priceValue.textContent = formatNumber(data.current_price);
  els.priceMeta.textContent = `${change >= 0 ? "+" : ""}${formatPct(change)} as of ${data.as_of_date}`;

  els.avgUpValue.textContent = formatPct(avgUp);
  els.avgUpMeta.textContent = avgUp >= 55 ? "Bullish lean" : avgUp <= 45 ? "Bearish lean" : "Mixed signal";

  els.voteValue.textContent = `${data.up_votes} / ${data.down_votes}`;
  els.voteMeta.textContent = "UP / DOWN";

  els.cacheChip.textContent = data.trained_fresh ? "Trained fresh" : "Cached model";
  renderGauge(avgUp);
  renderModels(data.per_model || []);
}

function renderModels(models) {
  if (!models.length) {
    els.modelList.innerHTML = `<div class="empty-state">No model output loaded</div>`;
    return;
  }

  els.modelList.innerHTML = models
    .map((model) => {
      const isUp = model.prediction === "UP";
      const upProbability = clamp(Number(model.up_probability || 0), 0, 100);
      return `
        <div class="model-row">
          <div class="model-name">${escapeHtml(model.model)}</div>
          <span class="model-badge ${isUp ? "up" : "down"}">${escapeHtml(model.prediction)}</span>
          <div class="prob-track" aria-label="${escapeHtml(model.model)} up probability">
            <span class="prob-fill ${upProbability < 50 ? "low" : ""}" style="width:${upProbability}%"></span>
          </div>
          <div class="model-score">${formatPct(upProbability)}</div>
        </div>
      `;
    })
    .join("");
}

function renderGauge(value) {
  const pct = clamp(Number(value || 0), 0, 100);
  const degrees = pct * 3.6;
  const color = pct >= 55 ? "#137a63" : pct <= 45 ? "#b7443e" : "#bd7b18";
  els.probGauge.style.background = `conic-gradient(${color} ${degrees}deg, #dfe7df 0deg)`;
  els.gaugeValue.textContent = formatPct(pct);
}

function renderInsight(data) {
  const provider = providerLabel(data.provider_used || getSelectedProvider());
  els.providerChip.textContent = provider;
  els.insightCopy.innerHTML = textToParagraphs(data.insight || "No insight returned.");
}

function renderInsightPlaceholder() {
  setProviderChip();
  els.insightCopy.innerHTML = `<p>Insight appears here after a completed analysis.</p>`;
}

function renderChartError(message) {
  els.chartTitle.textContent = "Recent Close";
  els.chartMeta.textContent = "Unavailable";
  els.priceChart.innerHTML = `<div class="empty-state">${escapeHtml(message)}</div>`;
}

function renderPredictionError() {
  els.directionCard.classList.remove("up", "down");
  els.directionValue.textContent = "Error";
  els.directionMeta.textContent = "Prediction failed";
  els.avgUpValue.textContent = "--";
  els.voteValue.textContent = "--";
  els.cacheChip.textContent = "Cache pending";
  els.modelList.innerHTML = `<div class="empty-state">No model output loaded</div>`;
  renderGauge(0);
}

function setLoading(isLoading) {
  state.loading = isLoading;
  document.body.classList.toggle("loading", isLoading);
  els.analyzeButton.disabled = isLoading;
  els.tickerInput.disabled = isLoading;
  els.forceRetrain.disabled = isLoading;
}

function resetSteps() {
  els.statusSteps.forEach((step) => step.classList.remove("active"));
}

function setStep(name, active) {
  const step = els.statusSteps.find((item) => item.dataset.step === name);
  if (step) {
    step.classList.toggle("active", active);
  }
}

function setProviderChip() {
  els.providerChip.textContent = providerLabel(getSelectedProvider());
}

function getSelectedProvider() {
  return document.querySelector("input[name='provider']:checked")?.value || "openai";
}

function normalizeTicker(value) {
  return String(value || "").trim().toUpperCase();
}

async function fetchJson(url) {
  const response = await fetch(url);
  let payload = null;

  try {
    payload = await response.json();
  } catch (error) {
    payload = null;
  }

  if (!response.ok) {
    throw new Error(readApiError(payload, response.statusText));
  }

  return payload;
}

function readApiError(payload, fallback) {
  if (!payload) {
    return fallback || "Request failed.";
  }

  if (typeof payload.detail === "string") {
    return payload.detail;
  }

  if (Array.isArray(payload.detail)) {
    return payload.detail.map((item) => item.msg || "Validation error").join(" ");
  }

  return fallback || "Request failed.";
}

function showNotice(message, type = "", timeout = 0) {
  els.notice.textContent = message;
  els.notice.className = `notice show ${type}`.trim();

  if (timeout) {
    window.setTimeout(() => {
      if (els.notice.textContent === message) {
        els.notice.className = "notice";
      }
    }, timeout);
  }
}

function providerLabel(value) {
  const provider = String(value || "").toLowerCase();
  if (provider === "gemini") {
    return "Gemini";
  }
  return "OpenAI";
}

function formatNumber(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "--";
  }
  return new Intl.NumberFormat("en-US", {
    maximumFractionDigits: number >= 100 ? 2 : 4,
  }).format(number);
}

function formatPct(value) {
  const number = Number(value);
  if (!Number.isFinite(number)) {
    return "--";
  }
  return `${number.toFixed(2)}%`;
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function escapeHtml(value) {
  return String(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#039;");
}

function textToParagraphs(value) {
  const chunks = String(value || "")
    .split(/\n{2,}|\r?\n/)
    .map((chunk) => chunk.trim())
    .filter(Boolean);

  if (!chunks.length) {
    return "<p>No insight returned.</p>";
  }

  return chunks.map((chunk) => `<p>${escapeHtml(chunk)}</p>`).join("");
}
