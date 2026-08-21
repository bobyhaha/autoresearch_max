const DEFAULT_LOG_URL = "../remote_training_runs/20260705_190741/live.log";
const WS_RECONNECT_MS = 1500;

const state = {
  logText: "",
  parsed: emptyParsed(),
  frame: 0,
  playing: true,
  live: true,
  manual: false,
  speed: 1,
  lastFetchSize: 0,
  stickLog: true,
  sourceUrl: DEFAULT_LOG_URL,
  socket: null,
  structuredUrl: "",
  structuredPayload: null,
  filters: {
    search: "",
    scope: "all",
    layer: "all",
    metric: "all",
  },
};

const els = {
  trainingChart: document.querySelector("#trainingChart"),
  overviewChart: document.querySelector("#overviewChart"),
  observableGrid: document.querySelector("#observableGrid"),
  commandLog: document.querySelector("#commandLog"),
  logFile: document.querySelector("#logFile"),
  logUrl: document.querySelector("#logUrl"),
  loadUrl: document.querySelector("#loadUrl"),
  scrollLog: document.querySelector("#scrollLog"),
  statStep: document.querySelector("#statStep"),
  statProgress: document.querySelector("#statProgress"),
  statTrain: document.querySelector("#statTrain"),
  statVal: document.querySelector("#statVal"),
  statTok: document.querySelector("#statTok"),
  statMfu: document.querySelector("#statMfu"),
  observableSearch: document.querySelector("#observableSearch"),
  observableScope: document.querySelector("#observableScope"),
  observableLayer: document.querySelector("#observableLayer"),
  observableMetric: document.querySelector("#observableMetric"),
  observableCount: document.querySelector("#observableCount"),
};

els.logUrl.value = DEFAULT_LOG_URL;

function emptyParsed() {
  return {
    steps: [],
    train: [],
    val: [],
    valByStep: new Map(),
    observables: new Map(),
    final: {},
    run: {},
  };
}

function parseLog(text) {
  const parsed = emptyParsed();
  const trainRe = /^step\s+(\d+)\s+\(([\d.]+)%\)\s+\|\s+train_loss:\s+([-+\deE.]+).*?tok\/sec:\s+([\d,]+).*?mfu:\s+([-+\deE.]+)%.*?epoch:\s+(-?\d+)/;
  const valRe = /^val_loss:\s+([-+\deE.]+)\s+step:\s+(\d+)\s+epoch:\s+(-?\d+)/;
  const finalRe = /^(val_bpb|training_seconds|total_seconds|peak_vram_mb|mfu_percent|total_tokens_M|num_steps|num_params_M|depth):\s+([-+\deE.]+)/;
  const dataRe = /^Data split (train|test) ids:\s+(.+)/;

  for (const line of text.split(/\r?\n/)) {
    const train = line.match(trainRe);
    if (train) {
      const step = Number(train[1]);
      parsed.steps.push(step);
      parsed.train.push({
        step,
        progress: Number(train[2]),
        value: Number(train[3]),
        tokPerSec: Number(train[4].replace(/,/g, "")),
        mfu: Number(train[5]),
        epoch: Number(train[6]),
      });
      continue;
    }

    const val = line.match(valRe);
    if (val) {
      const point = { step: Number(val[2]), value: Number(val[1]), epoch: Number(val[3]) };
      parsed.val.push(point);
      parsed.valByStep.set(point.step, point.value);
      continue;
    }

    if (line.startsWith("observable: ")) {
      const obs = parseKeyValueLine(line.slice("observable: ".length));
      if (Number.isFinite(obs.step)) {
        for (const [key, value] of Object.entries(obs)) {
          if (key === "step" || key === "epoch" || !Number.isFinite(value)) continue;
          if (!parsed.observables.has(key)) parsed.observables.set(key, []);
          parsed.observables.get(key).push({ step: obs.step, value });
        }
      }
      continue;
    }

    const final = line.match(finalRe);
    if (final) {
      parsed.final[final[1]] = Number(final[2]);
      continue;
    }

    const data = line.match(dataRe);
    if (data) {
      parsed.run[data[1]] = data[2];
    }
  }

  return parsed;
}

function parseKeyValueLine(payload) {
  const values = {};
  for (const item of payload.trim().split(/\s+/)) {
    const idx = item.indexOf("=");
    if (idx < 0) continue;
    values[item.slice(0, idx)] = Number(item.slice(idx + 1));
  }
  return values;
}

function setLogText(text) {
  state.logText = text;
  state.parsed = parseLog(text);
  if (state.live || state.frame >= state.parsed.train.length - 2) {
    state.frame = Math.max(0, state.parsed.train.length - 1);
  } else {
    state.frame = Math.min(state.frame, Math.max(0, state.parsed.train.length - 1));
  }
  renderAll();
}

async function fetchLog() {
  try {
    const response = await fetch(state.sourceUrl, { cache: "no-store" });
    if (!response.ok) throw new Error(`${response.status} ${response.statusText}`);
    const text = await response.text();
    if (text.length !== state.lastFetchSize || text !== state.logText) {
      state.lastFetchSize = text.length;
      setLogText(text);
    }
    fetchStructuredCurves();
  } catch (error) {
    console.warn(`Could not load ${state.sourceUrl}`, error);
  }
}

function connectLogSocket() {
  if (state.socket) {
    state.socket.close();
  }
  const protocol = window.location.protocol === "https:" ? "wss:" : "ws:";
  const socket = new WebSocket(`${protocol}//${window.location.host}/ws/logs`);
  state.socket = socket;

  socket.addEventListener("message", (event) => {
    if (state.manual) return;
    let payload;
    try {
      payload = JSON.parse(event.data);
    } catch (error) {
      console.warn("Bad log socket payload", error);
      return;
    }
    const sourceUrl = payload.source_url || DEFAULT_LOG_URL;
    if (sourceUrl !== state.sourceUrl) {
      setSourceUrl(sourceUrl);
      state.logText = "";
      state.parsed = emptyParsed();
      state.structuredUrl = "";
      state.structuredPayload = null;
      state.frame = 0;
      renderAll();
    }
    state.live = true;
    state.playing = true;
    if (payload.log_text !== state.logText) {
      state.lastFetchSize = payload.log_text.length;
      setLogText(payload.log_text);
    }
    fetchStructuredCurves();
  });

  socket.addEventListener("close", () => {
    if (state.socket === socket) {
      setTimeout(connectLogSocket, WS_RECONNECT_MS);
    }
  });

  socket.addEventListener("error", () => {
    socket.close();
  });
}

async function postJson(url) {
  const response = await fetch(url, { method: "POST" });
  const payload = await response.json();
  if (!response.ok || !payload.ok) {
    throw new Error(payload.message || `${response.status} ${response.statusText}`);
  }
  return payload;
}

function setSourceUrl(url) {
  state.sourceUrl = url || DEFAULT_LOG_URL;
  els.logUrl.value = state.sourceUrl;
  state.lastFetchSize = 0;
  state.structuredUrl = "";
  state.structuredPayload = null;
}

function visibleParsed() {
  const parsed = mergeStructuredCurves(state.parsed, state.structuredPayload);
  const limit = Math.max(0, Math.min(state.frame + 1, state.parsed.train.length));
  const maxStep = state.parsed.train[limit - 1]?.step ?? Number.POSITIVE_INFINITY;
  return {
    train: parsed.train.slice(0, limit),
    val: parsed.val.filter((point) => point.step <= maxStep),
    observables: new Map(
      [...parsed.observables.entries()].map(([key, points]) => [
        key,
        points.filter((point) => point.step <= maxStep),
      ])
    ),
    final: parsed.final,
  };
}

async function fetchStructuredCurves() {
  const url = inferStructuredCurvesUrl();
  if (!url || state.structuredPayload || state.structuredUrl === url) return;
  state.structuredUrl = url;
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      state.structuredUrl = "";
      return;
    }
    state.structuredPayload = await response.json();
    renderAll();
  } catch (error) {
    state.structuredUrl = "";
  }
}

function inferStructuredCurvesUrl() {
  const match = state.sourceUrl.match(/^(.*remote_training_runs\/([^/]+))\/live\.log$/);
  if (!match) return "";
  return `${match[1]}/fetched/${match[2]}/observable_curves.json`;
}

function mergeStructuredCurves(parsed, payload) {
  if (!payload || typeof payload !== "object") return parsed;
  const merged = {
    train: parsed.train,
    val: parsed.val,
    observables: new Map(parsed.observables),
    final: parsed.final,
  };
  const series = payload.series && typeof payload.series === "object" ? payload.series : {};
  for (const [key, points] of Object.entries(series)) {
    if (!Array.isArray(points)) continue;
    merged.observables.set(
      key,
      points
        .map((point) => ({ step: Number(point.step), value: Number(point.value) }))
        .filter((point) => Number.isFinite(point.step) && Number.isFinite(point.value))
    );
  }
  return merged;
}

function renderAll() {
  const visible = visibleParsed();
  renderStats(visible);
  renderLog();
  drawTrainingChart(els.trainingChart, visible.train, visible.val);
  drawTrainingChart(els.overviewChart, visible.train, visible.val, { compact: true });
  renderObservableFilters(visible);
  renderObservableGrid(visible);
}

function renderStats(visible) {
  const last = visible.train.at(-1);
  const lastVal = visible.val.at(-1);
  els.statStep.textContent = last ? String(last.step) : "-";
  els.statProgress.textContent = last ? `${last.progress.toFixed(1)}%` : "-";
  els.statTrain.textContent = last ? last.value.toFixed(4) : "-";
  els.statVal.textContent = lastVal ? lastVal.value.toFixed(4) : "-";
  els.statTok.textContent = last ? `${Math.round(last.tokPerSec / 1000)}k/s` : "-";
  els.statMfu.textContent = last ? `${last.mfu.toFixed(1)}%` : "-";

}

function renderLog() {
  els.commandLog.textContent = state.logText || "No log loaded yet.";
  if (state.stickLog) {
    els.commandLog.scrollTop = els.commandLog.scrollHeight;
  }
}

function drawTrainingChart(canvas, train, val, opts = {}) {
  const series = [
    { name: "train", points: train.map((p) => ({ x: p.step, y: p.value })), color: cssVar("--train") },
    { name: "val", points: val.map((p) => ({ x: p.step, y: p.value })), color: cssVar("--val") },
  ];
  drawLineChart(canvas, series, opts);
}

function renderObservableGrid(visible) {
  const preferred = [
    "raw_train_loss",
    "train_val_gap",
    "tok_per_sec_k",
    "mfu_percent",
    "step_time_ms",
    "lrm_muon",
    "lrm_adam",
    "muon_momentum",
    "muon_weight_decay",
  ];
  const keys = [...visible.observables.keys()]
    .filter((key) => matchesObservableFilters(key))
    .sort((a, b) => {
      const ai = preferred.indexOf(a);
      const bi = preferred.indexOf(b);
      if (ai !== -1 || bi !== -1) return (ai === -1 ? 999 : ai) - (bi === -1 ? 999 : bi);
      return a.localeCompare(b);
    });
  els.observableCount.textContent = `${keys.length} / ${visible.observables.size} series`;

  const currentKeys = [...els.observableGrid.querySelectorAll(".mini-card")].map((node) => node.dataset.key);
  if (currentKeys.join("\n") !== keys.join("\n")) {
    els.observableGrid.innerHTML = "";
    for (const key of keys) {
      const card = document.createElement("div");
      card.className = "mini-card";
      card.dataset.key = key;
      const title = document.createElement("div");
      title.className = "mini-title";
      title.textContent = key;
      const canvas = document.createElement("canvas");
      canvas.width = 520;
      canvas.height = 260;
      card.append(title, canvas);
      els.observableGrid.append(card);
    }
    if (!keys.length) {
      const empty = document.createElement("div");
      empty.className = "observable-empty";
      empty.textContent = "No observable series match the current filters.";
      els.observableGrid.append(empty);
    }
  }

  for (const card of els.observableGrid.querySelectorAll(".mini-card")) {
    const key = card.dataset.key;
    const canvas = card.querySelector("canvas");
    const points = (visible.observables.get(key) || []).map((point) => ({
      x: point.step,
      y: point.value,
    }));
    const lossOverlays = [
      {
        name: "train",
        points: visible.train.map((point) => ({ x: point.step, y: point.value })),
        color: cssVar("--train"),
        overlay: true,
      },
      {
        name: "val",
        points: visible.val.map((point) => ({ x: point.step, y: point.value })),
        color: cssVar("--val"),
        overlay: true,
      },
    ];
    drawLineChart(
      canvas,
      [...lossOverlays, { name: key, points, color: colorForKey(key) }],
      { compact: true, fill: true }
    );
  }
}

function renderObservableFilters(visible) {
  updateSelectOptions(
    els.observableLayer,
    ["all", ...observableLayers(visible.observables.keys())],
    state.filters.layer,
    (value) => (value === "all" ? "All layers" : value.replace("_", " "))
  );
  updateSelectOptions(
    els.observableMetric,
    ["all", ...observableMetrics(visible.observables.keys())],
    state.filters.metric,
    (value) => (value === "all" ? "All metrics" : value)
  );
}

function updateSelectOptions(select, values, selected, labelFor) {
  const signature = values.join("\n");
  if (select.dataset.signature !== signature) {
    select.innerHTML = "";
    for (const value of values) {
      const option = document.createElement("option");
      option.value = value;
      option.textContent = labelFor(value);
      select.append(option);
    }
    select.dataset.signature = signature;
  }
  select.value = values.includes(selected) ? selected : "all";
  if (select.value !== selected) {
    if (select === els.observableLayer) state.filters.layer = select.value;
    if (select === els.observableMetric) state.filters.metric = select.value;
  }
}

function observableParts(key) {
  const parts = key.split(".");
  const scope = parts[0] === "train" || parts[0] === "val" ? parts[0] : "global";
  const layer = parts.find((part) => /^layer_\d+$/.test(part)) || "global";
  let metric = key;
  if (layer !== "global") {
    const layerIndex = parts.indexOf(layer);
    metric = parts.slice(layerIndex + 1).join(".");
  } else if (scope !== "global") {
    metric = parts.slice(1).join(".");
  }
  return { scope, layer, metric };
}

function observableLayers(keys) {
  return [...new Set([...keys].map((key) => observableParts(key).layer).filter((layer) => layer !== "global"))]
    .sort((a, b) => Number(a.split("_")[1]) - Number(b.split("_")[1]));
}

function observableMetrics(keys) {
  return [...new Set([...keys].map((key) => observableParts(key).metric).filter(Boolean))]
    .sort((a, b) => a.localeCompare(b));
}

function matchesObservableFilters(key) {
  const filters = state.filters;
  const parts = observableParts(key);
  if (filters.scope !== "all" && parts.scope !== filters.scope) return false;
  if (filters.layer !== "all" && parts.layer !== filters.layer) return false;
  if (filters.metric !== "all" && parts.metric !== filters.metric) return false;
  const query = filters.search.trim().toLowerCase();
  if (query && !key.toLowerCase().includes(query) && !parts.metric.toLowerCase().includes(query)) return false;
  return true;
}

function drawLineChart(canvas, series, opts = {}) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const width = Math.max(1, Math.floor(rect.width * dpr));
  const height = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  ctx.clearRect(0, 0, width, height);
  ctx.fillStyle = opts.fill ? "#171a20" : "#111419";
  ctx.fillRect(0, 0, width, height);

  const points = series.flatMap((s) => s.points).filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  if (points.length < 2) {
    drawEmpty(ctx, width, height);
    return;
  }

  const pad = opts.compact
    ? { left: 38 * dpr, right: 12 * dpr, top: 14 * dpr, bottom: 26 * dpr }
    : { left: 54 * dpr, right: 20 * dpr, top: 22 * dpr, bottom: 40 * dpr };
  const mainPoints = series
    .filter((item) => !item.overlay)
    .flatMap((item) => item.points)
    .filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
  const yPoints = mainPoints.length >= 2 ? mainPoints : points;
  const minX = Math.min(...points.map((p) => p.x));
  const maxX = Math.max(...points.map((p) => p.x));
  let minY = Math.min(...yPoints.map((p) => p.y));
  let maxY = Math.max(...yPoints.map((p) => p.y));
  if (minY === maxY) {
    minY -= 1;
    maxY += 1;
  }
  const yPad = (maxY - minY) * 0.12;
  minY -= yPad;
  maxY += yPad;

  const plotW = width - pad.left - pad.right;
  const plotH = height - pad.top - pad.bottom;
  const sx = (x) => pad.left + ((x - minX) / Math.max(1, maxX - minX)) * plotW;
  const sy = (y, localMinY = minY, localMaxY = maxY) =>
    pad.top + (1 - (y - localMinY) / Math.max(1e-12, localMaxY - localMinY)) * plotH;

  drawGrid(ctx, width, height, pad, minX, maxX, minY, maxY, opts.compact);

  for (const item of series) {
    const clean = item.points.filter((p) => Number.isFinite(p.x) && Number.isFinite(p.y));
    if (clean.length < 2) continue;
    let localMinY = minY;
    let localMaxY = maxY;
    if (item.overlay) {
      localMinY = Math.min(...clean.map((p) => p.y));
      localMaxY = Math.max(...clean.map((p) => p.y));
      if (localMinY === localMaxY) {
        localMinY -= 1;
        localMaxY += 1;
      }
      const localPad = (localMaxY - localMinY) * 0.12;
      localMinY -= localPad;
      localMaxY += localPad;
    }
    ctx.globalAlpha = item.overlay ? 0.34 : 1;
    ctx.lineWidth = (item.overlay ? 1.1 : opts.compact ? 1.6 : 2.4) * dpr;
    ctx.strokeStyle = item.color;
    ctx.beginPath();
    clean.forEach((point, index) => {
      const x = sx(point.x);
      const y = sy(point.y, localMinY, localMaxY);
      if (index === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    const last = clean.at(-1);
    ctx.fillStyle = item.color;
    ctx.beginPath();
    ctx.arc(
      sx(last.x),
      sy(last.y, localMinY, localMaxY),
      (item.overlay ? 1.8 : opts.compact ? 2.6 : 4) * dpr,
      0,
      Math.PI * 2
    );
    ctx.fill();
    ctx.globalAlpha = 1;
  }
}

function drawGrid(ctx, width, height, pad, minX, maxX, minY, maxY, compact) {
  ctx.strokeStyle = "#2a3038";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#7f8a97";
  ctx.font = `${compact ? 10 : 11}px ui-monospace, monospace`;
  ctx.textBaseline = "middle";
  const ticks = compact ? 3 : 5;
  for (let i = 0; i <= ticks; i += 1) {
    const y = pad.top + ((height - pad.top - pad.bottom) * i) / ticks;
    ctx.beginPath();
    ctx.moveTo(pad.left, y);
    ctx.lineTo(width - pad.right, y);
    ctx.stroke();
    const value = maxY - ((maxY - minY) * i) / ticks;
    ctx.fillText(formatTick(value), 6, y);
  }
  ctx.textBaseline = "alphabetic";
  ctx.fillText(String(Math.round(minX)), pad.left, height - 7);
  ctx.textAlign = "right";
  ctx.fillText(String(Math.round(maxX)), width - pad.right, height - 7);
  ctx.textAlign = "left";
}

function drawEmpty(ctx, width, height) {
  ctx.fillStyle = "#7f8a97";
  ctx.font = "13px ui-sans-serif, system-ui";
  ctx.textAlign = "center";
  ctx.fillText("Waiting for series data", width / 2, height / 2);
  ctx.textAlign = "left";
}

function formatTick(value) {
  const abs = Math.abs(value);
  if (abs >= 1000) return `${(value / 1000).toFixed(1)}k`;
  if (abs >= 10) return value.toFixed(0);
  if (abs >= 1) return value.toFixed(2);
  return value.toPrecision(2);
}

function cssVar(name) {
  return getComputedStyle(document.documentElement).getPropertyValue(name).trim();
}

function colorForKey(key) {
  if (key.includes("loss") || key.includes("gap")) return "#ff7a59";
  if (key.includes("mfu") || key.includes("tok")) return "#4fd1c5";
  if (key.includes("lr") || key.includes("momentum")) return "#f6c85f";
  if (key.includes("weight")) return "#9ae66e";
  return "#9fb7ff";
}

function tickPlayback() {
  if (state.playing && !state.live && state.parsed.train.length) {
    state.frame = Math.min(state.parsed.train.length - 1, state.frame + Math.ceil(state.speed));
    renderAll();
  }
  requestAnimationFrame(tickPlayback);
}

els.logFile.addEventListener("change", async (event) => {
  const file = event.target.files?.[0];
  if (!file) return;
  state.manual = true;
  state.live = false;
  state.playing = false;
  state.frame = 0;
  setSourceUrl(file.name);
  setLogText(await file.text());
});

els.loadUrl.addEventListener("click", () => {
  state.manual = false;
  setSourceUrl(els.logUrl.value.trim() || DEFAULT_LOG_URL);
  state.live = true;
  fetchLog();
});

els.scrollLog.addEventListener("click", () => {
  state.stickLog = !state.stickLog;
  els.scrollLog.textContent = state.stickLog ? "Stick to bottom" : "Free scroll";
  renderLog();
});

els.observableSearch.addEventListener("input", () => {
  state.filters.search = els.observableSearch.value;
  renderAll();
});

els.observableScope.addEventListener("change", () => {
  state.filters.scope = els.observableScope.value;
  renderAll();
});

els.observableLayer.addEventListener("change", () => {
  state.filters.layer = els.observableLayer.value;
  renderAll();
});

els.observableMetric.addEventListener("change", () => {
  state.filters.metric = els.observableMetric.value;
  renderAll();
});

els.commandLog.addEventListener("scroll", () => {
  const distance = els.commandLog.scrollHeight - els.commandLog.scrollTop - els.commandLog.clientHeight;
  state.stickLog = distance < 24;
  els.scrollLog.textContent = state.stickLog ? "Stick to bottom" : "Free scroll";
});

window.addEventListener("resize", renderAll);

connectLogSocket();
tickPlayback();
