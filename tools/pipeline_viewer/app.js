"use strict";

const SAMPLE_CSV = `start,end,resource,detail
-16,0,"C2C.E.Prefetch","page=0 bank=0 bindings=gate+up bytes=4096 lanes=8 bandwidth=256B/cycle planned=true"
-16,0,"C2C.W.Prefetch","page=0 bank=0 bindings=gate+up bytes=4096 lanes=8 bandwidth=256B/cycle planned=true"
0,24,"MEM.E.Read","slice=0 addr=0 stream=E0 count=24 interval=1 stride=1"
8,12,"VXM.ALU0","multiply"
12,16,"VXM.ALU8","cast -> E8"
20,52,"MXM.E0.Load","IW buffer=0 column=0"
52,84,"MXM.E0.Compute","Compute buffer=0 act=E0 out=W0 acc=0 stride=1 dst=sram"
52,84,"MXM.W0.Compute","Compute buffer=0 act=W0 out=E0 acc=0 stride=1 dst=sram"
80,112,"MEM.W.Read","slice=8 addr=320 stream=W8 count=32 interval=1 stride=1"
96,128,"MXM.E1.Load","IW buffer=1 column=1"
128,160,"MXM.E1.Compute","Compute buffer=1 act=E4 out=W4 acc=32 stride=1 dst=stream"
136,168,"SXM.E.Transpose","transpose matrix_columns"
172,204,"SXM.E.Permute","permute matrix_columns"
186,218,"VXM.ALU1","square"
187,219,"VXM.ALU2","add"
220,224,"VXM.ALU5","sqrt"
224,256,"VXM.ALU7","multiply -> E0"
230,262,"MEM.E.Write","slice=40 addr=0 stream=E0 count=32 interval=1 stride=1"
248,280,"ACC.E.MXM0","partial sum -> SRAM"
264,296,"ACC.E.MXM0","final sum -> stream + clear"
276,308,"MXM.W1.Compute","AccumulatorRead acc=64 out=E12 clear"
296,320,"MEM.W.Write","slice=41 addr=96 stream=W1 count=24 interval=1 stride=1"`;

const palette = {
  "C2C.Prefetch": "#d58a35",
  "MEM.Read": "#62a982",
  "MEM.Write": "#e0ae4f",
  "MEM.ReadWrite": "#b78c48",
  "MEM.Gather": "#559b83",
  "MEM.Scatter": "#d0914c",
  SXM: "#2e9e96",
  VXM: "#dc7855",
  "MXM.Load": "#78a7d7",
  "MXM.Compute": "#4f85c5",
  "MXM.Partial": "#9270b6",
  "MXM.Final": "#c45055",
  other: "#85949b",
};

const OVERVIEW_HANDLE_MIN_WIDTH = 24;
const OVERVIEW_FINE_PAN_FACTOR = 0.1;

const state = {
  allEvents: [],
  events: [],
  resources: [],
  rows: new Map(),
  groups: new Map(),
  overviewDirty: true,
  overviewCache: null,
  fullStart: 0,
  fullEnd: 320,
  viewStart: 0,
  viewEnd: 320,
  scrollY: 0,
  cursorA: 80,
  cursorB: 240,
  activeCursor: "A",
  selected: null,
  hover: null,
  drag: null,
  overviewDrag: null,
  family: "ALL",
  query: "",
  fileName: "sample_trace.csv",
  drawPending: false,
  drawFullPending: false,
  filterTimer: null,
  loadWorker: null,
};

const dom = {
  fileInput: document.querySelector("#fileInput"),
  fitButton: document.querySelector("#fitButton"),
  zoomInButton: document.querySelector("#zoomInButton"),
  zoomOutButton: document.querySelector("#zoomOutButton"),
  searchInput: document.querySelector("#searchInput"),
  familyFilter: document.querySelector("#familyFilter"),
  cursorAButton: document.querySelector("#cursorAButton"),
  cursorBButton: document.querySelector("#cursorBButton"),
  resourceLabels: document.querySelector("#resourceLabels"),
  canvasViewport: document.querySelector("#canvasViewport"),
  waveCanvas: document.querySelector("#waveCanvas"),
  rulerCanvas: document.querySelector("#rulerCanvas"),
  overviewCanvas: document.querySelector("#overviewCanvas"),
  emptyState: document.querySelector("#emptyState"),
  traceSummary: document.querySelector("#traceSummary"),
  viewportReadout: document.querySelector("#viewportReadout"),
  spanReadout: document.querySelector("#spanReadout"),
  cursorAReadout: document.querySelector("#cursorAReadout"),
  cursorBReadout: document.querySelector("#cursorBReadout"),
  cursorDeltaReadout: document.querySelector("#cursorDeltaReadout"),
  visibleReadout: document.querySelector("#visibleReadout"),
  fileName: document.querySelector("#fileName"),
  interactionState: document.querySelector("#interactionState"),
  detailResource: document.querySelector("#detailResource"),
  detailCycles: document.querySelector("#detailCycles"),
  detailDuration: document.querySelector("#detailDuration"),
  detailText: document.querySelector("#detailText"),
  eventColor: document.querySelector("#eventColor"),
};

function parseCsv(text) {
  const events = [];
  let headers = null;
  let positions = null;
  let row = [];
  let field = "";
  let quoted = false;

  function consumeRow() {
    if (!row.some(Boolean)) {
      row = [];
      return;
    }
    if (!headers) {
      headers = row.map((value) => value.trim().toLowerCase());
      const required = ["start", "end", "resource", "detail"];
      if (required.some((key) => !headers.includes(key))) {
        throw new Error("CSV 必须包含 start,end,resource,detail");
      }
      positions = Object.fromEntries(
        required.map((key) => [key, headers.indexOf(key)]));
      row = [];
      return;
    }
    const start = Number(row[positions.start]);
    const end = Number(row[positions.end]);
    if (!Number.isFinite(start) || !Number.isFinite(end) || end <= start) {
      throw new Error(`第 ${events.length + 2} 行 cycle 范围无效`);
    }
    events.push({
      id: events.length,
      start,
      end,
      resource: row[positions.resource] || "Unknown",
      detail: row[positions.detail] || "",
    });
    row = [];
  }

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    if (quoted) {
      if (char === '"' && text[index + 1] === '"') {
        field += '"';
        index += 1;
      } else if (char === '"') {
        quoted = false;
      } else {
        field += char;
      }
    } else if (char === '"') {
      quoted = true;
    } else if (char === ",") {
      row.push(field);
      field = "";
    } else if (char === "\n") {
      row.push(field.replace(/\r$/, ""));
      field = "";
      consumeRow();
    } else {
      field += char;
    }
  }
  if (field || row.length) {
    row.push(field.replace(/\r$/, ""));
    consumeRow();
  }
  return events.map((event) => ({
    ...event,
    pattern: "single",
    patternEnd: event.end,
    innerCount: 1,
    innerInterval: 0,
    innerStride: 0,
    outerCount: 1,
    outerInterval: 0,
    outerStride: 0,
    skipFirst: false,
    induction: "none",
    baseDelta: 0,
    logicalCount: 1,
  }));
}

function csvWorkerMain() {
  const optionalColumns = [
    "pattern", "inner_count", "inner_interval", "inner_stride",
    "outer_count", "outer_interval", "outer_stride", "skip_first",
    "induction", "base_delta",
  ];
  let headers = null;
  let positions = null;
  let row = [];
  let field = "";
  let quoted = false;
  let pendingQuote = false;
  let lineNumber = 0;
  let nextId = 0;
  let batch = [];

  function emitBatch() {
    if (!batch.length) return;
    self.postMessage({type: "batch", patterns: batch});
    batch = [];
  }

  function consumeRow() {
    lineNumber += 1;
    if (!row.some(Boolean)) {
      row = [];
      return;
    }
    if (!headers) {
      headers = row.map((item) => item.trim().toLowerCase());
      const required = ["start", "end", "resource", "detail"];
      if (required.some((key) => !headers.includes(key)))
        throw new Error("CSV must contain start,end,resource,detail");
      positions = Object.fromEntries([...required, ...optionalColumns]
        .map((key) => [key, headers.indexOf(key)]));
      row = [];
      return;
    }
    const value = (key, fallback = "") => positions[key] >= 0
      ? (row[positions[key]] ?? fallback) : fallback;
    const number = (key, fallback) => {
      const result = Number(value(key, String(fallback)));
      if (!Number.isFinite(result))
        throw new Error(`line ${lineNumber}: invalid ${key}`);
      return result;
    };
    const start = number("start", 0);
    const end = number("end", 0);
    if (end <= start)
      throw new Error(`line ${lineNumber}: invalid cycle range`);
    const pattern = value("pattern", "single").trim().toLowerCase() || "single";
    if (!["single", "repeat", "repeat2d"].includes(pattern))
      throw new Error(`line ${lineNumber}: invalid pattern ${pattern}`);
    const innerCount = number("inner_count", 1);
    const innerInterval = number("inner_interval", 0);
    const innerStride = number("inner_stride", 0);
    const outerCount = number("outer_count", 1);
    const outerInterval = number("outer_interval", 0);
    const outerStride = number("outer_stride", 0);
    const baseDelta = number("base_delta", 0);
    const skipFirst = number("skip_first", 0) !== 0;
    if (!Number.isInteger(innerCount) || innerCount < 1
        || !Number.isInteger(outerCount) || outerCount < 1
        || innerInterval < 0 || outerInterval < 0
        || (innerCount > 1 && innerInterval <= 0)
        || (outerCount > 1 && outerInterval <= 0))
      throw new Error(`line ${lineNumber}: invalid repeat parameters`);
    const logicalCount = innerCount * outerCount - (skipFirst ? 1 : 0);
    if (logicalCount > 0) {
      batch.push({
        id: nextId,
        start,
        end,
        patternEnd: start + (end - start)
          + (innerCount - 1) * innerInterval
          + (outerCount - 1) * outerInterval,
        resource: value("resource") || "Unknown",
        detail: value("detail"),
        pattern,
        innerCount,
        innerInterval,
        innerStride,
        outerCount,
        outerInterval,
        outerStride,
        skipFirst,
        induction: value("induction", "none") || "none",
        baseDelta,
        logicalCount,
      });
      nextId += 1;
    }
    row = [];
    if (batch.length >= 50_000) emitBatch();
  }

  function parseChunk(text) {
    let index = 0;
    if (pendingQuote) {
      pendingQuote = false;
      if (text[0] === '"') {
        field += '"';
        index = 1;
      } else {
        quoted = false;
      }
    }
    for (; index < text.length; index += 1) {
      const char = text[index];
      if (quoted) {
        if (char === '"' && text[index + 1] === '"') {
          field += '"';
          index += 1;
        } else if (char === '"' && index + 1 === text.length) {
          pendingQuote = true;
        } else if (char === '"') {
          quoted = false;
        } else {
          field += char;
        }
      } else if (char === '"') {
        quoted = true;
      } else if (char === ",") {
        row.push(field);
        field = "";
      } else if (char === "\n") {
        row.push(field.replace(/\r$/, ""));
        field = "";
        consumeRow();
      } else {
        field += char;
      }
    }
  }

  self.onmessage = async ({data}) => {
    try {
      const reader = data.file.stream().getReader();
      const decoder = new TextDecoder();
      let loaded = 0;
      let reported = 0;
      while (true) {
        const {done, value: bytes} = await reader.read();
        if (done) break;
        loaded += bytes.byteLength;
        parseChunk(decoder.decode(bytes, {stream: true}));
        if (loaded - reported >= 4 * 1024 * 1024) {
          self.postMessage({type: "progress", loaded, size: data.file.size});
          reported = loaded;
        }
      }
      parseChunk(decoder.decode());
      if (pendingQuote) {
        pendingQuote = false;
        quoted = false;
      }
      if (quoted) throw new Error("unterminated quoted CSV field");
      if (field || row.length) {
        row.push(field.replace(/\r$/, ""));
        consumeRow();
      }
      emitBatch();
      self.postMessage({type: "done", patternCount: nextId});
    } catch (error) {
      self.postMessage({type: "error", message: error.message});
    }
  };
}

function createCsvWorker() {
  const source = `(${csvWorkerMain.toString()})()`;
  const url = URL.createObjectURL(new Blob([source], {type: "text/javascript"}));
  const worker = new Worker(url);
  worker.sourceUrl = url;
  return worker;
}

function disposeCsvWorker(worker) {
  if (!worker) return;
  worker.terminate();
  URL.revokeObjectURL(worker.sourceUrl);
}

function resourceFamily(resource) {
  const family = resource.split(".")[0].toUpperCase();
  return family === "ACC" ? "MXM" : family;
}

function resourceOrder(resource) {
  const family = resourceFamily(resource);
  const familyRank = { C2C: 0, MEM: 1, MXM: 2, VXM: 3, SXM: 4 };
  const side = resource.includes(".E") ? 0 : resource.includes(".W") ? 1 : 2;
  const number = Number((resource.match(/(?:ALU|MXM)?(\d+)/) || [0, 0])[1]);
  const operation = resource.endsWith("Read") || resource.endsWith("Load")
    ? 0
    : resource.endsWith("Write") || resource.endsWith("Compute")
      ? 1
      : 2;
  return [familyRank[family] ?? 9, side, operation, number, resource];
}

function compareTuple(a, b) {
  for (let index = 0; index < Math.max(a.length, b.length); index += 1) {
    if (a[index] < b[index]) return -1;
    if (a[index] > b[index]) return 1;
  }
  return 0;
}

function eventColor(event) {
  const family = resourceFamily(event.resource);
  if (family === "C2C") return palette["C2C.Prefetch"];
  if (family === "MEM") {
    const operation = event.resource.split(".").at(-1);
    return palette[`MEM.${operation}`] || palette.other;
  }
  if (family === "MXM") {
    if (event.resource.endsWith("Load")) return palette["MXM.Load"];
    const detail = event.detail.toLowerCase();
    if (detail.includes("clear")
        || detail.includes("dst=stream")
        || detail.includes("final sum")) {
      return palette["MXM.Final"];
    }
    if (detail.includes("dst=sram")
        || detail.includes("partial sum")
        || event.resource.startsWith("ACC.")) {
      return palette["MXM.Partial"];
    }
    return palette["MXM.Compute"];
  }
  return palette[family] || palette.other;
}

function setTrace(events, fileName) {
  if (!events.length) throw new Error("CSV 中没有 schedule event");
  state.allEvents = events;
  state.fullStart = Infinity;
  state.fullEnd = -Infinity;
  for (const event of events) {
    state.fullStart = Math.min(state.fullStart, event.start);
    state.fullEnd = Math.max(state.fullEnd, event.patternEnd);
  }
  state.fileName = fileName;
  state.cursorA = state.fullStart + (state.fullEnd - state.fullStart) * 0.25;
  state.cursorB = state.fullStart + (state.fullEnd - state.fullStart) * 0.75;
  state.selected = null;
  state.scrollY = 0;
  applyFilters();
  fitAll();
  dom.fileName.textContent = fileName;
  const fullSpan = Math.ceil(state.fullEnd - state.fullStart);
  dom.traceSummary.textContent =
    `${events.length.toLocaleString()} events · ${fullSpan.toLocaleString()} cycles`;
  const logicalCount = events.reduce(
    (total, event) => total + event.logicalCount, 0);
  dom.traceSummary.textContent = `${events.length.toLocaleString()} patterns / `
    + `${logicalCount.toLocaleString()} events / `
    + `${fullSpan.toLocaleString()} cycles`;
  dom.interactionState.textContent = "Loaded";
}

function applyFilters() {
  const query = state.query.trim().toLowerCase();
  state.events = state.family === "ALL" && !query
    ? state.allEvents
    : state.allEvents.filter((event) => {
      const familyMatches = state.family === "ALL"
        || resourceFamily(event.resource) === state.family;
      const queryMatches = !query
        || event.resource.toLowerCase().includes(query)
        || event.detail.toLowerCase().includes(query)
        || String(event.start).includes(query)
        || String(event.end).includes(query);
      return familyMatches && queryMatches;
    });
  state.resources = [...new Set(state.events.map((event) => event.resource))]
    .sort((a, b) => compareTuple(resourceOrder(a), resourceOrder(b)));
  state.rows = new Map(state.resources.map((resource, index) => [resource, index]));
  state.groups = new Map(state.resources.map((resource) => [
    resource,
    { events: [], prefixMaxEnd: [] },
  ]));
  for (const event of state.events) state.groups.get(event.resource).events.push(event);
  for (const group of state.groups.values()) {
    group.events.sort((a, b) => a.start - b.start || a.patternEnd - b.patternEnd);
    let maxEnd = -Infinity;
    group.prefixMaxEnd = group.events.map((event) => {
      maxEnd = Math.max(maxEnd, event.patternEnd);
      return maxEnd;
    });
  }
  state.overviewDirty = true;
  state.scrollY = Math.min(state.scrollY, maxScrollY());
  buildLabels();
  drawAll();
  dom.interactionState.textContent = "Ready";
}

function fitAll() {
  const padding = Math.max(1, (state.fullEnd - state.fullStart) * 0.015);
  state.viewStart = state.fullStart - padding;
  state.viewEnd = state.fullEnd + padding;
  drawAll();
}

function zoomAt(factor, anchorCycle, deferred = false) {
  const span = state.viewEnd - state.viewStart;
  const fullSpan = Math.max(1, state.fullEnd - state.fullStart);
  const newSpan = Math.min(fullSpan * 2.5, Math.max(4, span * factor));
  const ratio = (anchorCycle - state.viewStart) / span;
  state.viewStart = anchorCycle - newSpan * ratio;
  state.viewEnd = state.viewStart + newSpan;
  clampViewport();
  if (deferred) requestDrawAll();
  else drawAll();
}

function panCycles(delta, deferred = false) {
  state.viewStart += delta;
  state.viewEnd += delta;
  clampViewport();
  if (deferred) requestDrawAll();
  else drawAll();
}

function clampViewport() {
  const span = state.viewEnd - state.viewStart;
  const min = state.fullStart - span * 0.1;
  const max = state.fullEnd + span * 0.1;
  if (state.viewStart < min) {
    state.viewStart = min;
    state.viewEnd = min + span;
  }
  if (state.viewEnd > max) {
    state.viewEnd = max;
    state.viewStart = max - span;
  }
}

function maxScrollY() {
  return Math.max(0, state.resources.length * 30 - dom.canvasViewport.clientHeight);
}

function buildLabels() {
  dom.resourceLabels.replaceChildren();
  state.resources.forEach((resource, index) => {
    const label = document.createElement("div");
    label.className = "resource-label";
    label.style.top = `${index * 30 - state.scrollY}px`;
    label.textContent = resource;
    label.title = resource;
    dom.resourceLabels.append(label);
  });
}

function resizeCanvas(canvas) {
  const rect = canvas.getBoundingClientRect();
  const dpr = window.devicePixelRatio || 1;
  const width = Math.max(1, Math.round(rect.width * dpr));
  const height = Math.max(1, Math.round(rect.height * dpr));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  return { context, width: rect.width, height: rect.height, dpr };
}

function cycleToX(cycle, width) {
  return ((cycle - state.viewStart) / (state.viewEnd - state.viewStart)) * width;
}

function xToCycle(x, width) {
  return state.viewStart + (x / width) * (state.viewEnd - state.viewStart);
}

function niceStep(span, targetTicks) {
  const rough = span / targetTicks;
  const exponent = 10 ** Math.floor(Math.log10(Math.max(rough, 1e-9)));
  const fraction = rough / exponent;
  const nice = fraction <= 1 ? 1 : fraction <= 2 ? 2 : fraction <= 5 ? 5 : 10;
  return nice * exponent;
}

function drawRuler() {
  const { context, width, height } = resizeCanvas(dom.rulerCanvas);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#f3f6f7";
  context.fillRect(0, 0, width, height);
  const span = state.viewEnd - state.viewStart;
  const step = niceStep(span, Math.max(3, Math.floor(width / 110)));
  const first = Math.ceil(state.viewStart / step) * step;
  context.strokeStyle = "#aeb9be";
  context.fillStyle = "#526068";
  context.font = '10px "Segoe UI", sans-serif';
  context.textAlign = "center";
  for (let cycle = first; cycle <= state.viewEnd; cycle += step) {
    const x = cycleToX(cycle, width);
    context.beginPath();
    context.moveTo(x + 0.5, height - 7);
    context.lineTo(x + 0.5, height);
    context.stroke();
    context.fillText(formatCycle(cycle), x, 12);
  }
}

function formatCycle(value) {
  const rounded = Math.round(value);
  if (Math.abs(rounded) >= 1_000_000) return `${(rounded / 1_000_000).toFixed(2)}M`;
  if (Math.abs(rounded) >= 10_000) return `${(rounded / 1_000).toFixed(1)}k`;
  return rounded.toLocaleString();
}

function patternVisibleSegments(pattern, viewStart, viewEnd) {
  const duration = pattern.end - pattern.start;
  const segments = [];
  let total = 0;
  for (let outer = 0; outer < pattern.outerCount; outer += 1) {
    const outerStart = pattern.start + outer * pattern.outerInterval;
    let first = pattern.innerCount === 1 ? 0
      : Math.ceil((viewStart - duration - outerStart) / pattern.innerInterval);
    let last = pattern.innerCount === 1 ? 0
      : Math.floor((viewEnd - outerStart) / pattern.innerInterval);
    first = Math.max(0, first);
    last = Math.min(pattern.innerCount - 1, last);
    if (pattern.skipFirst && outer === 0 && first === 0) first = 1;
    if (last < first) continue;
    const count = last - first + 1;
    segments.push({outer, first, count, offset: total});
    total += count;
  }
  return {pattern, segments, total};
}

function adjustInducedDetail(detail, induction, delta) {
  if (!delta) return detail;
  const update = (expression) => detail.replace(expression,
    (match, prefix, value) => `${prefix}${Number(value) + delta}`);
  if (induction === "mem_address") return update(/(\baddr=)(-?\d+)/);
  if (induction === "mxm_weight_column") return update(/(\bcolumn=)(-?\d+)/);
  if (induction === "mxm_accumulator_address") return update(/(\bacc=)(-?\d+)/);
  return detail;
}

function materializePatternInstance(info, rank) {
  let segment = info.segments[0];
  for (const candidate of info.segments) {
    if (rank < candidate.offset + candidate.count) {
      segment = candidate;
      break;
    }
  }
  const inner = segment.first + rank - segment.offset;
  const {pattern} = info;
  const start = pattern.start
    + segment.outer * pattern.outerInterval
    + inner * pattern.innerInterval;
  const delta = pattern.baseDelta
    + segment.outer * pattern.outerStride
    + inner * pattern.innerStride;
  return {
    id: `${pattern.id}:${segment.outer}:${inner}`,
    start,
    end: start + pattern.end - pattern.start,
    resource: pattern.resource,
    detail: adjustInducedDetail(pattern.detail, pattern.induction, delta),
  };
}

function visibleEvents() {
  const topRow = Math.max(0, Math.floor(state.scrollY / 30));
  const bottomRow = Math.min(
    state.resources.length - 1,
    topRow + Math.ceil(dom.canvasViewport.clientHeight / 30) + 1);
  const visible = [];
  let total = 0;
  let sampled = false;
  const visibleRows = Math.max(1, bottomRow - topRow + 1);
  const perRowBudget = Math.max(300, Math.floor(18_000 / visibleRows));
  for (let row = topRow; row <= bottomRow; row += 1) {
    const group = state.groups.get(state.resources[row]);
    if (!group) continue;
    const first = lowerBound(group.prefixMaxEnd, state.viewStart);
    const last = upperBoundEvents(group.events, state.viewEnd);
    const infos = [];
    let rowTotal = 0;
    for (let index = first; index < last; index += 1) {
      const pattern = group.events[index];
      if (pattern.patternEnd < state.viewStart) continue;
      const info = patternVisibleSegments(
        pattern, state.viewStart, state.viewEnd);
      if (!info.total) continue;
      info.offset = rowTotal;
      rowTotal += info.total;
      infos.push(info);
    }
    total += rowTotal;
    const stride = Math.max(1, Math.ceil(rowTotal / perRowBudget));
    if (stride > 1) sampled = true;
    let nextRank = 0;
    for (const info of infos) {
      while (nextRank < info.offset + info.total) {
        if (nextRank >= info.offset)
          visible.push(materializePatternInstance(info, nextRank - info.offset));
        nextRank += stride;
      }
    }
  }
  return {events: visible, total, sampled};
}

function lowerBound(values, target) {
  let low = 0;
  let high = values.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (values[middle] < target) low = middle + 1;
    else high = middle;
  }
  return low;
}

function upperBoundEvents(events, target) {
  let low = 0;
  let high = events.length;
  while (low < high) {
    const middle = (low + high) >>> 1;
    if (events[middle].start <= target) low = middle + 1;
    else high = middle;
  }
  return low;
}

function drawWave() {
  const { context, width, height } = resizeCanvas(dom.waveCanvas);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#ffffff";
  context.fillRect(0, 0, width, height);
  const firstRow = Math.floor(state.scrollY / 30);
  const offset = -(state.scrollY % 30);
  const rowCount = Math.ceil(height / 30) + 1;
  for (let local = 0; local < rowCount; local += 1) {
    const y = offset + local * 30;
    context.fillStyle = (firstRow + local) % 2 ? "#f8fafb" : "#ffffff";
    context.fillRect(0, y, width, 30);
    context.strokeStyle = "#e5e9eb";
    context.beginPath();
    context.moveTo(0, Math.round(y + 29.5));
    context.lineTo(width, Math.round(y + 29.5));
    context.stroke();
  }
  drawGrid(context, width, height);
  const visible = visibleEvents();
  for (const event of visible.events) drawEvent(context, width, event);
  drawCursor(context, width, height, state.cursorA, "#167a72", "A");
  drawCursor(context, width, height, state.cursorB, "#c45055", "B");
  dom.visibleReadout.textContent = visible.sampled
    ? `${visible.total.toLocaleString()} visible · LOD ${visible.events.length.toLocaleString()}`
    : `${visible.total.toLocaleString()} visible events`;
  dom.emptyState.hidden = state.events.length !== 0;
}

function drawGrid(context, width, height) {
  const span = state.viewEnd - state.viewStart;
  const step = niceStep(span, Math.max(3, Math.floor(width / 110)));
  const first = Math.ceil(state.viewStart / step) * step;
  context.strokeStyle = "#e1e6e8";
  context.lineWidth = 1;
  for (let cycle = first; cycle <= state.viewEnd; cycle += step) {
    const x = Math.round(cycleToX(cycle, width)) + 0.5;
    context.beginPath();
    context.moveTo(x, 0);
    context.lineTo(x, height);
    context.stroke();
  }
}

function drawEvent(context, width, event) {
  const row = state.rows.get(event.resource);
  const y = row * 30 - state.scrollY + 5;
  const x1 = cycleToX(event.start, width);
  const x2 = cycleToX(event.end, width);
  const eventWidth = Math.max(2, x2 - x1);
  const selected = state.selected?.id === event.id;
  const hovered = state.hover?.id === event.id;
  context.fillStyle = eventColor(event);
  context.globalAlpha = selected ? 1 : hovered ? 0.94 : 0.84;
  context.fillRect(x1, y, eventWidth, 20);
  context.globalAlpha = 1;
  context.strokeStyle = selected ? "#152228" : "rgba(18, 31, 37, 0.28)";
  context.lineWidth = selected ? 2 : 1;
  context.strokeRect(x1 + 0.5, y + 0.5, Math.max(1, eventWidth - 1), 19);
  if (eventWidth > 38) {
    const label = event.detail || event.resource.split(".").at(-1);
    context.save();
    context.beginPath();
    context.rect(x1 + 3, y, eventWidth - 6, 20);
    context.clip();
    context.fillStyle = "#132127";
    context.font = '10px "Segoe UI", sans-serif';
    context.textAlign = "left";
    context.fillText(label, x1 + 5, y + 14);
    context.restore();
  }
}

function drawCursor(context, width, height, cycle, color, label) {
  const x = cycleToX(cycle, width);
  if (x < -10 || x > width + 10) return;
  context.strokeStyle = color;
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(Math.round(x) + 0.5, 0);
  context.lineTo(Math.round(x) + 0.5, height);
  context.stroke();
  context.fillStyle = color;
  context.fillRect(x - 8, 0, 16, 14);
  context.fillStyle = "#fff";
  context.font = 'bold 9px "Segoe UI", sans-serif';
  context.textAlign = "center";
  context.fillText(label, x, 10);
}

function drawOverview() {
  const { context, width, height, dpr } = resizeCanvas(dom.overviewCanvas);
  context.clearRect(0, 0, width, height);
  if (!state.overviewCache
      || state.overviewCache.width !== width
      || state.overviewCache.height !== height
      || state.overviewCache.dpr !== dpr) {
    state.overviewCache = {
      canvas: document.createElement("canvas"),
      width,
      height,
      dpr,
    };
    state.overviewDirty = true;
  }
  if (state.overviewDirty) rebuildOverviewCache();
  context.drawImage(state.overviewCache.canvas, 0, 0, width, height);
  const { x: viewX, width: viewWidth } = overviewViewportGeometry(width);
  context.fillStyle = "rgba(22, 122, 114, 0.08)";
  context.fillRect(viewX, 1, viewWidth, height - 2);
  context.strokeStyle = "#167a72";
  context.lineWidth = 2;
  context.strokeRect(viewX + 1, 2, Math.max(0, viewWidth - 2), height - 4);
  if (viewWidth >= OVERVIEW_HANDLE_MIN_WIDTH) {
    const gripX = viewX + viewWidth / 2;
    context.strokeStyle = "rgba(22, 122, 114, 0.65)";
    context.lineWidth = 1;
    for (const offset of [-3, 0, 3]) {
      context.beginPath();
      context.moveTo(Math.round(gripX + offset) + 0.5, height / 2 - 5);
      context.lineTo(Math.round(gripX + offset) + 0.5, height / 2 + 5);
      context.stroke();
    }
  }
}

function overviewViewportGeometry(width) {
  const fullSpan = Math.max(1, state.fullEnd - state.fullStart);
  const center = ((state.viewStart + state.viewEnd) / 2 - state.fullStart)
    / fullSpan * width;
  const actualWidth = (state.viewEnd - state.viewStart) / fullSpan * width;
  const handleWidth = Math.min(width, Math.max(OVERVIEW_HANDLE_MIN_WIDTH, actualWidth));
  return {
    x: Math.max(0, Math.min(width - handleWidth, center - handleWidth / 2)),
    width: handleWidth,
  };
}

function rebuildOverviewCache() {
  const { canvas, width, height, dpr } = state.overviewCache;
  canvas.width = Math.max(1, Math.round(width * dpr));
  canvas.height = Math.max(1, Math.round(height * dpr));
  const context = canvas.getContext("2d");
  context.setTransform(dpr, 0, 0, dpr, 0, 0);
  context.fillStyle = "#eef2f3";
  context.fillRect(0, 0, width, height);
  const fullSpan = Math.max(1, state.fullEnd - state.fullStart);
  const resourceCount = Math.max(1, state.resources.length);
  const stride = Math.max(1, Math.ceil(state.events.length / 120_000));
  for (const [resource, group] of state.groups) {
    const row = state.rows.get(resource);
    for (let index = 0; index < group.events.length; index += stride) {
      const event = group.events[index];
      const x = ((event.start - state.fullStart) / fullSpan) * width;
      const eventWidth = Math.max(1,
        ((event.patternEnd - event.start) / fullSpan) * width);
      const y = 4 + (row / resourceCount) * (height - 8);
      context.fillStyle = eventColor(event);
      context.globalAlpha = 0.72;
      context.fillRect(x, y, eventWidth, Math.max(1, (height - 8) / resourceCount));
    }
  }
  context.globalAlpha = 1;
  state.overviewDirty = false;
}

function updateReadouts() {
  const start = Math.round(state.viewStart);
  const end = Math.round(state.viewEnd);
  dom.viewportReadout.textContent = `${start.toLocaleString()} – ${end.toLocaleString()}`;
  dom.spanReadout.textContent = `${Math.round(end - start).toLocaleString()} cycles`;
  dom.cursorAReadout.textContent = Math.round(state.cursorA).toLocaleString();
  dom.cursorBReadout.textContent = Math.round(state.cursorB).toLocaleString();
  dom.cursorDeltaReadout.textContent =
    `${Math.round(Math.abs(state.cursorB - state.cursorA)).toLocaleString()} cycles`;
}

function updateInspector(event) {
  if (!event) {
    dom.detailResource.textContent = "选择一个事件";
    dom.detailCycles.textContent = "—";
    dom.detailDuration.textContent = "—";
    dom.detailText.textContent = "—";
    dom.eventColor.style.background = "#d5dde0";
    return;
  }
  dom.detailResource.textContent = event.resource;
  dom.detailCycles.textContent = `${event.start.toLocaleString()} – ${event.end.toLocaleString()}`;
  dom.detailDuration.textContent = `${(event.end - event.start).toLocaleString()} cycles`;
  dom.detailText.textContent = event.detail || "—";
  dom.eventColor.style.background = eventColor(event);
}

function drawAll() {
  if (!state.allEvents.length) return;
  drawRuler();
  drawWave();
  drawOverview();
  updateReadouts();
}

function requestDrawAll() {
  state.drawFullPending = true;
  if (state.drawPending) return;
  state.drawPending = true;
  requestAnimationFrame(() => {
    state.drawPending = false;
    const full = state.drawFullPending;
    state.drawFullPending = false;
    if (full) drawAll();
    else drawWave();
  });
}

function requestDrawWave() {
  if (state.drawPending) return;
  state.drawPending = true;
  requestAnimationFrame(() => {
    state.drawPending = false;
    const full = state.drawFullPending;
    state.drawFullPending = false;
    if (full) drawAll();
    else drawWave();
  });
}

function hitTest(clientX, clientY) {
  const rect = dom.waveCanvas.getBoundingClientRect();
  const x = clientX - rect.left;
  const y = clientY - rect.top;
  const cycle = xToCycle(x, rect.width);
  const row = Math.floor((y + state.scrollY) / 30);
  const resource = state.resources[row];
  if (!resource) return null;
  const group = state.groups.get(resource);
  const candidates = [];
  const first = lowerBound(group.prefixMaxEnd, cycle);
  for (let index = first; index < group.events.length; index += 1) {
    const pattern = group.events[index];
    if (pattern.start > cycle) break;
    if (pattern.patternEnd < cycle) continue;
    const info = patternVisibleSegments(pattern, cycle, cycle);
    for (let rank = 0; rank < info.total; rank += 1)
      candidates.push(materializePatternInstance(info, rank));
  }
  return candidates.sort((a, b) => (a.end - a.start) - (b.end - b.start))[0] || null;
}

function setActiveCursor(name) {
  state.activeCursor = name;
  dom.cursorAButton.classList.toggle("active", name === "A");
  dom.cursorBButton.classList.toggle("active", name === "B");
}

async function loadFile(file) {
  if (!file) return;
  disposeCsvWorker(state.loadWorker);
  state.loadWorker = null;
  const patterns = [];
  try {
    dom.interactionState.textContent = "Streaming 0%";
    const worker = createCsvWorker();
    state.loadWorker = worker;
    await new Promise((resolve, reject) => {
      worker.onmessage = ({data}) => {
        if (data.type === "batch") {
          patterns.push(...data.patterns);
        } else if (data.type === "progress") {
          const percent = Math.min(100,
            Math.floor(data.loaded / Math.max(1, data.size) * 100));
          dom.interactionState.textContent = `Streaming ${percent}%`;
        } else if (data.type === "done") {
          resolve();
        } else if (data.type === "error") {
          reject(new Error(data.message));
        }
      };
      worker.onerror = (event) => reject(
        new Error(event.message || "CSV worker failed"));
      worker.postMessage({file});
    });
    disposeCsvWorker(worker);
    state.loadWorker = null;
    dom.interactionState.textContent = "Indexing";
    setTrace(patterns, file.name);
  } catch (error) {
    disposeCsvWorker(state.loadWorker);
    state.loadWorker = null;
    dom.interactionState.textContent = error.message;
  } finally {
    dom.fileInput.value = "";
  }
}

dom.fileInput.addEventListener("change", () => loadFile(dom.fileInput.files[0]));
dom.fitButton.addEventListener("click", fitAll);
dom.zoomInButton.addEventListener("click", () =>
  zoomAt(0.65, (state.viewStart + state.viewEnd) / 2));
dom.zoomOutButton.addEventListener("click", () =>
  zoomAt(1.5, (state.viewStart + state.viewEnd) / 2));
dom.cursorAButton.addEventListener("click", () => setActiveCursor("A"));
dom.cursorBButton.addEventListener("click", () => setActiveCursor("B"));
dom.familyFilter.addEventListener("change", () => {
  state.family = dom.familyFilter.value;
  state.scrollY = 0;
  applyFilters();
});
dom.searchInput.addEventListener("input", () => {
  state.query = dom.searchInput.value;
  clearTimeout(state.filterTimer);
  dom.interactionState.textContent = "Filtering";
  state.filterTimer = setTimeout(applyFilters, 180);
});

dom.waveCanvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const rect = dom.waveCanvas.getBoundingClientRect();
  if (event.shiftKey) {
    panCycles(event.deltaY * (state.viewEnd - state.viewStart) / rect.width, true);
  } else if (event.altKey || event.ctrlKey || event.metaKey) {
    state.scrollY = Math.max(0, Math.min(maxScrollY(), state.scrollY + event.deltaY));
    buildLabels();
    requestDrawWave();
  } else {
    const anchor = xToCycle(event.clientX - rect.left, rect.width);
    zoomAt(Math.exp(event.deltaY * 0.0015), anchor, true);
  }
}, { passive: false });

dom.resourceLabels.addEventListener("wheel", (event) => {
  event.preventDefault();
  state.scrollY = Math.max(0, Math.min(maxScrollY(), state.scrollY + event.deltaY));
  buildLabels();
  requestDrawWave();
}, { passive: false });

dom.waveCanvas.addEventListener("pointerdown", (event) => {
  dom.waveCanvas.setPointerCapture(event.pointerId);
  state.drag = {
    x: event.clientX,
    viewStart: state.viewStart,
    viewEnd: state.viewEnd,
    moved: false,
  };
  dom.waveCanvas.classList.add("panning");
});

dom.waveCanvas.addEventListener("pointermove", (event) => {
  const hit = hitTest(event.clientX, event.clientY);
  state.hover = hit;
  if (state.drag) {
    const rect = dom.waveCanvas.getBoundingClientRect();
    const pixels = event.clientX - state.drag.x;
    if (Math.abs(pixels) > 2) state.drag.moved = true;
    const delta = -pixels * (state.drag.viewEnd - state.drag.viewStart) / rect.width;
    state.viewStart = state.drag.viewStart + delta;
    state.viewEnd = state.drag.viewEnd + delta;
    clampViewport();
    dom.interactionState.textContent = "Panning";
  } else {
    dom.interactionState.textContent = hit
      ? `${hit.resource} · ${hit.start}–${hit.end}`
      : "Ready";
  }
  requestDrawAll();
});

dom.waveCanvas.addEventListener("pointerup", (event) => {
  if (state.drag && !state.drag.moved) {
    const rect = dom.waveCanvas.getBoundingClientRect();
    const cycle = xToCycle(event.clientX - rect.left, rect.width);
    if (state.activeCursor === "A") state.cursorA = cycle;
    else state.cursorB = cycle;
    state.selected = hitTest(event.clientX, event.clientY);
    updateInspector(state.selected);
  }
  state.drag = null;
  dom.waveCanvas.classList.remove("panning");
  dom.interactionState.textContent = "Ready";
  drawAll();
});

dom.waveCanvas.addEventListener("pointerleave", () => {
  if (!state.drag) {
    state.hover = null;
    requestDrawWave();
  }
});

function beginOverviewDrag(event) {
  const rect = dom.overviewCanvas.getBoundingClientRect();
  const x = event.clientX - rect.left;
  const handle = overviewViewportGeometry(rect.width);
  if (x < handle.x || x > handle.x + handle.width) {
    const center = state.fullStart
      + Math.max(0, Math.min(1, x / rect.width)) * (state.fullEnd - state.fullStart);
    const span = state.viewEnd - state.viewStart;
    state.viewStart = center - span / 2;
    state.viewEnd = center + span / 2;
    clampViewport();
  }
  dom.overviewCanvas.setPointerCapture(event.pointerId);
  state.overviewDrag = { x: event.clientX };
  dom.overviewCanvas.classList.add("panning");
  dom.interactionState.textContent = "Overview panning";
  drawAll();
}

dom.overviewCanvas.addEventListener("pointerdown", beginOverviewDrag);

dom.overviewCanvas.addEventListener("pointermove", (event) => {
  if (!state.overviewDrag) return;
  const rect = dom.overviewCanvas.getBoundingClientRect();
  const pixels = event.clientX - state.overviewDrag.x;
  const fineFactor = event.shiftKey ? OVERVIEW_FINE_PAN_FACTOR : 1;
  const cycles = pixels / rect.width
    * (state.fullEnd - state.fullStart) * fineFactor;
  state.overviewDrag.x = event.clientX;
  panCycles(cycles, true);
});

function endOverviewDrag(event) {
  if (!state.overviewDrag) return;
  state.overviewDrag = null;
  if (dom.overviewCanvas.hasPointerCapture(event.pointerId)) {
    dom.overviewCanvas.releasePointerCapture(event.pointerId);
  }
  dom.overviewCanvas.classList.remove("panning");
  dom.interactionState.textContent = "Ready";
  drawAll();
}

dom.overviewCanvas.addEventListener("pointerup", endOverviewDrag);
dom.overviewCanvas.addEventListener("pointercancel", endOverviewDrag);

dom.overviewCanvas.addEventListener("wheel", (event) => {
  event.preventDefault();
  const rect = dom.overviewCanvas.getBoundingClientRect();
  const wheelDelta = Math.abs(event.deltaX) > Math.abs(event.deltaY)
    ? event.deltaX
    : event.deltaY;
  const fineFactor = event.shiftKey ? OVERVIEW_FINE_PAN_FACTOR : 1;
  panCycles(wheelDelta * (state.viewEnd - state.viewStart)
    / rect.width * fineFactor, true);
}, { passive: false });

dom.overviewCanvas.addEventListener("keydown", (event) => {
  if (event.key !== "ArrowLeft" && event.key !== "ArrowRight") return;
  event.preventDefault();
  const direction = event.key === "ArrowLeft" ? -1 : 1;
  const step = (state.viewEnd - state.viewStart)
    * (event.shiftKey ? 0.02 : 0.1);
  panCycles(direction * step);
});

document.body.addEventListener("dragover", (event) => {
  event.preventDefault();
  dom.canvasViewport.classList.add("dragging");
});
document.body.addEventListener("dragleave", (event) => {
  if (!event.relatedTarget) dom.canvasViewport.classList.remove("dragging");
});
document.body.addEventListener("drop", (event) => {
  event.preventDefault();
  dom.canvasViewport.classList.remove("dragging");
  loadFile(event.dataTransfer.files[0]);
});

window.addEventListener("resize", () => {
  state.overviewDirty = true;
  state.scrollY = Math.min(state.scrollY, maxScrollY());
  buildLabels();
  drawAll();
});

setTrace(parseCsv(SAMPLE_CSV), "sample_trace.csv");
