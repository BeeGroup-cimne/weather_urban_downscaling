const state = {
  layer: "prediction",
  latest: null,
  summary: null,
  hourly: [],
  events: [],
  mapPayload: null,
};

const levelClass = {
  normal: "level-normal",
  watch: "level-watch",
  warning: "level-warning",
  severe: "level-severe",
};

function fmt(value, digits = 1) {
  const n = Number(value);
  if (!Number.isFinite(n)) return "--";
  return n.toFixed(digits);
}

function pct(value) {
  return `${fmt(Number(value) * 100, 1)}%`;
}

async function fetchJson(url) {
  const response = await fetch(url, { cache: "no-store" });
  if (!response.ok) throw new Error(`${url} returned ${response.status}`);
  return response.json();
}

function setText(id, value) {
  document.getElementById(id).textContent = value;
}

function updateStatus() {
  const latest = state.latest;
  const level = String(latest.alert_level || "normal");
  const pill = document.getElementById("level-pill");
  pill.textContent = level;
  pill.className = `level-pill ${levelClass[level] || "level-normal"}`;

  setText("latest-time", `Latest forecast: ${latest.time || "--"} UTC`);
  setText("max-pred", `${fmt(latest.max_pred_c, 1)} C`);
  setText("mean-pred", `${fmt(latest.mean_pred_c, 1)} C`);
  setText("affected-area", pct(latest.exceed_fraction || 0));
  setText("active-event", latest.is_active_heatwave_event ? "Yes" : "No");
}

function colorRamp(t) {
  const stops = [
    [0.00, [63, 106, 154]],
    [0.32, [110, 182, 166]],
    [0.52, [240, 216, 106]],
    [0.72, [213, 144, 50]],
    [1.00, [185, 54, 50]],
  ];
  const x = Math.max(0, Math.min(1, t));
  for (let i = 0; i < stops.length - 1; i += 1) {
    const [aPos, a] = stops[i];
    const [bPos, b] = stops[i + 1];
    if (x >= aPos && x <= bPos) {
      const local = (x - aPos) / (bPos - aPos || 1);
      return [
        Math.round(a[0] + (b[0] - a[0]) * local),
        Math.round(a[1] + (b[1] - a[1]) * local),
        Math.round(a[2] + (b[2] - a[2]) * local),
      ];
    }
  }
  return stops[stops.length - 1][1];
}

function displayDomain(payload) {
  if (state.layer === "prediction") {
    return { min: 5, max: 45 };
  }
  const maxAbs = Math.max(Math.abs(Number(payload.min)), Math.abs(Number(payload.max)), 5);
  return { min: -maxAbs, max: maxAbs };
}

function exceedanceColor(value, min, max) {
  const v = Number(value);
  if (v <= 0) {
    const t = Math.max(0, Math.min(1, (v - min) / (0 - min || 1)));
    const blue = [68, 112, 158];
    const pale = [228, 235, 228];
    return [
      Math.round(blue[0] + (pale[0] - blue[0]) * t),
      Math.round(blue[1] + (pale[1] - blue[1]) * t),
      Math.round(blue[2] + (pale[2] - blue[2]) * t),
    ];
  }
  const t = Math.max(0, Math.min(1, v / (max || 1)));
  const warm = [242, 216, 106];
  const red = [185, 54, 50];
  return [
    Math.round(warm[0] + (red[0] - warm[0]) * t),
    Math.round(warm[1] + (red[1] - warm[1]) * t),
    Math.round(warm[2] + (red[2] - warm[2]) * t),
  ];
}

async function drawMap() {
  const endpoint = state.layer === "prediction" ? "/api/map/prediction" : "/api/map/exceedance";
  const payload = await fetchJson(`${endpoint}?max_size=180`);
  state.mapPayload = payload;
  const canvas = document.getElementById("map-canvas");
  canvas.width = payload.width;
  canvas.height = payload.height;
  const ctx = canvas.getContext("2d");
  const image = ctx.createImageData(payload.width, payload.height);
  const domain = displayDomain(payload);
  const min = domain.min;
  const max = domain.max;
  const spread = max - min || 1;

  payload.values.forEach((value, i) => {
    const normalized = (Number(value) - min) / spread;
    const [r, g, b] = state.layer === "exceedance" ? exceedanceColor(value, min, max) : colorRamp(normalized);
    const o = i * 4;
    image.data[o] = r;
    image.data[o + 1] = g;
    image.data[o + 2] = b;
    image.data[o + 3] = 255;
  });
  ctx.putImageData(image, 0, 0);

  const label = state.layer === "prediction" ? "Predicted air temperature in degrees Celsius." : "Difference from alert threshold in degrees Celsius.";
  setText("map-range", label);
  setText("legend-min", `${fmt(min, 1)} C`);
  setText("legend-max", `${fmt(max, 1)} C`);
  updateGeoCaption(payload.geo);
  drawOverlay(payload);
}

function updateGeoCaption(geo) {
  if (!geo || geo.lat_min === undefined) {
    setText("geo-caption", "Geographic bounds unavailable.");
    return;
  }
  setText(
    "geo-caption",
    `Bounds: ${fmt(geo.lat_min, 3)} to ${fmt(geo.lat_max, 3)} N, ${fmt(geo.lon_min, 3)} to ${fmt(geo.lon_max, 3)} E`
  );
}

function drawOverlay(payload) {
  const canvas = document.getElementById("overlay-canvas");
  canvas.width = payload.width;
  canvas.height = payload.height;
  const ctx = canvas.getContext("2d");
  ctx.clearRect(0, 0, canvas.width, canvas.height);

  ctx.strokeStyle = "rgba(255,255,255,0.55)";
  ctx.lineWidth = Math.max(1, canvas.width / 180);
  for (let i = 1; i < 4; i += 1) {
    const x = (canvas.width * i) / 4;
    const y = (canvas.height * i) / 4;
    ctx.beginPath();
    ctx.moveTo(x, 0);
    ctx.lineTo(x, canvas.height);
    ctx.moveTo(0, y);
    ctx.lineTo(canvas.width, y);
    ctx.stroke();
  }

  if (state.layer === "exceedance") {
    ctx.fillStyle = "rgba(185, 54, 50, 0.58)";
    payload.values.forEach((value, index) => {
      if (Number(value) > 0) {
        const x = index % payload.width;
        const y = Math.floor(index / payload.width);
        ctx.fillRect(x, y, 1, 1);
      }
    });
  }
}

function mapCellFromEvent(event) {
  const payload = state.mapPayload;
  if (!payload) return null;
  const canvas = document.getElementById("map-canvas");
  const rect = canvas.getBoundingClientRect();
  const x = Math.max(0, Math.min(payload.width - 1, Math.floor(((event.clientX - rect.left) / rect.width) * payload.width)));
  const y = Math.max(0, Math.min(payload.height - 1, Math.floor(((event.clientY - rect.top) / rect.height) * payload.height)));
  const index = y * payload.width + x;
  return { x, y, index, rect };
}

function showMapTooltip(event) {
  const cell = mapCellFromEvent(event);
  const payload = state.mapPayload;
  const tooltip = document.getElementById("map-tooltip");
  if (!cell || !payload) {
    tooltip.hidden = true;
    return;
  }
  const value = Number(payload.values[cell.index]);
  const geo = payload.geo || {};
  const lat = geo.latitudes ? Number(geo.latitudes[cell.index]) : NaN;
  const lon = geo.longitudes ? Number(geo.longitudes[cell.index]) : NaN;
  tooltip.innerHTML = `
    <strong>${fmt(value, 2)} C</strong><br>
    Lat ${fmt(lat, 4)}, Lon ${fmt(lon, 4)}
  `;
  tooltip.style.left = `${Math.min(cell.rect.width - 210, Math.max(8, event.clientX - cell.rect.left + 12))}px`;
  tooltip.style.top = `${Math.max(8, event.clientY - cell.rect.top + 12)}px`;
  tooltip.hidden = false;
}

function drawHourlyChart() {
  const canvas = document.getElementById("hourly-chart");
  const ctx = canvas.getContext("2d");
  const width = canvas.width;
  const height = canvas.height;
  ctx.clearRect(0, 0, width, height);

  const rows = state.hourly;
  if (!rows.length) {
    ctx.fillStyle = "#5f6b63";
    ctx.fillText("No hourly data", 24, 40);
    return;
  }

  const pad = { left: 50, right: 18, top: 24, bottom: 44 };
  const values = rows.flatMap((row) => [Number(row.mean_pred_c), Number(row.max_pred_c)]).filter(Number.isFinite);
  const min = Math.floor(Math.min(...values) - 1);
  const max = Math.ceil(Math.max(...values) + 1);
  const x = (i) => pad.left + (i * (width - pad.left - pad.right)) / Math.max(1, rows.length - 1);
  const y = (v) => height - pad.bottom - ((v - min) * (height - pad.top - pad.bottom)) / (max - min || 1);

  ctx.strokeStyle = "#cfd8d0";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(pad.left, pad.top);
  ctx.lineTo(pad.left, height - pad.bottom);
  ctx.lineTo(width - pad.right, height - pad.bottom);
  ctx.stroke();

  ctx.fillStyle = "#5f6b63";
  ctx.font = "13px system-ui";
  ctx.fillText(`${max} C`, 10, pad.top + 4);
  ctx.fillText(`${min} C`, 10, height - pad.bottom + 4);

  function line(key, color) {
    ctx.strokeStyle = color;
    ctx.lineWidth = 3;
    ctx.beginPath();
    rows.forEach((row, i) => {
      const px = x(i);
      const py = y(Number(row[key]));
      if (i === 0) ctx.moveTo(px, py);
      else ctx.lineTo(px, py);
    });
    ctx.stroke();
  }

  line("max_pred_c", "#b93632");
  line("mean_pred_c", "#0f7c80");

  ctx.fillStyle = "#1d2420";
  rows.forEach((row, i) => {
    if (i % Math.ceil(rows.length / 6) === 0 || i === rows.length - 1) {
      const hour = String(row.time || "").slice(11, 16);
      ctx.fillText(hour, x(i) - 12, height - 16);
    }
  });

  ctx.fillStyle = "#0f7c80";
  ctx.fillText("Mean", width - 150, 24);
  ctx.fillStyle = "#b93632";
  ctx.fillText("Max", width - 90, 24);
}

function updateEvents() {
  const body = document.getElementById("events-body");
  const events = state.events;
  if (!events.length) {
    body.innerHTML = '<tr><td colspan="5">No detected heatwave events.</td></tr>';
    setText("event-summary", "No active heatwave event.");
    return;
  }
  setText("event-summary", `${events.length} event(s) detected.`);
  body.innerHTML = events.map((event) => `
    <tr>
      <td>${event.start_date || "--"}</td>
      <td>${event.end_date || "--"}</td>
      <td>${event.duration_days || "--"} days</td>
      <td>${event.peak_alert_level || "--"}</td>
      <td>${pct(event.peak_exceed_fraction || 0)}</td>
    </tr>
  `).join("");
}

async function loadAll() {
  const [latest, summary, hourly, events] = await Promise.all([
    fetchJson("/api/latest-alert"),
    fetchJson("/api/summary"),
    fetchJson("/api/hourly"),
    fetchJson("/api/events"),
  ]);
  state.latest = latest;
  state.summary = summary;
  state.hourly = hourly;
  state.events = events;
  updateStatus();
  updateEvents();
  drawHourlyChart();
  await drawMap();
}

document.getElementById("prediction-btn").addEventListener("click", async () => {
  state.layer = "prediction";
  document.getElementById("prediction-btn").classList.add("active");
  document.getElementById("exceedance-btn").classList.remove("active");
  await drawMap();
});

document.getElementById("exceedance-btn").addEventListener("click", async () => {
  state.layer = "exceedance";
  document.getElementById("exceedance-btn").classList.add("active");
  document.getElementById("prediction-btn").classList.remove("active");
  await drawMap();
});

document.getElementById("refresh-btn").addEventListener("click", loadAll);
document.getElementById("map-canvas").addEventListener("mousemove", showMapTooltip);
document.getElementById("map-canvas").addEventListener("mouseleave", () => {
  document.getElementById("map-tooltip").hidden = true;
});

loadAll().catch((error) => {
  console.error(error);
  setText("latest-time", `Unable to load dashboard data: ${error.message}`);
});
