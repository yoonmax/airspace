// In-browser airspace demo: MediaPipe hand tracking, a fingertip cursor and
// pinch / drag / open-palm gestures over a few glass cards. The feature
// extraction, thresholds, One Euro smoothing and pinch-stable pointer are
// ports of the desktop app (airspace/vision/*, airspace/interaction/cursor.py).
// Camera frames never leave the page.

const MP_VERSION = "1.2.2";
const MP = `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MP_VERSION}`;
const MODEL_URL =
  "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";

// Desktop defaults (settings.py / gesture_classifier.py / gesture_state_machine.py)
const PINCH_ENTER = 0.30;
const PINCH_EXIT = 0.45;
const OPEN_PALM_MIN_GAP = 0.55;
const PALM_HOLD_MS = 380;
const FIST_HOLD_MS = 300;
const LOSS_GRACE_MS = 500;

const $ = (id) => document.getElementById(id);
const stage = $("stage");
const video = $("video");
const cursorEl = $("cursor");
const cardsEl = $("cards");
const launcherEl = $("launcher");
const gate = $("gate");
const startBtn = $("start");
const gateNote = $("gate-note");
const pills = { cam: $("pill-cam"), hand: $("pill-hand"), gesture: $("pill-gesture"), fps: $("pill-fps") };

function setPill(el, text, state) {
  el.className = "pill" + (state ? " " + state : "");
  const dot = el.querySelector("i");
  el.textContent = text;
  if (dot) el.prepend(dot);
}

// ------------------------------------------------------------------ filters
function alpha(cutoff, dt) {
  const tau = 1 / (2 * Math.PI * cutoff);
  return 1 / (1 + tau / dt);
}

class OneEuro2D {
  constructor(minCutoff = 1.0, beta = 40.0, dCutoff = 1.0) { // "balanced" preset, normalised units/s
    Object.assign(this, { minCutoff, beta, dCutoff });
    this.reset();
  }
  reset() { this.x = null; this.dx = [0, 0]; this.t = null; }
  filter(v, t) {
    if (this.x === null) { this.x = [...v]; this.dx = [0, 0]; this.t = t; return [...v]; }
    const dt = t - this.t;
    if (dt <= 1e-6) return [...this.x];
    this.t = t;
    const ad = alpha(this.dCutoff, dt);
    this.dx = [0, 1].map((i) => ad * ((v[i] - this.x[i]) / dt) + (1 - ad) * this.dx[i]);
    const cutoff = this.minCutoff + this.beta * Math.hypot(this.dx[0], this.dx[1]);
    const a = alpha(cutoff, dt);
    this.x = [0, 1].map((i) => a * v[i] + (1 - a) * this.x[i]);
    return [...this.x];
  }
}

// Cursor on the fingertip; during a pinch it re-anchors to the index knuckle
// with the fingertip offset from just before the fingers began closing.
class PinchStablePointer {
  static GLIDE_TAU = 0.08;
  static REWIND_S = 0.12;
  static HISTORY_S = 0.4;
  constructor() { this.filter = new OneEuro2D(); this.reset(); }
  reset() {
    this.filter.reset(); this.offset = [0, 0]; this.mode = "tip";
    this.lastRaw = null; this.lastT = null; this.history = [];
  }
  relativeBefore(t) {
    const target = t - PinchStablePointer.REWIND_S;
    let best = null;
    for (const [ts, rel] of this.history) { if (ts <= target) best = rel; else break; }
    return best ?? this.history[0]?.[1] ?? null;
  }
  update(tip, knuckle, pinched, t) {
    const mode = pinched ? "knuckle" : "tip";
    let snap = false;
    if (mode === "knuckle" && this.mode === "tip") {
      this.offset = this.relativeBefore(t) ?? [tip[0] - knuckle[0], tip[1] - knuckle[1]];
      snap = true;
    } else if (mode === "tip" && this.mode === "knuckle" && this.lastRaw) {
      this.offset = [this.lastRaw[0] - tip[0], this.lastRaw[1] - tip[1]];
    } else if (mode === "tip" && this.lastT !== null) {
      const k = Math.exp(-Math.max(t - this.lastT, 0) / PinchStablePointer.GLIDE_TAU);
      this.offset = [this.offset[0] * k, this.offset[1] * k];
    }
    if (mode === "tip") {
      this.history.push([t, [tip[0] - knuckle[0], tip[1] - knuckle[1]]]);
      while (this.history.length && t - this.history[0][0] > PinchStablePointer.HISTORY_S) this.history.shift();
    }
    this.mode = mode;
    const anchor = mode === "knuckle" ? knuckle : tip;
    const raw = [anchor[0] + this.offset[0], anchor[1] + this.offset[1]];
    this.lastRaw = raw;
    if (snap) this.filter.reset();
    this.lastT = t;
    return this.filter.filter(raw, t);
  }
}

// ------------------------------------------------------------------ features
const CHAINS = [[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16], [17, 18, 19, 20]];
const d3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
const d2 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

function handFeatures(lm, aspect) {
  const p = lm.map((q) => [q.x * aspect, q.y, q.z * aspect]);
  const palm = [0, 5, 9, 13, 17];
  const pc = [0, 1].map((i) => palm.reduce((s, k) => s + p[k][i], 0) / palm.length);
  const scale = Math.max((d3(p[0], p[5]) + d3(p[0], p[17]) + d3(p[5], p[17]) * 1.6) / 3, 1e-4);
  const pinch = d3(p[4], p[8]) / scale;
  const ext = [], curl = [];
  CHAINS.forEach((chain, f) => {
    let seg = 0;
    for (let i = 1; i < chain.length; i++) seg += d3(p[chain[i]], p[chain[i - 1]]);
    const s = seg > 1e-6 ? d3(p[chain[0]], p[chain[3]]) / seg : 0;
    const dPalm = d2(p[chain[3]], pc) / scale;
    if (f === 0) {
      const d = d3(p[4], p[5]) / scale;
      ext.push(d > 0.5 && s > 0.8);
      curl.push(d < 0.4);
    } else {
      ext.push(s > 0.8 && d3(p[chain[3]], p[0]) > d3(p[chain[1]], p[0]) * 1.08);
      curl.push(s < 0.6 && dPalm < 0.8);
    }
  });
  const spread = d2(p[8], p[20]) / scale;
  const nExt = ext.slice(1).filter(Boolean).length;
  const nCurl = curl.slice(1).filter(Boolean).length;
  let pose = "hand";
  if (nCurl === 4 && !ext[0]) pose = "fist";
  else if (pinch < 0.35) pose = "pinch";
  else if (nExt === 4 && !curl[0] && pinch > OPEN_PALM_MIN_GAP && spread > 0.8) pose = "open palm";
  else if (ext[1] && (curl.slice(2).filter(Boolean).length >= 2 || nExt === 1)) pose = "point";
  return { pinch, pose, palmCenter: [lm[0].x, lm[0].y].map((v, i) => (v + lm[9][i ? "y" : "x"]) / 2) };
}

// ------------------------------------------------------------------ cards
let zTop = 3;
let accent = "128, 222, 255";
const CARDS = {
  welcome: {
    title: "welcome",
    body: () => `<p>Pinch this title bar and move your hand to drag the card.</p>
      <p style="margin-top:8px;color:var(--muted)">Open your palm and hold it still to open the launcher.</p>`,
  },
  counter: {
    title: "counter",
    body: () => `<div class="count" data-count>0</div>
      <div class="row"><button class="ctl accent" data-act="inc">+1</button><button class="ctl" data-act="reset">reset</button></div>`,
  },
  colors: {
    title: "colors",
    body: () => `<div class="swatches">${[
      "128, 222, 255", "156, 240, 190", "255, 196, 120", "255, 140, 170", "190, 160, 255", "240, 240, 240",
    ].map((c) => `<button class="swatch${c === accent ? " on" : ""}" data-act="color" data-color="${c}" style="background:rgb(${c})" aria-label="accent colour"></button>`).join("")}</div>`,
  },
  clock: {
    title: "clock",
    body: () => `<div class="clock" data-clock>--:--</div><div class="clock-sub" data-date></div>`,
  },
};

function cardPosition(kind) {
  const W = stage.clientWidth, H = stage.clientHeight, narrow = W < 640;
  const cw = narrow ? 210 : 260;
  const spots = narrow ? {
    welcome: [W * 0.04, 64],
    counter: [W * 0.04, 64],
    colors: [W * 0.96 - cw, H - 182],
    clock: [W * 0.96 - cw, H * 0.3],
  } : {
    welcome: [W * 0.05, H * 0.2],
    counter: [W * 0.95 - cw, H * 0.2],
    colors: [W * 0.5 - cw / 2, H * 0.56],
    clock: [W * 0.5 - cw / 2, H * 0.18],
  };
  return spots[kind];
}

function openCard(kind) {
  let card = cardsEl.querySelector(`[data-kind="${kind}"]`);
  if (!card) {
    const def = CARDS[kind];
    card = document.createElement("div");
    card.className = "card";
    card.dataset.kind = kind;
    card.innerHTML = `<div class="card-head"><span class="title">${def.title}</span><button class="x" data-act="close" aria-label="close">×</button></div>
      <div class="card-body">${def.body()}</div>`;
    cardsEl.append(card);
    placeCard(card, ...cardPosition(kind));
    card.querySelector(".card-head").addEventListener("pointerdown", mouseDragStart);
  }
  card.style.zIndex = ++zTop;
  tickClock();
}

function placeCard(card, x, y) {
  const W = stage.clientWidth, H = stage.clientHeight;
  const w = card.offsetWidth || 260, h = card.offsetHeight || 160;
  card.style.left = Math.max(0, Math.min(W - w, x)) + "px";
  card.style.top = Math.max(48, Math.min(H - Math.min(h, 60), y)) + "px";
}

cardsEl.addEventListener("click", (e) => {
  const btn = e.target.closest("[data-act]");
  if (!btn) return;
  const card = btn.closest(".card");
  const act = btn.dataset.act;
  if (act === "close") card.remove();
  if (act === "inc" || act === "reset") {
    const el = card.querySelector("[data-count]");
    el.textContent = act === "inc" ? String(Number(el.textContent) + 1) : "0";
  }
  if (act === "color") {
    accent = btn.dataset.color;
    stage.style.setProperty("--holo", accent);
    card.querySelectorAll(".swatch").forEach((s) => s.classList.toggle("on", s === btn));
  }
});

function tickClock() {
  const now = new Date();
  for (const el of cardsEl.querySelectorAll("[data-clock]"))
    el.textContent = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" });
  for (const el of cardsEl.querySelectorAll("[data-date]"))
    el.textContent = now.toLocaleDateString([], { weekday: "long", month: "long", day: "numeric" });
}
setInterval(tickClock, 1000);

// Mouse: drag by the title bar, double-click empty space for the launcher.
function mouseDragStart(e) {
  if (e.pointerType !== "mouse" && e.pointerType !== "touch" && e.pointerType !== "pen") return;
  if (e.target.closest(".x")) return;
  const head = e.currentTarget; // currentTarget is cleared once the event finishes dispatching
  const card = head.closest(".card");
  const r = card.getBoundingClientRect();
  const off = [e.clientX - r.left, e.clientY - r.top];
  const sr = stage.getBoundingClientRect();
  card.style.zIndex = ++zTop;
  card.classList.add("dragging");
  head.setPointerCapture(e.pointerId);
  const move = (ev) => placeCard(card, ev.clientX - sr.left - off[0], ev.clientY - sr.top - off[1]);
  const up = () => {
    card.classList.remove("dragging");
    head.removeEventListener("pointermove", move);
    head.removeEventListener("pointerup", up);
    head.removeEventListener("pointercancel", up);
  };
  head.addEventListener("pointermove", move);
  head.addEventListener("pointerup", up);
  head.addEventListener("pointercancel", up);
}
stage.addEventListener("dblclick", (e) => {
  if (e.target.closest(".card, .launcher, .gate")) return;
  const r = stage.getBoundingClientRect();
  openLauncher(e.clientX - r.left, e.clientY - r.top);
});

// ------------------------------------------------------------------ launcher
const LAUNCH_ITEMS = [["counter", "+"], ["colors", "◐"], ["clock", "◷"], ["welcome", "?"]];
function openLauncher(x, y) {
  const W = stage.clientWidth, H = stage.clientHeight, R = 172;
  x = Math.max(R, Math.min(W - R, x));
  y = Math.max(R - 40, Math.min(H - R + 40, y));
  launcherEl.style.left = x + "px";
  launcherEl.style.top = y + "px";
  launcherEl.innerHTML = `<div class="ring-bg"></div>` + LAUNCH_ITEMS.map(([kind, icon], i) => {
    const a = -Math.PI / 2 + (i * 2 * Math.PI) / LAUNCH_ITEMS.length;
    return `<button class="item" data-open="${kind}" style="left:${Math.cos(a) * 92}px;top:${Math.sin(a) * 92}px"><div><span>${icon}</span>${kind}</div></button>`;
  }).join("");
  launcherEl.hidden = false;
}
function closeLauncher() { launcherEl.hidden = true; launcherEl.innerHTML = ""; }
launcherEl.addEventListener("click", (e) => {
  const item = e.target.closest("[data-open]");
  if (item) { openCard(item.dataset.open); closeLauncher(); }
});
stage.addEventListener("click", (e) => {
  if (!launcherEl.hidden && !e.target.closest(".launcher")) closeLauncher();
});

// ------------------------------------------------------------------ hand input
const hand = {
  pointer: new PinchStablePointer(),
  pinched: false,
  lostAt: null,
  pos: null,
  hover: null,
  press: null,
  drag: null,
  palmSince: null,
  fistSince: null,
  palmFired: false,
};

function toStage(nx, ny) {
  const W = stage.clientWidth, H = stage.clientHeight;
  const vw = video.videoWidth || 1280, vh = video.videoHeight || 720;
  const s = Math.max(W / vw, H / vh);
  const dw = vw * s, dh = vh * s;
  return [(W - dw) / 2 + (1 - nx) * dw, (H - dh) / 2 + ny * dh]; // video is mirrored on screen
}

function targetAt(x, y) {
  const r = stage.getBoundingClientRect();
  const el = document.elementFromPoint(r.left + x, r.top + y);
  if (!el || !stage.contains(el)) return null;
  const click = el.closest("button");
  if (click) return { kind: "click", el: click };
  const head = el.closest(".card-head");
  if (head) return { kind: "drag", el: head };
  if (el.closest(".card")) return { kind: "card", el: el.closest(".card") };
  return null;
}

function setHover(el) {
  if (hand.hover === el) return;
  hand.hover?.classList.remove("hover");
  hand.hover = el;
  el?.classList.add("hover");
}

function ripple(x, y) {
  const r = document.createElement("div");
  r.className = "ripple";
  r.style.left = x + "px";
  r.style.top = y + "px";
  stage.append(r);
  setTimeout(() => r.remove(), 500);
}

function pinchStart() {
  const [x, y] = hand.pos;
  const t = targetAt(x, y);
  if (!launcherEl.hidden && !(t && t.el.closest(".launcher"))) closeLauncher();
  if (t?.kind === "drag") {
    const card = t.el.closest(".card");
    card.style.zIndex = ++zTop;
    card.classList.add("dragging");
    hand.drag = { card, off: [x - card.offsetLeft, y - card.offsetTop] };
  } else if (t?.kind === "click") {
    hand.press = t.el;
    t.el.classList.add("pressed");
  } else if (t?.kind === "card") {
    t.el.style.zIndex = ++zTop;
  }
}

function pinchEnd(cancel = false) {
  if (hand.drag) {
    hand.drag.card.classList.remove("dragging");
    hand.drag = null;
  }
  if (hand.press) {
    const pressed = hand.press;
    pressed.classList.remove("pressed");
    hand.press = null;
    if (!cancel && hand.pos) {
      const t = targetAt(...hand.pos);
      if (t?.el === pressed) pressed.click();
    }
  }
  if (!cancel && hand.pos) ripple(...hand.pos);
}

function handLost(now) {
  if (hand.lostAt === null) hand.lostAt = now;
  if (hand.pinched && now - hand.lostAt < LOSS_GRACE_MS) return; // brief tracking dropouts keep the pinch
  if (hand.pinched) { hand.pinched = false; pinchEnd(true); }
  if (now - hand.lostAt > 300) {
    cursorEl.hidden = true;
    setHover(null);
    hand.pointer.reset();
    hand.pos = null;
  }
  hand.palmSince = hand.fistSince = null;
  hand.palmFired = false;
  setPill(pills.hand, "no hands");
  setPill(pills.gesture, "idle");
}

function onHand(lm, now) {
  hand.lostAt = null;
  const f = handFeatures(lm, (video.videoWidth || 16) / (video.videoHeight || 9));

  // pinch with hysteresis
  const wasPinched = hand.pinched;
  if (!hand.pinched && f.pinch < PINCH_ENTER) hand.pinched = true;
  else if (hand.pinched && f.pinch > PINCH_EXIT) hand.pinched = false;

  const p = hand.pointer.update([lm[8].x, lm[8].y], [lm[5].x, lm[5].y], hand.pinched, now / 1000);
  hand.pos = toStage(p[0], p[1]);
  const [x, y] = hand.pos;
  cursorEl.hidden = false;
  cursorEl.style.transform = `translate(${x}px, ${y}px)`;

  if (hand.pinched && !wasPinched) pinchStart();
  if (!hand.pinched && wasPinched) pinchEnd();
  if (hand.drag) placeCard(hand.drag.card, x - hand.drag.off[0], y - hand.drag.off[1]);

  // pinch preparation: the ring tightens as the fingers approach
  const prep = Math.max(0, Math.min(1, (PINCH_EXIT + 0.25 - f.pinch) / (PINCH_EXIT + 0.25 - PINCH_ENTER)));
  cursorEl.querySelector(".cursor-ring").style.transform = `scale(${hand.pinched ? 0.62 : 1 - 0.3 * prep})`;
  cursorEl.classList.toggle("pressed", hand.pinched);

  const t = hand.pinched ? null : targetAt(x, y);
  if (!hand.pinched) setHover(t && t.kind !== "card" ? t.el : null);

  // open palm (held) opens the launcher; fist closes it
  const busy = hand.pinched || hand.drag || (t && t.kind !== "card");
  if (f.pose === "open palm" && !busy) {
    hand.palmSince ??= now;
    if (!hand.palmFired && now - hand.palmSince > PALM_HOLD_MS) {
      hand.palmFired = true;
      openLauncher(...toStage(...f.palmCenter));
    }
  } else {
    hand.palmSince = null;
    hand.palmFired = false;
  }
  cursorEl.classList.toggle("palm", hand.palmSince !== null && !hand.palmFired);
  if (f.pose === "fist") {
    hand.fistSince ??= now;
    if (now - hand.fistSince > FIST_HOLD_MS && !launcherEl.hidden) closeLauncher();
  } else hand.fistSince = null;

  setPill(pills.hand, "1 hand", "ok");
  setPill(pills.gesture, hand.drag ? "dragging" : hand.pinched ? "pinch" : f.pose);
}

// ------------------------------------------------------------------ camera + model
let landmarker = null;
let stream = null;
let running = false;
let pausedByVisibility = false;
let lastVideoTime = -1;
let frames = 0;
let fpsT0 = performance.now();

async function loadLandmarker() {
  const { FilesetResolver, HandLandmarker } = await import(`${MP}/vision_bundle.mjs`);
  const fileset = await FilesetResolver.forVisionTasks(`${MP}/wasm`);
  const options = (delegate) => ({
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode: "VIDEO",
    numHands: 1,
    minHandDetectionConfidence: 0.6,
    minHandPresenceConfidence: 0.5,
    minTrackingConfidence: 0.5,
  });
  let lmk;
  try {
    lmk = await HandLandmarker.createFromOptions(fileset, options("GPU"));
  } catch (err) {
    console.warn("GPU delegate unavailable, using CPU", err);
    lmk = await HandLandmarker.createFromOptions(fileset, options("CPU"));
  }
  // The first detections compile GPU shaders (seconds); do that now, not on live frames.
  const warm = document.createElement("canvas");
  warm.width = 640;
  warm.height = 360;
  warm.getContext("2d").fillRect(0, 0, 640, 360);
  const t0 = performance.now() - 2000; // keep timestamps below the first live frame
  lmk.detectForVideo(warm, t0);
  lmk.detectForVideo(warm, t0 + 33);
  return lmk;
}

async function openCamera() {
  stream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: "user" },
    audio: false,
  });
  video.srcObject = stream;
  await video.play();
}

function stopCamera() {
  running = false;
  stream?.getTracks().forEach((t) => t.stop());
  stream = null;
  video.srcObject = null;
  stage.classList.remove("live");
  if (hand.pinched) { hand.pinched = false; pinchEnd(true); }
  cursorEl.hidden = true;
  setHover(null);
}

function showError(msg) {
  gate.hidden = false;
  gateNote.textContent = msg + " ";
  gateNote.classList.add("err");
  const alt = document.createElement("button");
  alt.className = "linkish";
  alt.textContent = "Use the mouse instead";
  alt.onclick = () => { gate.hidden = true; };
  gateNote.append(alt);
  startBtn.disabled = false;
  startBtn.textContent = "Try again";
  setPill(pills.cam, "camera off", "err");
}

async function start() {
  if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
    showError("This browser can't open the camera on this page. Try a current Chrome, Edge, Safari or Firefox.");
    return;
  }
  startBtn.disabled = true;
  gateNote.classList.remove("err");
  try {
    startBtn.textContent = "Waiting for camera…";
    setPill(pills.cam, "waiting for permission", "warn");
    await openCamera();
  } catch (err) {
    const denied = err?.name === "NotAllowedError" || err?.name === "SecurityError";
    showError(denied
      ? "Camera access was blocked. Allow it for this site in your browser's settings, then try again."
      : err?.name === "NotFoundError" ? "No camera was found." : `The camera couldn't start (${err?.name || err}).`);
    return;
  }
  try {
    if (!landmarker) {
      startBtn.textContent = "Loading hand tracking…";
      setPill(pills.cam, "loading model", "warn");
      landmarker = await loadLandmarker();
    }
  } catch (err) {
    stopCamera();
    showError(`Hand tracking couldn't load (${err?.message || err}). Check your connection and try again.`);
    return;
  }
  gate.hidden = true;
  stage.classList.add("live");
  setPill(pills.cam, `camera ${video.videoWidth}×${video.videoHeight}`, "ok");
  running = true;
  requestAnimationFrame(loop);
}

function loop() {
  if (!running) return;
  const now = performance.now();
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    let res = null;
    try { res = landmarker.detectForVideo(video, now); } catch (err) { console.error(err); }
    frames++;
    const lm = res?.landmarks?.[0];
    if (lm) onHand(lm, now); else handLost(now);
  }
  if (now - fpsT0 > 1000) {
    setPill(pills.fps, `${Math.round((frames * 1000) / (now - fpsT0))} fps`);
    frames = 0;
    fpsT0 = now;
  }
  requestAnimationFrame(loop);
}

// The camera is released whenever the tab is hidden and reopened when it returns.
document.addEventListener("visibilitychange", async () => {
  if (document.hidden && running) {
    pausedByVisibility = true;
    stopCamera();
    setPill(pills.cam, "paused", "warn");
  } else if (!document.hidden && pausedByVisibility) {
    pausedByVisibility = false;
    try {
      await openCamera();
      stage.classList.add("live");
      setPill(pills.cam, `camera ${video.videoWidth}×${video.videoHeight}`, "ok");
      running = true;
      requestAnimationFrame(loop);
    } catch {
      showError("The camera couldn't restart. Press the button to try again.");
    }
  }
});

startBtn.addEventListener("click", start);
$("mouse-only").addEventListener("click", () => {
  gate.hidden = true;
  setPill(pills.gesture, "mouse");
});
// Lay the first cards out once the stage has a real size; afterwards keep them inside it.
let laidOut = false;
function layoutCards() {
  if (!stage.clientWidth) return;
  if (!laidOut) {
    laidOut = true;
    (stage.clientWidth < 640 ? ["counter", "colors"] : ["welcome", "counter", "colors"]).forEach(openCard);
  } else {
    cardsEl.querySelectorAll(".card").forEach((c) => placeCard(c, c.offsetLeft, c.offsetTop));
  }
}
new ResizeObserver(layoutCards).observe(stage);
layoutCards();
