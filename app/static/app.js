"use strict";

/* ====================== tiện ích ====================== */
const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const GROUP_COLOR = {
  cam: "#e5484d", hieu_lenh: "#2f6fed", nguy_hiem: "#e89b0c", chi_dan: "#12a594", den_tin_hieu: "#8e4ec6",
};

const state = {
  source: "camera",       // camera | video | image
  running: false,
  loopToken: 0,
  stream: null,
  detections: [],
  classes: [],            // từ /api/classes
  groups: {},
  seen: new Map(),        // class_id -> {count, lastAt, lastSpokenAt}
  fps: 0,
};

/* ====================== khởi tạo ====================== */
async function init() {
  setupTabs();
  setupControls();
  await Promise.all([loadServerStatus(), loadClasses()]);
  listCameras();
}

async function loadServerStatus() {
  const chip = $("serverStatus");
  try {
    const info = await (await fetch("/api/models")).json();
    const ready = Object.values(info.available).some(Boolean);
    chip.textContent = ready ? `Máy chủ sẵn sàng · ${info.device}` : "Chưa có mô hình — hãy chạy các bước huấn luyện";
    chip.className = "status " + (ready ? "ok" : "bad");
    for (const opt of $("modelSelect").options) {
      opt.disabled = !info.available[opt.value];
    }
    if ($("modelSelect").selectedOptions[0]?.disabled) {
      const firstOk = [...$("modelSelect").options].find((o) => !o.disabled);
      if (firstOk) $("modelSelect").value = firstOk.value;
    }
  } catch {
    chip.textContent = "Không kết nối được máy chủ";
    chip.className = "status bad";
  }
}

function setupTabs() {
  document.querySelectorAll(".tab").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((b) => b.classList.toggle("active", b === btn));
      document.querySelectorAll(".panel").forEach((p) => p.classList.toggle("active", p.id === "tab-" + btn.dataset.tab));
      if (btn.dataset.tab === "evaluation") loadEvaluation();
    });
  });
}

function setupControls() {
  $("sourceSwitch").addEventListener("click", (e) => {
    const btn = e.target.closest("button");
    if (btn) setSource(btn.dataset.source);
  });
  $("startBtn").addEventListener("click", () => (state.running ? stopAll() : startCamera()));
  $("fileInput").addEventListener("change", (e) => {
    const file = e.target.files[0];
    if (file) loadFile(file);
    e.target.value = "";
  });
  $("confRange").addEventListener("input", () => {
    $("confValue").textContent = $("confRange").value + "%";
    if (state.source === "image") predictStill();
  });
  $("modelSelect").addEventListener("change", () => state.source === "image" && predictStill());
  $("sizeSelect").addEventListener("change", () => state.source === "image" && predictStill());
  $("singleToggle").addEventListener("change", () => state.source === "image" && predictStill());
  $("speakToggle").addEventListener("change", () => {
    if ($("speakToggle").checked) speak("Đã bật đọc to biển báo");
  });
  $("clearSeen").addEventListener("click", () => { state.seen.clear(); renderSeen(); });

  const stage = $("stage");
  ["dragenter", "dragover"].forEach((ev) => stage.addEventListener(ev, (e) => { e.preventDefault(); stage.classList.add("drag"); }));
  ["dragleave", "drop"].forEach((ev) => stage.addEventListener(ev, (e) => { e.preventDefault(); stage.classList.remove("drag"); }));
  stage.addEventListener("drop", (e) => { const f = e.dataTransfer.files[0]; if (f) loadFile(f); });
  window.addEventListener("resize", () => { fitOverlay(); drawOverlay(); });
  $("video").addEventListener("loadedmetadata", () => showMedia($("video").videoWidth, $("video").videoHeight));
  $("photo").addEventListener("load", () => { showMedia($("photo").naturalWidth, $("photo").naturalHeight); predictStill(); });
}

/* ====================== nguồn ảnh ====================== */
function setSource(src) {
  if (src === state.source) return;
  stopAll();
  state.source = src;
  document.querySelectorAll("#sourceSwitch button").forEach((b) => b.classList.toggle("active", b.dataset.source === src));
  const isCam = src === "camera";
  $("cameraSelect").hidden = !isCam;
  $("startBtn").hidden = !isCam;
  $("fileBtn").hidden = isCam;
  $("fileInput").accept = src === "video" ? "video/*" : "image/*";
  $("fileBtnText").textContent = src === "video" ? "Chọn video…" : "Chọn ảnh…";
}

async function listCameras() {
  try {
    const devices = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === "videoinput");
    const sel = $("cameraSelect");
    sel.innerHTML = devices.length
      ? devices.map((d, i) => `<option value="${esc(d.deviceId)}">${esc(d.label || "Camera " + (i + 1))}</option>`).join("")
      : '<option value="">Không thấy camera</option>';
  } catch {
    $("cameraSelect").innerHTML = '<option value="">Không truy cập được camera</option>';
  }
}

async function startCamera() {
  if (!navigator.mediaDevices?.getUserMedia) {
    alert("Trình duyệt không cho dùng camera ở địa chỉ này. Hãy mở bằng http://localhost:8000 hoặc https.");
    return;
  }
  try {
    const deviceId = $("cameraSelect").value;
    state.stream = await navigator.mediaDevices.getUserMedia({
      video: { ...(deviceId ? { deviceId: { exact: deviceId } } : {}), width: { ideal: 1280 }, height: { ideal: 720 } },
      audio: false,
    });
  } catch (err) {
    alert("Không mở được camera: " + err.message);
    return;
  }
  await listCameras();           // sau khi được cấp quyền thì mới có tên camera
  const video = $("video");
  video.srcObject = state.stream;
  video.loop = false;
  $("photo").hidden = true;
  video.hidden = false;
  await video.play();
  beginLoop();
}

function loadFile(file) {
  stopAll();
  const url = URL.createObjectURL(file);
  if (file.type.startsWith("video/")) {
    if (state.source !== "video") setSource("video");
    const video = $("video");
    $("photo").hidden = true;
    video.hidden = false;
    video.srcObject = null;
    video.src = url;
    video.loop = true;
    video.play().then(beginLoop);
  } else if (file.type.startsWith("image/")) {
    if (state.source !== "image") setSource("image");
    $("video").hidden = true;
    $("photo").hidden = false;
    $("photo").src = url;
  }
}

function stopAll() {
  state.loopToken++;
  state.running = false;
  state.fps = 0;
  state.stream?.getTracks().forEach((t) => t.stop());
  state.stream = null;
  const video = $("video");
  video.pause();
  video.removeAttribute("src");
  video.srcObject = null;
  $("startBtn").textContent = "Bật camera";
  $("startBtn").classList.remove("stop");
  state.detections = [];
  drawOverlay();
  renderDetections();
  $("stage").classList.remove("has-media");
  $("mFps").textContent = $("mMs").textContent = "–";
  $("mCount").textContent = "0";
}

function showMedia(w, h) {
  $("stage").style.aspectRatio = `${w} / ${h}`;
  $("stage").style.setProperty("--ar", w / h);   // để CSS giới hạn chiều cao theo cửa sổ
  $("stage").classList.add("has-media");
  fitOverlay();
}

function fitOverlay() {
  const c = $("overlay");
  const dpr = window.devicePixelRatio || 1;
  c.width = Math.round(c.clientWidth * dpr);
  c.height = Math.round(c.clientHeight * dpr);
}

/* ====================== vòng lặp nhận dạng ====================== */
function currentMedia() {
  return state.source === "image" ? $("photo") : $("video");
}

function beginLoop() {
  state.running = true;
  $("startBtn").textContent = "Dừng";
  $("startBtn").classList.add("stop");
  const token = ++state.loopToken;
  (async () => {
    let lastStart = 0;
    while (state.running && token === state.loopToken) {
      const t0 = performance.now();
      if (lastStart) {
        const inst = 1000 / (t0 - lastStart);
        state.fps = state.fps ? state.fps * 0.8 + inst * 0.2 : inst;
        $("mFps").textContent = state.fps.toFixed(1);
      }
      lastStart = t0;
      const video = $("video");
      if (video.readyState >= 2 && !video.paused) {
        try {
          await detectOnce(video, token);
        } catch (err) {
          showError(err.message);
          await sleep(1000);
        }
      }
      await sleep(Math.max(0, 50 - (performance.now() - t0)));
    }
  })();
}

async function predictStill() {
  const photo = $("photo");
  if (!photo.complete || !photo.naturalWidth) return;
  try {
    await detectOnce(photo, state.loopToken);
  } catch (err) {
    showError(err.message);
  }
}

async function detectOnce(media, token) {
  const imgsz = Number($("sizeSelect").value);
  const sw = media.videoWidth || media.naturalWidth;
  const sh = media.videoHeight || media.naturalHeight;
  const scale = Math.min(1, imgsz / Math.max(sw, sh));
  const cv = document.createElement("canvas");
  cv.width = Math.round(sw * scale);
  cv.height = Math.round(sh * scale);
  cv.getContext("2d").drawImage(media, 0, 0, cv.width, cv.height);
  const jpeg = canvasToJpeg(cv);

  const q = new URLSearchParams({
    model: $("modelSelect").value,
    conf: ($("confRange").value / 100).toFixed(2),
    imgsz,
    single: $("singleToggle").checked,
  });
  const resp = await fetch("/api/predict?" + q, { method: "POST", body: jpeg });
  if (!resp.ok) throw new Error((await resp.json().catch(() => ({}))).detail || "Lỗi máy chủ " + resp.status);
  const data = await resp.json();
  if (token !== state.loopToken) return;   // đã dừng/đổi nguồn trong lúc chờ
  state.detections = data.detections;
  $("mMs").textContent = data.ms.toFixed(0);
  $("mCount").textContent = data.detections.length;
  drawOverlay();
  renderDetections();
  trackSeen(data.detections);
}

// toDataURL chạy đồng bộ, ổn định hơn toBlob trên các trình duyệt hạn chế vẽ nền
function canvasToJpeg(canvas) {
  const bin = atob(canvas.toDataURL("image/jpeg", 0.88).split(",")[1]);
  const bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  return bytes;
}

function showError(msg) {
  $("detections").innerHTML = `<li class="muted">⚠ ${esc(msg)}</li>`;
}

/* ====================== vẽ khung ====================== */
function drawOverlay() {
  const c = $("overlay");
  const ctx = c.getContext("2d");
  ctx.clearRect(0, 0, c.width, c.height);
  const dpr = window.devicePixelRatio || 1;
  ctx.lineWidth = 3 * dpr;
  ctx.font = `600 ${13 * dpr}px "Segoe UI", sans-serif`;
  ctx.textBaseline = "middle";
  for (const d of state.detections) {
    const [x1, y1, x2, y2] = d.box;
    const x = x1 * c.width, y = y1 * c.height, w = (x2 - x1) * c.width, h = (y2 - y1) * c.height;
    const color = GROUP_COLOR[d.group] || "#fff";
    ctx.strokeStyle = color;
    ctx.strokeRect(x, y, w, h);
    const text = `${d.name_vi} ${Math.round(d.conf * 100)}%`;
    const pad = 5 * dpr, th = 20 * dpr;
    const tw = ctx.measureText(text).width + pad * 2;
    const lx = Math.min(Math.max(0, x), Math.max(0, c.width - tw));
    const ly = y - th >= 0 ? y - th : y + h;
    ctx.fillStyle = color;
    ctx.fillRect(lx, ly, tw, th);
    ctx.fillStyle = "#fff";
    ctx.fillText(text, lx + pad, ly + th / 2);
  }
}

/* ====================== danh sách bên phải ====================== */
function groupBadge(d) {
  return `<span class="badge" style="background:${GROUP_COLOR[d.group] || "#666"}">${esc(d.group_label)}</span>`;
}

function renderDetections() {
  const ul = $("detections");
  if (!state.detections.length) {
    ul.innerHTML = '<li class="muted">Chưa có biển báo nào</li>' + (state.source === "image" && $("photo").src && !$("singleToggle").checked
      ? '<li class="muted">Không tìm thấy khung biển báo. Nếu ảnh chỉ có một biển báo (hình vẽ, ảnh cận cảnh), hãy bật "Ảnh chỉ chứa một biển báo".</li>' : "");
    return;
  }
  ul.innerHTML = state.detections.map((d) => `
    <li>
      <img src="/samples/${d.class_id}.jpg" alt="">
      <div class="det-body">
        <div class="det-name">${esc(d.name_vi)}</div>
        <div class="det-sub">${groupBadge(d)}#${d.class_id} · ${Math.round(d.conf * 100)}%</div>
        <div class="bar"><i style="width:${Math.round(d.conf * 100)}%"></i></div>
        ${d.alternatives ? `<div class="det-sub">Cũng có thể là: ${d.alternatives.map((a) => esc(a.name_vi) + " " + Math.round(a.conf * 100) + "%").join("; ")}${d.model_used ? " (phân loại bằng " + d.model_used.toUpperCase() + ")" : ""}</div>` : ""}
      </div>
    </li>`).join("");
}

function trackSeen(detections) {
  const now = Date.now();
  for (const d of detections) {
    const rec = state.seen.get(d.class_id) || { count: 0, lastAt: 0, lastSpokenAt: 0, d };
    // coi là "lần gặp mới" nếu biển này vắng mặt hơn 2 giây
    if (now - rec.lastAt > 2000) {
      rec.count++;
      if ($("speakToggle").checked && now - rec.lastSpokenAt > 6000) {
        speak(d.name_vi);
        rec.lastSpokenAt = now;
      }
    }
    rec.lastAt = now;
    rec.d = d;
    state.seen.set(d.class_id, rec);
  }
  renderSeen();
}

function renderSeen() {
  const ul = $("seen");
  const rows = [...state.seen.values()].sort((a, b) => b.lastAt - a.lastAt).slice(0, 30);
  ul.innerHTML = rows.length
    ? rows.map((r) => `
      <li>
        <img src="/samples/${r.d.class_id}.jpg" alt="">
        <div class="det-body"><div class="det-name">${esc(r.d.name_vi)}</div><div class="det-sub">${groupBadge(r.d)}</div></div>
        <span class="count">×${r.count}</span>
      </li>`).join("")
    : '<li class="muted">Chưa có lịch sử</li>';
}

function speak(text) {
  if (!("speechSynthesis" in window)) return;
  const u = new SpeechSynthesisUtterance(text);
  u.lang = "vi-VN";
  const voice = speechSynthesis.getVoices().find((v) => v.lang.toLowerCase().startsWith("vi"));
  if (voice) u.voice = voice;
  speechSynthesis.speak(u);
}

/* ====================== danh mục ====================== */
async function loadClasses() {
  const data = await (await fetch("/api/classes")).json();
  state.classes = data.classes;
  state.groups = data.groups;
  renderCatalog("");
  $("catalogSearch").addEventListener("input", (e) => renderCatalog(e.target.value));
}

function renderCatalog(filter) {
  const f = filter.trim().toLowerCase();
  const match = (c) => !f || `${c.id} ${c.name_vi} ${c.name_en}`.toLowerCase().includes(f);
  let html = "";
  for (const [key, label] of Object.entries(state.groups)) {
    const items = state.classes.filter((c) => c.group === key && match(c));
    if (!items.length) continue;
    html += `<h3 class="group-title"><span class="dot" style="background:${GROUP_COLOR[key]}"></span>${esc(label)} <small>(${items.length})</small></h3><div class="grid">`;
    html += items.map((c) => `
      <div class="sign-card">
        <img src="/samples/${c.id}.jpg" alt="" loading="lazy">
        <div><b>${esc(c.name_vi)}</b><small>#${c.id} · ${esc(c.name_en)}${c.train_samples ? " · " + c.train_samples + " mẫu" : ""}</small></div>
      </div>`).join("");
    html += "</div>";
  }
  $("catalog").innerHTML = html || '<p class="muted">Không tìm thấy biển báo phù hợp</p>';
}

/* ====================== đánh giá ====================== */
const FIGURE_CAPTIONS = {
  "class_distribution.png": "Phân bố số mẫu theo lớp (train/val/test)",
  "sample_grid.png": "Ảnh mẫu của từng lớp",
  "cnn_training_curves.png": "Quá trình huấn luyện CNN",
  "confusion_matrix_cnn.png": "Ma trận nhầm lẫn – CNN (tập test)",
  "confusion_matrix_svm.png": "Ma trận nhầm lẫn – SVM (tập test)",
  "confusion_matrix_knn.png": "Ma trận nhầm lẫn – KNN (tập test)",
  "detector_results.png": "Quá trình huấn luyện YOLOv8",
  "detector_confusion_matrix.png": "Ma trận nhầm lẫn – YOLOv8 (tập test)",
  "detector_PR_curve.png": "Đường Precision–Recall – YOLOv8 (tập test)",
};
const MODEL_LABEL = { knn: "KNN (HOG + PCA)", svm: "SVM RBF (HOG + PCA)", cnn: "CNN" };
let evaluationLoaded = false;

async function loadEvaluation() {
  if (evaluationLoaded) return;
  const box = $("evaluation");
  const r = await (await fetch("/api/report")).json();
  const pct = (v) => (v == null ? "–" : (v * 100).toFixed(2) + "%");
  let html = "";

  if (r.dataset) {
    const s = r.dataset.splits;
    html += `<div class="stat-grid">
      <div><b>${s.train.samples.toLocaleString("vi")}</b><small>mẫu train (${s.train.images.toLocaleString("vi")} ảnh)</small></div>
      <div><b>${s.val.samples.toLocaleString("vi")}</b><small>mẫu val</small></div>
      <div><b>${s.test.samples.toLocaleString("vi")}</b><small>mẫu test</small></div>
      <div><b>${s.train.classes_present}</b><small>lớp</small></div></div>`;
  }

  if (r.classifiers?.models) {
    const rows = Object.entries(r.classifiers.models);
    const bestAcc = Math.max(...rows.map(([, m]) => m.accuracy));
    html += `<div class="card"><h2>Bộ phân loại trên ảnh biển báo đã cắt (tập test, ${r.classifiers.n_test} mẫu)</h2><table>
      <tr><th>Mô hình</th><th>Accuracy</th><th>Precision</th><th>Recall</th><th>F1</th><th>Top-5</th><th title="Chỉ tính các mẫu test không có ảnh gần trùng trong tập train">Acc (mẫu lạ)</th><th>ms/biển</th></tr>
      ${rows.map(([k, m]) => `<tr class="${m.accuracy === bestAcc ? "best" : ""}"><td>${esc(MODEL_LABEL[k] || k)}</td>
        <td>${pct(m.accuracy)}</td><td>${pct(m.precision_macro)}</td><td>${pct(m.recall_macro)}</td>
        <td>${pct(m.f1_macro)}</td><td>${pct(m.top5_accuracy)}</td><td>${pct(m.accuracy_novel)}</td><td>${m.ms_per_crop.toFixed(2)}</td></tr>`).join("")}
      </table><small>Precision / Recall / F1 là trung bình macro trên các lớp. "Acc (mẫu lạ)" chỉ tính ${r.classifiers.n_novel} mẫu test không có ảnh gần trùng trong tập train (độ tương đồng ≤ ${r.classifiers.novel_similarity_threshold}) — số liệu khắt khe hơn, phản ánh khả năng tổng quát hoá.</small></div>`;
  }

  if (r.detector) {
    const d = r.detector;
    html += `<div class="card"><h2>YOLOv8 – phát hiện + nhận dạng (tập test, ${d.n_images} ảnh)</h2><table>
      <tr><th>Mô hình</th><th>Precision</th><th>Recall</th><th>mAP@0.5</th><th>mAP@0.5:0.95</th></tr>
      <tr><td>${esc(d.model)}</td><td>${pct(d.precision)}</td><td>${pct(d.recall)}</td><td>${pct(d.map50)}</td><td>${pct(d.map50_95)}</td></tr></table></div>`;
  }

  if (!r.classifiers && !r.detector) {
    html += '<p class="muted">Chưa có số liệu đánh giá — chạy <code>scripts/step4_evaluate.py</code>.</p>';
  }

  const figs = r.images.filter((n) => FIGURE_CAPTIONS[n]);
  if (figs.length) {
    html += '<div class="figures">' + figs.map((n) =>
      `<figure><img src="/reports/${esc(n)}" alt="${esc(FIGURE_CAPTIONS[n])}" loading="lazy"><figcaption>${esc(FIGURE_CAPTIONS[n])}</figcaption></figure>`).join("") + "</div>";
  }
  box.innerHTML = html;
  evaluationLoaded = r.classifiers && r.detector ? true : false;   // chưa đủ số liệu thì lần sau tải lại
}

init();
