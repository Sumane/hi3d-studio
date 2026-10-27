import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";
import { GLTFLoader } from "three/addons/loaders/GLTFLoader.js";

const $ = (s) => document.querySelector(s);
const api = (p, o) => fetch(p, o).then(async (r) => { if (!r.ok) throw new Error((await r.text()) || r.statusText); return r.json(); });

/* ------------------------------------------------------------------ inputs */
const views = []; // {file, url}
let front = 0;

function renderThumbs() {
  const box = $("#thumbs");
  box.innerHTML = "";
  views.forEach((v, i) => {
    const d = document.createElement("div");
    d.className = "thumb" + (i === front ? " front" : "");
    d.innerHTML = `<img src="${v.url}" alt=""><button class="star" title="Use as front view">★</button><button class="x" title="Remove">✕</button><span class="idx">${i === front ? "front" : "view " + (i + 1)}</span>`;
    d.querySelector(".star").onclick = () => { front = i; renderThumbs(); };
    d.querySelector(".x").onclick = () => { URL.revokeObjectURL(v.url); views.splice(i, 1); if (front >= views.length) front = 0; renderThumbs(); };
    box.appendChild(d);
  });
  $("#view-count").textContent = `${views.length} / 8`;
  const multi = views.length > 1;
  $("#coarse_mode").value = multi && $("#coarse_mode").value === "single" ? "multidiffusion" : (!multi && ["multidiffusion", "stochastic"].includes($("#coarse_mode").value) ? "single" : $("#coarse_mode").value);
  for (const o of $("#coarse_mode").options) if (["multidiffusion", "stochastic"].includes(o.value)) o.disabled = !multi;
  for (const o of $("#fusion").options) if (o.value !== "front") o.disabled = !multi;
  if (!multi) $("#fusion").value = "front";
  syncForm();
}
function addFiles(list) {
  for (const f of list) {
    if (!f.type.startsWith("image/") || views.length >= 8) continue;
    views.push({ file: f, url: URL.createObjectURL(f) });
  }
  renderThumbs();
}
const drop = $("#drop");
drop.onclick = () => $("#file").click();
$("#file").onchange = (e) => { addFiles(e.target.files); e.target.value = ""; };
drop.ondragover = (e) => { e.preventDefault(); drop.classList.add("over"); };
drop.ondragleave = () => drop.classList.remove("over");
drop.ondrop = (e) => { e.preventDefault(); drop.classList.remove("over"); addFiles(e.dataTransfer.files); };
document.addEventListener("paste", (e) => addFiles([...e.clipboardData.files]));

function syncForm() {
  const mode = $("#coarse_mode").value;
  $("#coarse-upload").classList.toggle("hidden", mode !== "upload");
  $("#coarse_res").disabled = $("#bg_removal").disabled = mode === "upload";
  $("#mv-note").classList.toggle("hidden", !["multidiffusion", "stochastic"].includes(mode));
  $("#refine-opts").style.opacity = $("#refine").checked ? 1 : 0.75;
  $("#run").textContent = $("#refine").checked ? "Generate" : "Generate coarse (TRELLIS only)";
  $("#fw-row").classList.toggle("hidden", $("#fusion").value !== "multidiffusion");
  $("#tex-opts").classList.toggle("hidden", !$("#texture").checked);
  $("#fw-out").textContent = (+$("#front_weight").value).toFixed(1) + "×";
}
["#coarse_mode", "#refine", "#fusion", "#front_weight", "#texture"].forEach((s) => $(s).addEventListener("input", syncForm));

function params() {
  const mode = $("#coarse_mode").value;
  return {
    coarse_mode: mode, coarse_res: +$("#coarse_res").value, bg_removal: $("#bg_removal").value,
    refine: $("#refine").checked, hi3d_res: +$("#hi3d_res").value, steps: +$("#steps").value,
    fusion: $("#fusion").value, front_weight: +$("#front_weight").value, remove_bg: $("#remove_bg").value,
    seed: +$("#seed").value, colour: $("#colour").checked, hybrid: $("#hybrid").checked, texture: $("#texture").checked,
    tex_faces: +$("#tex_faces").value, atlas: +$("#atlas").value, offload: $("#offload").checked, torch_conv: $("#torch_conv").checked, front,
    gpu: $("#gpu-select").value === "" ? null : +$("#gpu-select").value,
  };
}

/* ------------------------------------------------------------------ jobs */
let activeJob = null, selectedJob = null, pollTimer = null;

$("#run").onclick = async () => {
  if (!views.length) return flash("Add at least one reference image — Hi3D conditions on it.");
  const p = params();
  if (p.coarse_mode === "upload" && !$("#coarse_file").files[0]) return flash("Choose a coarse mesh file.");
  const fd = new FormData();
  fd.append("params", JSON.stringify(p));
  views.forEach((v) => fd.append("images", v.file, v.file.name));
  if (p.coarse_mode === "upload") fd.append("coarse_mesh", $("#coarse_file").files[0]);
  if ($("#fit_file").files[0]) fd.append("fit_mesh", $("#fit_file").files[0]);
  try {
    const { id } = await api("/api/jobs", { method: "POST", body: fd });
    activeJob = id; selectJob(id); poll();
  } catch (e) { flash(e.message, true); }
};
$("#pp_run").onclick = async () => {
  if (!selectedJob) return flash("Select a finished result in History first.", true);
  const body = { decimate: +$("#pp_decimate").value, texture: $("#pp_texture").value === "1",
                 tex_faces: +$("#pp_tex_faces").value, atlas: +$("#pp_atlas").value, hybrid: $("#hybrid").checked };
  try { await api(`/api/jobs/${selectedJob}/post`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
        activeJob = selectedJob; poll(); } catch (e) { flash(e.message, true); }
};
const REFINE_KEYS = ["hi3d_res", "steps", "fusion", "front_weight", "remove_bg", "seed", "colour", "hybrid",
                     "texture", "tex_faces", "atlas", "offload", "torch_conv", "gpu"];
$("#refine_sel").onclick = async () => {
  if (!selectedJob) return flash("Select a result in History first.", true);
  const p = params(), body = {};
  for (const k of REFINE_KEYS) body[k] = p[k];
  try {
    await api(`/api/jobs/${selectedJob}/refine`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) });
    activeJob = selectedJob; poll();
  } catch (e) { flash(e.message, true); }
};
function updateRefineButton(j) {
  const b = $("#refine_sel");
  const ok = !!j && !!j.outputs?.coarse && !["running", "queued"].includes(j.status);
  b.disabled = !ok;
  b.textContent = !ok ? "Refine selected result with Hi3D"
    : j.outputs.refined ? `Re-run Hi3D on ${j.id.slice(9, 15)} (replaces its refined files)` : `Refine ${j.id.slice(9, 15)} with Hi3D`;
}
$("#cancel").onclick =() => activeJob && api(`/api/jobs/${activeJob}/cancel`, { method: "POST" });
$("#quit").onclick = async () => { if (confirm("Stop Hi3D Studio? Running jobs will be killed.")) { await api("/api/shutdown", { method: "POST" }).catch(() => {}); window.close(); } };

function flash(msg, err) {
  $("#progress").classList.remove("hidden");
  $("#prog-msg").textContent = msg;
  $("#prog-msg").classList.toggle("err", !!err);
}

async function poll() {
  clearTimeout(pollTimer);
  if (!activeJob) return;
  let j;
  try { j = await api(`/api/jobs/${activeJob}`); } catch { pollTimer = setTimeout(poll, 2000); return; }
  const running = j.status === "running" || j.status === "queued";
  $("#progress").classList.remove("hidden");
  $("#bar-fill").style.width = (j.progress * 100).toFixed(1) + "%";
  $("#prog-msg").textContent = j.status === "error" ? j.error : j.message;
  $("#prog-msg").classList.toggle("err", j.status === "error");
  $("#run").disabled = running;
  $("#cancel").classList.toggle("hidden", !running);
  if (selectedJob === j.id) { showLog(j.log); updateRefineButton(j); }
  if (selectedJob === j.id && Object.keys(j.outputs).length !== Object.keys(loadedOutputs).length) showJob(j);
  if (running) pollTimer = setTimeout(poll, 1000);
  else { activeJob = null; refreshJobs(); if (j.status === "done") showJob(j); }
}
function showLog(lines) {
  const el = $("#log");
  const atBottom = el.scrollTop + el.clientHeight >= el.scrollHeight - 20;
  el.textContent = (lines || []).join("\n");
  if (atBottom) el.scrollTop = el.scrollHeight;
}

async function refreshJobs() {
  let jobs;
  try { jobs = await api("/api/jobs"); } catch { return; }
  const box = $("#jobs");
  box.innerHTML = "";
  for (const j of jobs) {
    const d = document.createElement("div");
    d.className = "job" + (j.id === selectedJob ? " sel" : "");
    const img = j.images[j.params.front || 0];
    const p = j.params;
    const desc = p.coarse_mode === "upload" ? "own mesh" : `${p.coarse_mode === "single" ? "1 view" : j.images.length + " views"} · T${p.coarse_res}`;
    d.innerHTML = `<img src="${img ? `/api/jobs/${j.id}/files/inputs/${img}` : "icon.svg"}" alt="">
      <div><div class="t">${j.id.slice(4, 15).replace("-", " ")}</div>
      <div class="s ${j.status}">${j.status === "done" ? desc + (p.refine ? ` → H${p.hi3d_res}` : " · coarse only") : j.status}</div>
      ${j.stats?.seconds ? `<div class="s">${Math.round(j.stats.seconds)} s · ${(j.stats.faces / 1e6).toFixed(2)}M faces</div>` : ""}</div>
      <button class="del" title="Delete">✕</button>`;
    d.onclick = () => selectJob(j.id);
    d.querySelector(".del").onclick = async (e) => { e.stopPropagation(); if (confirm("Delete this result?")) { await api(`/api/jobs/${j.id}`, { method: "DELETE" }).catch((x) => alert(x.message)); refreshJobs(); } };
    box.appendChild(d);
    if (!activeJob && (j.status === "running" || j.status === "queued")) { activeJob = j.id; poll(); }
  }
}
async function selectJob(id) {
  selectedJob = id;
  refreshJobs();
  const j = await api(`/api/jobs/${id}`);
  showLog(j.log);
  showJob(j);
}

async function refreshStatus() {
  try {
    const s = await api("/api/status");
    const sel = $("#gpu-select");
    if (!sel.options.length && s.gpus.length) {
      let saved = null;
      try { saved = localStorage.getItem("hi3d-gpu"); } catch {}
      for (const g of s.gpus) sel.add(new Option(`${g.index} · ${g.name}`, g.index));
      sel.value = saved !== null && s.gpus.some((g) => String(g.index) === saved) ? saved : String(s.gpu_index);
      sel.onchange = () => { try { localStorage.setItem("hi3d-gpu", sel.value); } catch {} refreshStatus(); };
    }
    const g = s.gpus.find((x) => String(x.index) === sel.value) || s.gpus[0];
    const freeGB = g ? g.free_mb / 1024 : 0;
    $("#gpu").textContent = g ? `${freeGB.toFixed(1)} GB free of ${(g.total_mb / 1024).toFixed(0)} · ${g.util}%` : "—";
    $("#gpu").classList.toggle("low", !!g && freeGB < 8);
    $("#gpu").title = g && freeGB < 8 ? "Something else is using this GPU — Hi3D needs roughly 8–14 GB" : "";
    $("#eng-trellis").className = "pill " + (s.trellis_ready ? "ok" : "bad");
    $("#eng-hi3d").className = "pill " + (s.hi3d_ready ? "ok" : "bad");
  } catch { $("#gpu").textContent = "server offline"; }
}

/* ------------------------------------------------------------------ viewer */
const host = $("#viewer");
const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
renderer.setScissorTest(true);
renderer.outputColorSpace = THREE.SRGBColorSpace;
host.prepend(renderer.domElement);

const scene = new THREE.Scene();
scene.add(new THREE.HemisphereLight(0xffffff, 0x30343f, 1.4));
const key = new THREE.DirectionalLight(0xffffff, 2.2); key.position.set(2, 3, 4); scene.add(key);
const rim = new THREE.DirectionalLight(0x9fd8ff, 1.0); rim.position.set(-3, 1, -3); scene.add(rim);
const root = new THREE.Group(); scene.add(root);

const persp = new THREE.PerspectiveCamera(35, 1, 0.01, 100);
persp.position.set(2.2, 1.2, 3.2);
const controls = new OrbitControls(persp, renderer.domElement);
controls.enableDamping = true;
controls.addEventListener("change", () => (dirty = true));
persp.add(new THREE.DirectionalLight(0xffffff, 0.6)); scene.add(persp);

const ortho = (pos, up = [0, 1, 0]) => { const c = new THREE.OrthographicCamera(-1, 1, 1, -1, 0.01, 100); c.position.set(...pos); c.up.set(...up); c.lookAt(0, 0, 0); return c; };
const quadCams = [
  { name: "Front", cam: ortho([0, 0, 6]) }, { name: "Right", cam: ortho([6, 0, 0]) },
  { name: "Back", cam: ortho([0, 0, -6]) }, { name: "Top", cam: ortho([0, 6, 0], [0, 0, -1]) },
];

let layout = "quad", shade = "clay", modelKind = "refined", split = 0.5, dirty = true;
const models = {}; // kind -> THREE.Object3D (normalised)
let loadedOutputs = {}, loadedJob = null;
const clay = new THREE.MeshStandardMaterial({ color: 0xc9ccd3, roughness: 0.62, metalness: 0.0 });
const normalMat = new THREE.MeshNormalMaterial();
const wireMat = new THREE.MeshBasicMaterial({ color: 0x5eead4, wireframe: true, transparent: true, opacity: 0.35 });

function setSeg(id, v) { for (const b of document.querySelectorAll(`#${id} button`)) b.classList.toggle("on", b.dataset.v === v); }
for (const [id, set] of [["layout", (v) => (layout = v)], ["shade", (v) => (shade = v)], ["model", (v) => (modelKind = v)]]) {
  document.querySelector(`#${id}`).addEventListener("click", (e) => {
    const b = e.target.closest("button"); if (!b || b.disabled) return;
    set(b.dataset.v); setSeg(id, b.dataset.v);
    if (id === "model" || (id === "layout" && b.dataset.v === "compare")) ensureModel(id === "model" ? b.dataset.v : "coarse").then(applyView);
    else applyView();
  });
}

function prepare(obj) {
  obj.traverse((o) => {
    if (!o.isMesh) return;
    if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
    // Files with no glTF material get the spec default (fully metallic), which renders
    // near-black in patches. Older Hi3D Studio exports had none; treat them as dielectric.
    const mm = o.material;
    if (mm && mm.isMeshStandardMaterial && mm.metalness === 1 && !mm.metalnessMap && !mm.map) {
      mm.metalness = 0; mm.roughness = 0.8; mm.needsUpdate = true;
    }
    o.userData.orig = o.material;
    o.userData.wire = null;
  });
  const box = new THREE.Box3().setFromObject(obj);
  const sphere = box.getBoundingSphere(new THREE.Sphere());
  const wrap = new THREE.Group();
  obj.position.sub(sphere.center);
  wrap.add(obj);
  wrap.scale.setScalar(1 / (sphere.radius || 1));
  let faces = 0, verts = 0;
  obj.traverse((o) => { if (o.isMesh) { verts += o.geometry.attributes.position.count; faces += (o.geometry.index ? o.geometry.index.count : o.geometry.attributes.position.count) / 3; } });
  wrap.userData.stats = { faces, verts };
  return wrap;
}
function applyShade(obj) {
  obj.traverse((o) => {
    if (!o.isMesh) return;
    o.material = shade === "normals" ? normalMat : shade === "color" ? o.userData.orig : clay;
    if (shade === "wire" && !o.userData.wire) { o.userData.wire = new THREE.Mesh(o.geometry, wireMat); o.add(o.userData.wire); }
    if (o.userData.wire) o.userData.wire.visible = shade === "wire";
  });
}
function applyView() {
  const avail = (k) => !!loadedOutputs[k];
  for (const b of document.querySelectorAll("#model button")) b.disabled = !avail(b.dataset.v);
  root.children.slice().forEach((c) => root.remove(c));
  for (const k of Object.keys(models)) { applyShade(models[k]); }
  const main = models[modelKind];
  if (main) root.add(main);
  const st = main?.userData.stats;
  $("#stats").textContent = st ? `${(st.faces / 1e6).toFixed(2)}M tris · ${(st.verts / 1e6).toFixed(2)}M verts` : "";
  const file = loadedOutputs[modelKind];
  $("#dl").classList.toggle("hidden", !file);
  if (file) { $("#dl").href = `/api/jobs/${loadedJob}/files/${file}`; $("#dl").download = `${loadedJob}_${file}`; }
  controls.enabled = layout !== "quad";
  $("#empty").classList.toggle("hidden", !!main);
  $("#empty").textContent = loadedOutputs[modelKind] && !main ? "Loading mesh…" : "Generate or pick a result from history.";
  const fl = $("#file-list");
  fl.innerHTML = Object.entries(loadedOutputs).map(([k, f]) => `<a href="/api/jobs/${loadedJob}/files/${f}" download="${loadedJob}_${f}">${f}</a>`).join("");
  $("#files").classList.toggle("hidden", !Object.keys(loadedOutputs).length);
  layoutLabels();
  dirty = true;
}

const loader = new GLTFLoader();
const VIEWABLE = ["textured", "refined_coloured", "refined_hybrid", "refined", "refined_aligned", "coarse"];
const pending = {};
function ensureModel(kind) {
  const file = loadedOutputs[kind], job = loadedJob;
  if (!file || models[kind]) return Promise.resolve();
  const key = `${job}/${file}`;
  if (!pending[key]) {
    applyView();
    pending[key] = loader.loadAsync(`/api/jobs/${job}/files/${file}`)
      .then((gltf) => { if (loadedJob === job) models[kind] = prepare(gltf.scene); })
      .catch((e) => console.error(e))
      .finally(() => delete pending[key]);
  }
  return pending[key];
}
async function showJob(j) {
  updateRefineButton(j);
  if (loadedJob !== j.id) {
    for (const k of Object.keys(models)) delete models[k];
    root.rotation.set(0, 0, 0);
    loadedJob = j.id;
    modelKind = VIEWABLE.find((k) => j.outputs[k]) || "refined";
  } else {
    // Hi3D re-run on this job: drop meshes whose files were replaced or removed.
    for (const k of Object.keys(models)) if (!j.outputs[k] || j.outputs[k] !== loadedOutputs[k]) delete models[k];
    if (!loadedOutputs[modelKind] || !j.outputs[modelKind] || modelKind === "coarse") {
      modelKind = VIEWABLE.find((k) => j.outputs[k]) || modelKind;
    }
  }
  loadedOutputs = { ...j.outputs };
  setSeg("model", modelKind);
  await ensureModel(modelKind);
  applyView();
  if (layout === "compare") ensureModel("coarse").then(applyView);
}

/* quad-view drag rotates the object itself */
let drag = null;
renderer.domElement.addEventListener("pointerdown", (e) => {
  if (layout === "compare" && Math.abs(e.offsetX - split * host.clientWidth) < 14) { drag = { split: true }; return; }
  if (layout === "quad") drag = { x: e.clientX, y: e.clientY, ry: root.rotation.y, rx: root.rotation.x };
});
addEventListener("pointermove", (e) => {
  if (!drag) return;
  if (drag.split) { split = Math.min(0.95, Math.max(0.05, (e.clientX - host.getBoundingClientRect().left) / host.clientWidth)); layoutLabels(); dirty = true; return; }
  root.rotation.y = drag.ry + (e.clientX - drag.x) * 0.01;
  root.rotation.x = Math.max(-1.5, Math.min(1.5, drag.rx + (e.clientY - drag.y) * 0.01));
  dirty = true;
});
addEventListener("pointerup", () => (drag = null));
renderer.domElement.addEventListener("dblclick", () => { root.rotation.set(0, 0, 0); controls.reset(); dirty = true; });
addEventListener("keydown", (e) => {
  if (e.target.tagName === "INPUT") return;
  if (e.key === "c" && models.coarse) { const order = ["coarse", "refined", "refined_coloured"].filter((k) => models[k]); modelKind = order[(order.indexOf(modelKind) + 1) % order.length]; setSeg("model", modelKind); applyView(); }
});

function layoutLabels() {
  const L = $("#labels");
  L.innerHTML = "";
  const W = host.clientWidth, H = host.clientHeight;
  if (layout === "quad") {
    quadCams.forEach((q, i) => { const s = document.createElement("span"); s.textContent = q.name; s.style.left = (i % 2) * W / 2 + 8 + "px"; s.style.top = Math.floor(i / 2) * H / 2 + 8 + "px"; L.appendChild(s); });
  } else if (layout === "compare") {
    const a = document.createElement("span"); a.textContent = "Coarse · TRELLIS.2"; a.style.left = "8px"; a.style.top = "8px";
    const b = document.createElement("span"); b.textContent = (modelKind === "coarse" ? "refined" : modelKind.replace("_", " ")) + " · Hi3D"; b.style.right = "8px"; b.style.top = "8px";
    const bar = document.createElement("div"); bar.className = "split"; bar.style.left = split * W - 1 + "px";
    L.append(a, b, bar);
  }
}

function resize() {
  const W = host.clientWidth, H = host.clientHeight;
  renderer.setSize(W, H, false);
  layoutLabels(); dirty = true;
}
new ResizeObserver(resize).observe(host);

function draw(cam, x, y, w, h) {
  renderer.setViewport(x, y, w, h); renderer.setScissor(x, y, w, h);
  if (cam.isOrthographicCamera) { const a = w / h, s = 1.12; cam.left = -s * a; cam.right = s * a; cam.top = s; cam.bottom = -s; cam.updateProjectionMatrix(); }
  else { cam.aspect = w / h; cam.updateProjectionMatrix(); }
  renderer.render(scene, cam);
}
function frame() {
  requestAnimationFrame(frame);
  if (controls.enabled) controls.update();
  if (!dirty) return;
  dirty = false;
  const W = host.clientWidth, H = host.clientHeight;
  renderer.setScissor(0, 0, W, H); renderer.clear();
  if (layout === "quad") {
    const w = Math.floor(W / 2), h = Math.floor(H / 2);
    quadCams.forEach((q, i) => draw(q.cam, (i % 2) * w + (i % 2), (1 - Math.floor(i / 2)) * h + (1 - Math.floor(i / 2)), w - 1, h - 1));
  } else if (layout === "compare" && models.coarse) {
    const right = models[modelKind === "coarse" ? (models.refined_coloured ? "refined_coloured" : "refined") : modelKind];
    const sx = Math.floor(W * split);
    const shown = root.children.slice();
    shown.forEach((c) => root.remove(c));
    persp.aspect = W / H; persp.updateProjectionMatrix();
    renderer.setViewport(0, 0, W, H);
    root.add(models.coarse);
    renderer.setScissor(0, 0, sx, H); renderer.render(scene, persp);
    root.remove(models.coarse);
    if (right) { root.add(right); renderer.setScissor(sx, 0, W - sx, H); renderer.render(scene, persp); root.remove(right); }
    shown.forEach((c) => root.add(c));
  } else {
    draw(persp, 0, 0, W, H);
  }
}
/* ------------------------------------------------------------------ boot */
syncForm();
renderThumbs();
refreshStatus(); setInterval(refreshStatus, 3000);
refreshJobs(); setInterval(() => { if (!activeJob) refreshJobs(); }, 5000);
resize();
frame();
