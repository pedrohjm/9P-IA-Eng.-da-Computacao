const IMG_SIZE = 28;
const INPUT_SIZE = IMG_SIZE * IMG_SIZE;
const NUM_CLASSES = 62;
const CLASS_CHARS = [
  ..."0123456789",
  ..."ABCDEFGHIJKLMNOPQRSTUVWXYZ",
  ..."abcdefghijklmnopqrstuvwxyz"
];

const N_FILTERS = 4;
const KERNEL = 3;
const CONV_W = IMG_SIZE - KERNEL + 1;
const CONV_H = IMG_SIZE - KERNEL + 1;
const POOL = 2;
const POOL_W = Math.floor(CONV_W / POOL);
const POOL_H = Math.floor(CONV_H / POOL);
const FC_IN = N_FILTERS * POOL_W * POOL_H;
const MAX_DATASET_MB = 220;

let model = null;
let canvas, ctx, previewCanvas, previewCtx;

function randFloat(a, b) {
  return a + (b - a) * Math.random();
}

function classHexByIndex(idx) {
  if (idx < 10) return (48 + idx).toString(16);
  if (idx < 36) return (65 + idx - 10).toString(16);
  return (97 + idx - 36).toString(16);
}

function switchTab(tabName) {
  const isTrain = tabName === 'train-tab';
  document.querySelectorAll('.tab-content').forEach(el => el.classList.remove('active'));
  document.querySelectorAll('.tab').forEach(el => el.classList.remove('active'));
  document.getElementById(tabName).classList.add('active');
  document.getElementById(isTrain ? 'tabTrainBtn' : 'tabPredBtn').classList.add('active');
}

function addLog(msg) {
  const logBox = document.getElementById('logBox');
  const time = new Date().toLocaleTimeString();
  logBox.textContent += `[${time}] ${msg}\n`;
  logBox.scrollTop = logBox.scrollHeight;
}

function shuffle(arr) {
  for (let i = arr.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    const t = arr[i];
    arr[i] = arr[j];
    arr[j] = t;
  }
}

function makeModel() {
  const convW = new Float32Array(N_FILTERS * KERNEL * KERNEL);
  const convB = new Float32Array(N_FILTERS);
  const fcW = new Float32Array(NUM_CLASSES * FC_IN);
  const fcB = new Float32Array(NUM_CLASSES);

  for (let i = 0; i < convW.length; i++) convW[i] = randFloat(-0.12, 0.12);
  for (let i = 0; i < fcW.length; i++) fcW[i] = randFloat(-0.06, 0.06);

  return { convW, convB, fcW, fcB };
}

function softmax(logits, out) {
  let maxV = logits[0];
  for (let i = 1; i < NUM_CLASSES; i++) {
    if (logits[i] > maxV) maxV = logits[i];
  }
  let sum = 0;
  for (let i = 0; i < NUM_CLASSES; i++) {
    out[i] = Math.exp(logits[i] - maxV);
    sum += out[i];
  }
  if (sum < 1e-12) sum = 1e-12;
  for (let i = 0; i < NUM_CLASSES; i++) out[i] /= sum;
}

function forward(x, cache) {
  const convZ = cache.convZ;
  const convA = cache.convA;
  const pool = cache.pool;
  const poolIdx = cache.poolIdx;
  const flat = cache.flat;
  const logits = cache.logits;
  const probs = cache.probs;

  let flatPos = 0;

  for (let f = 0; f < N_FILTERS; f++) {
    const convFilterBase = f * KERNEL * KERNEL;
    for (let oy = 0; oy < CONV_H; oy++) {
      for (let ox = 0; ox < CONV_W; ox++) {
        let z = model.convB[f];
        for (let ky = 0; ky < KERNEL; ky++) {
          for (let kx = 0; kx < KERNEL; kx++) {
            const ix = ox + kx;
            const iy = oy + ky;
            z += x[iy * IMG_SIZE + ix] * model.convW[convFilterBase + ky * KERNEL + kx];
          }
        }
        const convIdx = f * CONV_H * CONV_W + oy * CONV_W + ox;
        convZ[convIdx] = z;
        convA[convIdx] = z > 0 ? z : 0;
      }
    }
  }

  for (let f = 0; f < N_FILTERS; f++) {
    for (let oy = 0; oy < POOL_H; oy++) {
      for (let ox = 0; ox < POOL_W; ox++) {
        const baseY = oy * POOL;
        const baseX = ox * POOL;
        let best = -1e30;
        let bestK = 0;
        let k = 0;

        for (let py = 0; py < POOL; py++) {
          for (let px = 0; px < POOL; px++) {
            const y = baseY + py;
            const xPos = baseX + px;
            const idx = f * CONV_H * CONV_W + y * CONV_W + xPos;
            const v = convA[idx];
            if (v > best) {
              best = v;
              bestK = k;
            }
            k++;
          }
        }

        const poolPos = f * POOL_H * POOL_W + oy * POOL_W + ox;
        pool[poolPos] = best;
        poolIdx[poolPos] = bestK;
        flat[flatPos++] = best;
      }
    }
  }

  for (let c = 0; c < NUM_CLASSES; c++) {
    let z = model.fcB[c];
    const row = c * FC_IN;
    for (let i = 0; i < FC_IN; i++) z += model.fcW[row + i] * flat[i];
    logits[c] = z;
  }

  softmax(logits, probs);
}

function createCache() {
  return {
    convZ: new Float32Array(N_FILTERS * CONV_H * CONV_W),
    convA: new Float32Array(N_FILTERS * CONV_H * CONV_W),
    pool: new Float32Array(N_FILTERS * POOL_H * POOL_W),
    poolIdx: new Uint8Array(N_FILTERS * POOL_H * POOL_W),
    flat: new Float32Array(FC_IN),
    logits: new Float32Array(NUM_CLASSES),
    probs: new Float32Array(NUM_CLASSES)
  };
}

function trainSample(x, y, lr, cache, grads) {
  forward(x, cache);

  const dLogits = grads.dLogits;
  const dFlat = grads.dFlat;
  const dPool = grads.dPool;
  const dConvA = grads.dConvA;
  const dConvZ = grads.dConvZ;

  for (let c = 0; c < NUM_CLASSES; c++) dLogits[c] = cache.probs[c];
  dLogits[y] -= 1;

  dFlat.fill(0);

  for (let c = 0; c < NUM_CLASSES; c++) {
    const g = dLogits[c];
    const row = c * FC_IN;
    for (let i = 0; i < FC_IN; i++) {
      const wOld = model.fcW[row + i];
      dFlat[i] += wOld * g;
      model.fcW[row + i] -= lr * g * cache.flat[i];
    }
    model.fcB[c] -= lr * g;
  }

  let p = 0;
  for (let f = 0; f < N_FILTERS; f++) {
    for (let oy = 0; oy < POOL_H; oy++) {
      for (let ox = 0; ox < POOL_W; ox++) {
        dPool[f * POOL_H * POOL_W + oy * POOL_W + ox] = dFlat[p++];
      }
    }
  }

  dConvA.fill(0);

  for (let f = 0; f < N_FILTERS; f++) {
    for (let oy = 0; oy < POOL_H; oy++) {
      for (let ox = 0; ox < POOL_W; ox++) {
        const baseY = oy * POOL;
        const baseX = ox * POOL;
        const poolPos = f * POOL_H * POOL_W + oy * POOL_W + ox;
        const pi = cache.poolIdx[poolPos];
        const py = Math.floor(pi / POOL);
        const px = pi % POOL;
        const convPos = f * CONV_H * CONV_W + (baseY + py) * CONV_W + (baseX + px);
        dConvA[convPos] += dPool[poolPos];
      }
    }
  }

  for (let i = 0; i < dConvZ.length; i++) {
    dConvZ[i] = cache.convZ[i] > 0 ? dConvA[i] : 0;
  }

  for (let f = 0; f < N_FILTERS; f++) {
    const filterBase = f * KERNEL * KERNEL;
    let gradB = 0;

    for (let ky = 0; ky < KERNEL; ky++) {
      for (let kx = 0; kx < KERNEL; kx++) {
        let gradW = 0;
        for (let oy = 0; oy < CONV_H; oy++) {
          for (let ox = 0; ox < CONV_W; ox++) {
            const ix = ox + kx;
            const iy = oy + ky;
            const pos = f * CONV_H * CONV_W + oy * CONV_W + ox;
            gradW += dConvZ[pos] * x[iy * IMG_SIZE + ix];
          }
        }
        model.convW[filterBase + ky * KERNEL + kx] -= lr * gradW;
      }
    }

    for (let oy = 0; oy < CONV_H; oy++) {
      for (let ox = 0; ox < CONV_W; ox++) {
        gradB += dConvZ[f * CONV_H * CONV_W + oy * CONV_W + ox];
      }
    }
    model.convB[f] -= lr * gradB;
  }

  return -Math.log(cache.probs[y] + 1e-9);
}

function predictVector(vec, cache) {
  forward(vec, cache);
  let best = 0;
  for (let i = 1; i < NUM_CLASSES; i++) {
    if (cache.probs[i] > cache.probs[best]) best = i;
  }
  return { idx: best, probs: cache.probs };
}

async function fileToVector(file) {
  const bmp = await createImageBitmap(file);
  const c = document.createElement("canvas");
  c.width = IMG_SIZE;
  c.height = IMG_SIZE;
  const cctx = c.getContext("2d");
  cctx.fillStyle = "white";
  cctx.fillRect(0, 0, IMG_SIZE, IMG_SIZE);
  cctx.drawImage(bmp, 0, 0, IMG_SIZE, IMG_SIZE);
  const data = cctx.getImageData(0, 0, IMG_SIZE, IMG_SIZE).data;
  const out = new Float32Array(INPUT_SIZE);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    out[p] = Math.min(1, Math.max(0, 1 - gray / 255));
  }
  bmp.close();
  return out;
}

function estimateDatasetMB(totalSamples) {
  return (totalSamples * INPUT_SIZE * 4) / (1024 * 1024);
}

async function scanClassCounts(rootHandle) {
  const classCounts = new Int32Array(NUM_CLASSES);

  for (let cls = 0; cls < NUM_CLASSES; cls++) {
    const hex = classHexByIndex(cls);
    let classDir;
    try {
      classDir = await rootHandle.getDirectoryHandle(hex, { create: false });
    } catch {
      continue;
    }

    let trainDir;
    try {
      trainDir = await classDir.getDirectoryHandle(`train_${hex}`, { create: false });
    } catch {
      continue;
    }

    let count = 0;
    for await (const [, handle] of trainDir.entries()) {
      if (handle.kind === "file" && handle.name.toLowerCase().endsWith(".png")) count++;
    }
    classCounts[cls] = count;
  }

  return classCounts;
}

function planSampling(classCounts, requestedPerClass) {
  const classesWithData = classCounts.reduce((acc, c) => acc + (c > 0 ? 1 : 0), 0);
  if (!classesWithData) throw new Error("Nenhuma imagem encontrada.");

  const maxSamplesByBudget = Math.max(1, Math.floor((MAX_DATASET_MB * 1024 * 1024) / (INPUT_SIZE * 4)));
  const budgetPerClass = Math.max(1, Math.floor(maxSamplesByBudget / classesWithData));
  const desiredPerClass = Number.isFinite(requestedPerClass)
    ? Math.max(1, requestedPerClass)
    : Number.POSITIVE_INFINITY;
  const effectivePerClass = Number.isFinite(desiredPerClass)
    ? Math.min(desiredPerClass, budgetPerClass)
    : budgetPerClass;

  let plannedTotal = 0;
  for (let cls = 0; cls < NUM_CLASSES; cls++) {
    plannedTotal += Math.min(classCounts[cls], effectivePerClass);
  }

  return {
    classesWithData,
    budgetPerClass,
    effectivePerClass,
    plannedTotal,
    estimatedMB: estimateDatasetMB(plannedTotal),
    cappedByMemory: effectivePerClass < desiredPerClass
  };
}

async function buildDatasetFromByClass(rootHandle, classCounts, effectivePerClass) {
  const plannedTotal = classCounts.reduce((acc, c) => acc + Math.min(c, effectivePerClass), 0);
  const x = new Float32Array(plannedTotal * INPUT_SIZE);
  const y = new Int32Array(plannedTotal);
  let writeIdx = 0;

  for (let cls = 0; cls < NUM_CLASSES; cls++) {
    const hex = classHexByIndex(cls);
    let classDir;
    try {
      classDir = await rootHandle.getDirectoryHandle(hex, { create: false });
    } catch {
      continue;
    }

    let trainDir;
    try {
      trainDir = await classDir.getDirectoryHandle(`train_${hex}`, { create: false });
    } catch {
      continue;
    }

    const files = [];
    for await (const [, handle] of trainDir.entries()) {
      if (handle.kind === "file" && handle.name.toLowerCase().endsWith(".png")) files.push(handle);
    }
    files.sort((a, b) => a.name.localeCompare(b.name));

    const limit = Math.min(files.length, effectivePerClass);
    let localCount = 0;

    for (let i = 0; i < limit; i++) {
      const fh = files[i];
      const f = await fh.getFile();
      const vec = await fileToVector(f);
      x.set(vec, writeIdx * INPUT_SIZE);
      y[writeIdx] = cls;
      writeIdx++;
      localCount++;
    }

    addLog(`Classe ${CLASS_CHARS[cls]} (${hex}) carregada: ${localCount} amostras`);
    await new Promise(r => setTimeout(r, 8));
  }

  if (!writeIdx) throw new Error("Nenhuma imagem encontrada.");

  if (writeIdx === plannedTotal) {
    return { x, y, total: writeIdx };
  }

  return {
    x: x.slice(0, writeIdx * INPUT_SIZE),
    y: y.slice(0, writeIdx),
    total: writeIdx
  };
}

function getSample(xFlat, idx) {
  const base = idx * INPUT_SIZE;
  return xFlat.subarray(base, base + INPUT_SIZE);
}

function calcAccuracy(indices, xFlat, yArr, cache) {
  if (!indices.length) return 0;
  let correct = 0;
  for (const idx of indices) {
    const p = predictVector(getSample(xFlat, idx), cache).idx;
    if (p === yArr[idx]) correct++;
  }
  return correct / indices.length;
}

async function startTraining() {
  const btn = document.getElementById('trainBtn');
  btn.disabled = true;
  document.getElementById('logBox').textContent = "";
  document.getElementById('status').innerHTML = '<span class="status-muted">Selecionando pasta...</span>';

  try {
    const root = await window.showDirectoryPicker({ mode: "read" });
    const maxPerClassRaw = parseInt(document.getElementById('maxPerClass').value, 10);
    const maxPerClass = Number.isFinite(maxPerClassRaw) && maxPerClassRaw > 0
      ? maxPerClassRaw
      : Number.POSITIVE_INFINITY;
    const epochs = Math.max(1, parseInt(document.getElementById('epochs').value) || 10);
    const batchSize = Math.max(8, parseInt(document.getElementById('batchSize').value) || 32);
    const lr = 0.006 * Math.min(1.8, Math.max(0.6, 32 / batchSize));

    addLog("Escaneando dataset para planejar uso de memoria...");
    const classCounts = await scanClassCounts(root);
    const samplingPlan = planSampling(classCounts, maxPerClass);

    if (!Number.isFinite(maxPerClass)) {
      addLog(`Amostras por classe: automatico (ate ${samplingPlan.budgetPerClass}, limite de ~${MAX_DATASET_MB}MB).`);
    } else if (samplingPlan.cappedByMemory) {
      addLog(`Amostras por classe solicitadas: ${maxPerClass}, ajustado para ${samplingPlan.effectivePerClass} por memoria.`);
    } else {
      addLog(`Amostras por classe: limite definido em ${samplingPlan.effectivePerClass}.`);
    }

    addLog(`Planejado: ${samplingPlan.plannedTotal} amostras (~${samplingPlan.estimatedMB.toFixed(1)} MB em tensores).`);
    addLog("Carregando imagens...");
    const dataPack = await buildDatasetFromByClass(root, classCounts, samplingPlan.effectivePerClass);
    document.getElementById('datasetInfo').style.display = 'block';
    document.getElementById('totalSamples').textContent = dataPack.total;
    addLog(`Total de amostras: ${dataPack.total}`);

    model = makeModel();
    const cache = createCache();
    const valCache = createCache();
    const grads = {
      dLogits: new Float32Array(NUM_CLASSES),
      dFlat: new Float32Array(FC_IN),
      dPool: new Float32Array(N_FILTERS * POOL_H * POOL_W),
      dConvA: new Float32Array(N_FILTERS * CONV_H * CONV_W),
      dConvZ: new Float32Array(N_FILTERS * CONV_H * CONV_W)
    };

    const all = Array.from({ length: dataPack.total }, (_, i) => i);
    shuffle(all);
    const valCount = Math.max(1, Math.floor(dataPack.total * 0.15));
    const valIdx = all.slice(0, valCount);
    const trainIdx = all.slice(valCount);

    addLog("Iniciando treino (forward/backprop)...");
    document.getElementById('status').innerHTML = '<span class="status-ok">[OK] Treinando...</span>';

    for (let ep = 1; ep <= epochs; ep++) {
      shuffle(trainIdx);
      let loss = 0;
      let acc = 0;

      for (let i = 0; i < trainIdx.length; i++) {
        const s = trainIdx[i];
        const x = getSample(dataPack.x, s);
        const y = dataPack.y[s];
        loss += trainSample(x, y, lr, cache, grads);
        const p = predictVector(x, cache).idx;
        if (p === y) acc++;
        if (i % 24 === 0) await new Promise(r => setTimeout(r, 0));
      }

      loss /= trainIdx.length;
      const trainAcc = acc / trainIdx.length;
      const valAcc = calcAccuracy(valIdx, dataPack.x, dataPack.y, valCache);

      document.getElementById('cycles').textContent = ep;
      document.getElementById('loss').textContent = loss.toFixed(4);
      document.getElementById('acc').textContent = (trainAcc * 100).toFixed(2) + '%';
      document.getElementById('valAcc').textContent = (valAcc * 100).toFixed(2) + '%';
      addLog(`Ciclo ${ep}: loss=${loss.toFixed(4)} | acc=${(trainAcc * 100).toFixed(2)}% | val=${(valAcc * 100).toFixed(2)}%`);
    }

    addLog("Treino finalizado!");
    document.getElementById('status').innerHTML = '<span class="status-ok">[OK] IA treinada com CNN manual em JavaScript.</span>';
  } catch (err) {
    addLog(`ERRO: ${err.message}`);
    document.getElementById('status').innerHTML = `<span class="status-warn">[!] ${err.message}</span>`;
  } finally {
    btn.disabled = false;
  }
}

function clearCanvas() {
  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  document.getElementById('predLetter').textContent = '?';
  document.getElementById('predConf').textContent = '0.0';
  document.getElementById('topPredictions').innerHTML = '';
  previewCtx.fillStyle = "#ffffff";
  previewCtx.fillRect(0, 0, 28, 28);
}

function getVector28() {
  const srcW = canvas.width;
  const srcH = canvas.height;
  const srcData = ctx.getImageData(0, 0, srcW, srcH).data;

  let minX = srcW;
  let minY = srcH;
  let maxX = -1;
  let maxY = -1;

  for (let y = 0; y < srcH; y++) {
    for (let x = 0; x < srcW; x++) {
      const i = (y * srcW + x) * 4;
      const gray = 0.299 * srcData[i] + 0.587 * srcData[i + 1] + 0.114 * srcData[i + 2];
      const ink = 1 - gray / 255;
      if (ink > 0.08) {
        if (x < minX) minX = x;
        if (y < minY) minY = y;
        if (x > maxX) maxX = x;
        if (y > maxY) maxY = y;
      }
    }
  }

  const tmp = document.createElement("canvas");
  tmp.width = 28;
  tmp.height = 28;
  const tctx = tmp.getContext("2d");
  tctx.fillStyle = "#ffffff";
  tctx.fillRect(0, 0, 28, 28);

  if (maxX >= minX && maxY >= minY) {
    const margin = 16;
    const sx = Math.max(0, minX - margin);
    const sy = Math.max(0, minY - margin);
    const ex = Math.min(srcW - 1, maxX + margin);
    const ey = Math.min(srcH - 1, maxY + margin);
    const sw = ex - sx + 1;
    const sh = ey - sy + 1;

    const targetSize = 20;
    const scale = targetSize / Math.max(sw, sh);
    const dw = sw * scale;
    const dh = sh * scale;
    const dx = (28 - dw) / 2;
    const dy = (28 - dh) / 2;

    tctx.imageSmoothingEnabled = true;
    tctx.drawImage(canvas, sx, sy, sw, sh, dx, dy, dw, dh);
  }

  previewCtx.fillStyle = "#ffffff";
  previewCtx.fillRect(0, 0, 28, 28);
  previewCtx.drawImage(tmp, 0, 0);

  const data = tctx.getImageData(0, 0, 28, 28).data;
  const out = new Float32Array(INPUT_SIZE);
  for (let i = 0, p = 0; i < data.length; i += 4, p++) {
    const gray = 0.299 * data[i] + 0.587 * data[i + 1] + 0.114 * data[i + 2];
    const norm = Math.min(1, Math.max(0, 1 - gray / 255));
    out[p] = norm < 0.1 ? 0 : Math.pow(norm, 0.9);
  }
  return out;
}

function renderPredictionBars(probs, animate) {
  const indexed = Array.from(probs, (v, i) => ({ i, v }))
    .sort((a, b) => b.v - a.v)
    .slice(0, 5);

  document.getElementById('predLetter').textContent = CLASS_CHARS[indexed[0].i];
  document.getElementById('predConf').textContent = (indexed[0].v * 100).toFixed(1);

  document.getElementById('topPredictions').innerHTML = indexed.map(p => `
    <div class="bar">
      <span>${CLASS_CHARS[p.i]}</span>
      <div class="bar-track">
        <div class="bar-fill" style="width: ${animate ? '0%' : (p.v * 100).toFixed(2) + '%'}"></div>
      </div>
      <span>${animate ? '0.0%' : (p.v * 100).toFixed(1) + '%'}</span>
    </div>
  `).join('');

  return indexed;
}

function predict() {
  if (!model) {
    alert("Treine a IA primeiro na aba Treinamento.");
    return;
  }
  const vec = getVector28();
  const cache = createCache();
  const out = predictVector(vec, cache);
  renderPredictionBars(out.probs, false);
}

async function predictAnimate() {
  if (!model) {
    alert("Treine a IA primeiro.");
    return;
  }
  const vec = getVector28();
  const cache = createCache();
  const out = predictVector(vec, cache);
  const top = renderPredictionBars(out.probs, true);

  const bars = document.querySelectorAll('.bar-fill');
  const steps = 14;
  for (let s = 1; s <= steps; s++) {
    const frac = s / steps;
    bars.forEach((bar, i) => {
      bar.style.width = (top[i].v * frac * 100).toFixed(2) + '%';
      bar.parentElement.parentElement.querySelector('span:last-child').textContent =
        (top[i].v * frac * 100).toFixed(1) + '%';
    });
    await new Promise(r => setTimeout(r, 28));
  }
}

document.addEventListener('DOMContentLoaded', function() {
  document.getElementById('tabTrainBtn').addEventListener('click', () => switchTab('train-tab'));
  document.getElementById('tabPredBtn').addEventListener('click', () => switchTab('pred-tab'));
  document.getElementById('trainBtn').addEventListener('click', startTraining);
  document.getElementById('clearBtn').addEventListener('click', clearCanvas);
  document.getElementById('predictBtn').addEventListener('click', predict);

  canvas = document.getElementById('drawCanvas');
  ctx = canvas.getContext('2d');
  previewCanvas = document.getElementById('preview28');
  previewCtx = previewCanvas.getContext('2d');

  ctx.fillStyle = "#ffffff";
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineCap = "round";
  ctx.lineJoin = "round";
  ctx.strokeStyle = "#000000";
  ctx.lineWidth = 18;

  previewCtx.fillStyle = "#fff";
  previewCtx.fillRect(0, 0, 28, 28);

  let drawing = false;
  const getPos = (evt) => {
    const rect = canvas.getBoundingClientRect();
    if (evt.touches && evt.touches.length > 0) {
      return {
        x: (evt.touches[0].clientX - rect.left) * (canvas.width / rect.width),
        y: (evt.touches[0].clientY - rect.top) * (canvas.height / rect.height)
      };
    }
    return {
      x: (evt.clientX - rect.left) * (canvas.width / rect.width),
      y: (evt.clientY - rect.top) * (canvas.height / rect.height)
    };
  };

  canvas.addEventListener('mousedown', (e) => {
    drawing = true;
    const p = getPos(e);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
  });

  canvas.addEventListener('mousemove', (e) => {
    if (!drawing) return;
    const p = getPos(e);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  });

  canvas.addEventListener('mouseup', () => { drawing = false; });
  canvas.addEventListener('mouseleave', () => { drawing = false; });

  canvas.addEventListener('touchstart', (e) => {
    e.preventDefault();
    drawing = true;
    const p = getPos(e);
    ctx.beginPath();
    ctx.moveTo(p.x, p.y);
  });

  canvas.addEventListener('touchmove', (e) => {
    e.preventDefault();
    if (!drawing) return;
    const p = getPos(e);
    ctx.lineTo(p.x, p.y);
    ctx.stroke();
  });

  canvas.addEventListener('touchend', () => { drawing = false; });
});
