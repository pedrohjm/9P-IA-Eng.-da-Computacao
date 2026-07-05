// ── Trabalho 05 — MLP Reconhecimento de Letras (TensorFlow.js) ──

// Elementos
const canvas = document.getElementById('draw');
const ctx = canvas.getContext('2d');
const clearBtn = document.getElementById('clear');
const predictBtn = document.getElementById('predict');
const csvFile = document.getElementById('csvFile');
const loadCsvBtn = document.getElementById('loadCsv');
const trainBtn = document.getElementById('train');
const logEl = document.getElementById('log');
const epochsInput = document.getElementById('epochs');
const batchSizeInput = document.getElementById('batchSize');
const saveModelBtn = document.getElementById('saveModel');
const loadModelBtn = document.getElementById('loadModel');
const datasetInfoEl = document.getElementById('datasetInfo');

// Estado
let model = null;
let dataset = { xs: null, ys: null, numClasses: 0, classLabels: null };
let firstLog = true;

// Converte o rótulo numérico (0-9, 10-35, 36-61) no caractere correspondente,
// seguindo a mesma convenção usada em Trabalho 06/tools/prepare_dataset.ps1
// (classes ordenadas por código ASCII: dígitos, depois A-Z, depois a-z).
function labelToChar(label) {
  if (label >= 0 && label <= 9) return String(label);
  if (label >= 10 && label <= 35) return String.fromCharCode('A'.charCodeAt(0) + (label - 10));
  if (label >= 36 && label <= 61) return String.fromCharCode('a'.charCodeAt(0) + (label - 36));
  return `#${label}`;
}

// Inicializa canvas (fundo branco)
function initCanvas() {
  ctx.fillStyle = 'white';
  ctx.fillRect(0, 0, canvas.width, canvas.height);
  ctx.lineWidth = 18;
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'black';
}
initCanvas();

function log(msg) {
  const t = new Date().toLocaleTimeString();
  if (firstLog) { logEl.innerText = ''; firstLog = false; }
  logEl.innerText = `[${t}] ${msg}\n` + logEl.innerText;
}

// Eventos de desenho
let drawing = false;
function pos(e) {
  const rect = canvas.getBoundingClientRect();
  const x = (e.touches ? e.touches[0].clientX : e.clientX) - rect.left;
  const y = (e.touches ? e.touches[0].clientY : e.clientY) - rect.top;
  return { x, y };
}
canvas.addEventListener('pointerdown', e => { drawing = true; const p = pos(e); ctx.beginPath(); ctx.moveTo(p.x, p.y); });
canvas.addEventListener('pointermove', e => { if (!drawing) return; const p = pos(e); ctx.lineTo(p.x, p.y); ctx.stroke(); });
canvas.addEventListener('pointerup', () => drawing = false);
canvas.addEventListener('pointerleave', () => drawing = false);

clearBtn.addEventListener('click', () => { initCanvas(); log('Canvas limpo'); });

// Preprocessa canvas: redimensiona para 28x28 e retorna array normalizado (1 - grayscale/255)
function canvasTo28() {
  const off = document.createElement('canvas');
  off.width = 28; off.height = 28;
  const octx = off.getContext('2d');
  octx.fillStyle = 'white'; octx.fillRect(0, 0, 28, 28);
  octx.drawImage(canvas, 0, 0, 28, 28);
  const id = octx.getImageData(0, 0, 28, 28).data;
  const arr = new Float32Array(28 * 28);
  for (let i = 0, j = 0; i < id.length; i += 4, j++) {
    const r = id[i], g = id[i + 1], b = id[i + 2];
    const gray = (r + g + b) / 3;
    arr[j] = 1 - (gray / 255); // branco->0, traço->1
  }
  return arr;
}

// Predição
predictBtn.addEventListener('click', async () => {
  if (!model) { log('Modelo não carregado. Treine ou carregue um modelo.'); return; }
  const input = canvasTo28();
  const x = tf.tensor2d(input, [1, 784]);
  const pred = model.predict(x);
  const p = await pred.array();
  const idx = p[0].indexOf(Math.max(...p[0]));
  const label = dataset.classLabels ? dataset.classLabels[idx] : idx;
  const char = labelToChar(label);
  log(`Predição: "${char}" (classe ${label}, prob=${(p[0][idx] * 100).toFixed(2)}%)`);
  x.dispose(); pred.dispose();
});

// CSV -> dataset (espera: label, pixel0, pixel1, ...)
function buildDatasetFromRows(rows) {
  const labels = [];
  const images = [];
  let maxPixel = 0;
  for (const r of rows) {
    if (r.length < 785) continue;
    const label = Number(r[0]);
    if (Number.isNaN(label)) continue; // ignora a linha de cabeçalho (label,pixel0,...)
    labels.push(label);
    const pix = new Float32Array(784);
    for (let i = 0; i < 784; i++) {
      const v = Number(r[i + 1]);
      if (v > maxPixel) maxPixel = v;
      pix[i] = v;
    }
    images.push(pix);
  }
  if (labels.length === 0) return null;


  if (maxPixel > 1) {
    for (const pix of images) for (let i = 0; i < pix.length; i++) pix[i] /= 255.0;
  }
  const uniqueLabels = Array.from(new Set(labels)).sort((a, b) => a - b);
  const numClasses = uniqueLabels.length;
  // map labels to 0..numClasses-1
  const labelMap = new Map(uniqueLabels.map((v, i) => [v, i]));
  const ys = Int32Array.from(labels.map(l => labelMap.get(l)));
  const xs = new Float32Array(images.length * 784);
  for (let i = 0; i < images.length; i++) xs.set(images[i], i * 784);
  return {
    xs: tf.tensor2d(xs, [images.length, 784]),
    ys: tf.tensor1d(ys, 'int32'),
    numClasses,
    classLabels: uniqueLabels, // classLabels[classeComprimida] = rótulo original do CSV
  };
}

loadCsvBtn.addEventListener('click', () => {
  const file = csvFile.files[0];
  if (!file) { alert('Selecione um CSV'); return; }
  log('Carregando CSV...');
  Papa.parse(file, {
    worker: true, skipEmptyLines: true, complete: (res) => {
      const d = buildDatasetFromRows(res.data);
      if (!d) { log('CSV inválido ou vazio'); return; }
      if (dataset.xs) dataset.xs.dispose();
      if (dataset.ys) dataset.ys.dispose();
      dataset = d;
      log(`Dataset carregado: ${dataset.xs.shape[0]} amostras, ${dataset.numClasses} classes`);
      datasetInfoEl.textContent = `${dataset.xs.shape[0]} amostras · ${dataset.numClasses} classes`;
      // cria modelo automaticamente com numClasses
      model = createModel(dataset.numClasses);
      log('Modelo inicializado');
    }, error: (err) => { log('Erro ao ler CSV: ' + err.message); }
  });
});

function createModel(numClasses) {
  const m = tf.sequential();
  m.add(tf.layers.dense({ inputShape: [784], units: 128, activation: 'relu' }));
  m.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  m.add(tf.layers.dense({ units: numClasses, activation: 'softmax' }));
  m.compile({ optimizer: tf.train.adam(0.001), loss: 'sparseCategoricalCrossentropy', metrics: ['accuracy'] });
  return m;
}

trainBtn.addEventListener('click', async () => {
  if (!dataset.xs) { alert('Carregue um CSV primeiro'); return; }
  const epochs = Number(epochsInput.value) || 10;
  const batchSize = Number(batchSizeInput.value) || 32;
  if (!model) model = createModel(dataset.numClasses);
  log(`Iniciando treino: epochs=${epochs}, batchSize=${batchSize}`);
  await model.fit(dataset.xs, dataset.ys, {
    epochs, batchSize,
    callbacks: {
      onEpochEnd: (epoch, logs) => { log(`Epoch ${epoch + 1}: loss=${logs.loss.toFixed(4)} acc=${(logs.acc || logs.acc).toFixed ? (logs.acc * 100).toFixed(2) + '%' : logs.acc}`); }
    }
  });
  log('Treino concluído');
});

// Save / load
saveModelBtn.addEventListener('click', async () => {
  if (!model) { alert('Modelo não existe'); return; }
  await model.save('downloads://modelo-mlp-letras');
  log('Modelo salvo (download)');
});
loadModelBtn.addEventListener('click', async () => {
  try {
    const url = prompt('Cole a URL do modelo (local ou remote) ou deixe em branco para cancelar');
    if (!url) return;
    model = await tf.loadLayersModel(url);
    log('Modelo carregado de: ' + url);
  } catch (err) { log('Erro ao carregar modelo: ' + err.message); }
});

// Aviso: liberar memória ao fechar a página
window.addEventListener('beforeunload', () => {
  if (dataset.xs) dataset.xs.dispose();
  if (dataset.ys) dataset.ys.dispose();
  if (model) model.dispose();
});
