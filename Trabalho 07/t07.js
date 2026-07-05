// ── Trabalho 07 — MLP para previsão de cotações (VIVA3) ──

const MAX_EPOCAS = 20000;
const TAXA_APRENDIZADO = 0.03;
const ERRO_ALVO = 0.0005;

const DATAS = [
  '2025-01-02', '2025-01-06', '2025-01-10', '2025-01-15', '2025-01-20', '2025-01-24', '2025-01-29',
  '2025-02-03', '2025-02-07', '2025-02-12', '2025-02-17', '2025-02-21', '2025-02-26',
  '2025-03-05', '2025-03-10', '2025-03-14', '2025-03-19', '2025-03-24', '2025-03-28',
  '2025-04-02', '2025-04-07', '2025-04-11', '2025-04-16', '2025-04-22', '2025-04-28',
  '2025-05-05', '2025-05-09', '2025-05-14', '2025-05-19', '2025-05-23', '2025-05-28',
  '2025-06-03', '2025-06-06', '2025-06-11', '2025-06-16', '2025-06-20', '2025-06-25', '2025-06-30',
  '2025-07-01', '2025-07-02', '2025-07-03', '2025-07-04', '2025-07-07', '2025-07-08', '2025-07-09',
];

const FECHAMENTOS = [
  23.10, 23.30, 23.05, 23.60, 23.90, 24.10, 23.80,
  24.20, 24.55, 24.40, 24.95, 25.10, 24.85,
  25.20, 25.45, 25.15, 25.70, 25.95, 25.60,
  25.85, 26.10, 25.90, 26.35, 26.55, 26.30,
  26.70, 26.45, 26.95, 27.20, 27.05, 27.30,
  27.10, 27.40, 27.15, 27.55, 27.35, 27.80, 27.60,
  27.75, 27.50, 27.95, 28.10, 27.85, 28.20, 28.05,
];

// ── Estado da aplicação ──────────────────────────────────────
const app = {
  dados: [], total: 0, carregado: false,
  idxTrainIni: -1, idxTrainFim: -1, idxValIni: -1, idxValFim: -1,
  minTrain: 0, maxTrain: 0,
  rede: { hidden: 0, w1: [], b1: [], w2: [], b2: 0, treinada: false },
};

let running = false;

// ── Utilidades de data ───────────────────────────────────────
function parseData(s) {
  const m = /^(\d+)-(\d+)-(\d+)$/.exec(s);
  if (!m) return null;
  return { y: +m[1], m: +m[2], d: +m[3] };
}

function dataCmp(a, b) {
  if (a.y !== b.y) return a.y - b.y;
  if (a.m !== b.m) return a.m - b.m;
  return a.d - b.d;
}

function normalizar(x, minv, maxv) {
  if (Math.abs(maxv - minv) < 1e-12) return 0.0;
  return 2.0 * (x - minv) / (maxv - minv) - 1.0;
}

function desnormalizar(x, minv, maxv) {
  return ((x + 1.0) * 0.5) * (maxv - minv) + minv;
}

function tanhDeriv(y) { return 1.0 - y * y; }

// ── MLP ───────────────────────────────────────────────────────
function mlpInicializar(hidden) {
  app.rede.hidden = hidden;
  app.rede.treinada = false;
  app.rede.w1 = Array.from({ length: hidden }, () => Math.random() * 2.0 - 1.0);
  app.rede.b1 = Array.from({ length: hidden }, () => Math.random() * 2.0 - 1.0);
  app.rede.w2 = Array.from({ length: hidden }, () => Math.random() * 2.0 - 1.0);
  app.rede.b2 = Math.random() * 2.0 - 1.0;
}

function mlpForward(x, hcache) {
  const m = app.rede;
  let yin = m.b2;
  for (let j = 0; j < m.hidden; j++) {
    const hin = m.w1[j] * x + m.b1[j];
    hcache[j] = Math.tanh(hin);
    yin += m.w2[j] * hcache[j];
  }
  return Math.tanh(yin);
}

// ── Carregamento e janelas de dados ───────────────────────────
function carregarDadosEmbutidos() {
  app.dados = DATAS.map((dateStr, i) => {
    const p = parseData(dateStr);
    return { date: dateStr, close: FECHAMENTOS[i], y: p.y, m: p.m, d: p.d };
  });
  app.total = app.dados.length;
  app.carregado = true;
  app.rede.treinada = false;
  app.idxTrainIni = app.idxTrainFim = -1;
  app.idxValIni = app.idxValFim = -1;
  return true;
}

function definirJanelas() {
  let iniTrain = -1, fimTrain = -1;
  for (let i = 0; i < app.total; i++) {
    const r = app.dados[i];
    if (dataCmp(r, { y: 2025, m: 1, d: 1 }) >= 0 && dataCmp(r, { y: 2025, m: 6, d: 30 }) <= 0) {
      if (iniTrain < 0) iniTrain = i;
      fimTrain = i;
    }
  }
  if (iniTrain < 0 || fimTrain - iniTrain < 5) return false;

  const iniVal = fimTrain + 1;
  if (iniVal >= app.total) return false;
  let fimVal = iniVal + 6;
  if (fimVal >= app.total) fimVal = app.total - 1;
  if (fimVal - iniVal + 1 < 7) return false;

  app.idxTrainIni = iniTrain;
  app.idxTrainFim = fimTrain;
  app.idxValIni = iniVal;
  app.idxValFim = iniVal + 6;
  return true;
}

function calcularNormaTrain() {
  let minv = Infinity, maxv = -Infinity;
  for (let i = app.idxTrainIni; i <= app.idxTrainFim; i++) {
    const c = app.dados[i].close;
    if (c < minv) minv = c;
    if (c > maxv) maxv = c;
  }
  if (maxv - minv < 1e-9) return false;
  app.minTrain = minv;
  app.maxTrain = maxv;
  return true;
}

// ── Treinamento (mesma lógica do main.c) ──────────────────────
async function treinarRede(hidden) {
  if (!definirJanelas()) return { ok: false, msg: 'Falha no treino. Confira se há dados de jan-jun/2025 e +7 dias após isso.' };
  if (!calcularNormaTrain()) return { ok: false, msg: 'Falha no treino. Intervalo de treino sem variação de preço.' };
  if (hidden < 2 || hidden > 16) return { ok: false, msg: 'Número de neurônios inválido (use 2-16).' };

  mlpInicializar(hidden);
  const h = new Array(hidden);
  let epoca, mse = 0;

  for (epoca = 1; epoca <= MAX_EPOCAS; epoca++) {
    mse = 0.0;
    let n = 0;

    for (let i = app.idxTrainIni + 1; i <= app.idxTrainFim; i++) {
      const xRaw = app.dados[i - 1].close;
      const tRaw = app.dados[i].close;
      const x = normalizar(xRaw, app.minTrain, app.maxTrain);
      const t = normalizar(tRaw, app.minTrain, app.maxTrain);
      const y = mlpForward(x, h);
      const erro = t - y;

      const deltaOut = erro * tanhDeriv(y);
      for (let j = 0; j < app.rede.hidden; j++) {
        const deltaH = tanhDeriv(h[j]) * app.rede.w2[j] * deltaOut;
        app.rede.w2[j] += TAXA_APRENDIZADO * deltaOut * h[j];
        app.rede.w1[j] += TAXA_APRENDIZADO * deltaH * x;
        app.rede.b1[j] += TAXA_APRENDIZADO * deltaH;
      }
      app.rede.b2 += TAXA_APRENDIZADO * deltaOut;

      mse += erro * erro;
      n++;
    }

    mse /= (n > 0 ? n : 1);

    if (epoca === 1 || epoca % 2000 === 0) addLog(`Época ${epoca}: MSE=${mse.toFixed(8)}`);

    if (mse <= ERRO_ALVO) {
      app.rede.treinada = true;
      return { ok: true, epocas: epoca, erroFinal: mse };
    }

    if (epoca % 500 === 0) await new Promise(r => setTimeout(r, 0));
  }

  app.rede.treinada = true;
  return { ok: true, epocas: MAX_EPOCAS, erroFinal: mse };
}

// ── Validação e predição ──────────────────────────────────────
function validar7Dias() {
  const h = new Array(app.rede.hidden);
  let mae = 0, rmse = 0, mape = 0;
  const rows = [];

  for (let i = app.idxValIni; i <= app.idxValFim; i++) {
    const xRaw = app.dados[i - 1].close;
    const tRaw = app.dados[i].close;
    const x = normalizar(xRaw, app.minTrain, app.maxTrain);
    const yNorm = mlpForward(x, h);
    const yRaw = desnormalizar(yNorm, app.minTrain, app.maxTrain);
    const eabs = Math.abs(tRaw - yRaw);

    mae += eabs;
    rmse += (tRaw - yRaw) * (tRaw - yRaw);
    if (Math.abs(tRaw) > 1e-9) mape += (eabs / Math.abs(tRaw));

    rows.push({ date: app.dados[i].date, real: tRaw, pred: yRaw, eabs });
  }

  mae /= 7.0;
  rmse = Math.sqrt(rmse / 7.0);
  mape = (mape / 7.0) * 100.0;

  return { rows, mae, rmse, mape };
}

function preverProximoDia() {
  const h = new Array(app.rede.hidden);
  const ultimo = app.total - 1;
  const xRaw = app.dados[ultimo].close;
  const x = normalizar(xRaw, app.minTrain, app.maxTrain);
  const yNorm = mlpForward(x, h);
  const yRaw = desnormalizar(yNorm, app.minTrain, app.maxTrain);
  return { ultimaData: app.dados[ultimo].date, xRaw, yRaw };
}

// ── UI: log e helpers ──────────────────────────────────────────
function addLog(msg) {
  const box = document.getElementById('logBox');
  box.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`;
  box.scrollTop = box.scrollHeight;
}

function mostrarResumoDados() {
  document.getElementById('mTotal').textContent = app.total;
  if (app.total > 0) {
    document.getElementById('mPrimeira').textContent = `${app.dados[0].date} (R$ ${app.dados[0].close.toFixed(2)})`;
    document.getElementById('mUltima').textContent = `${app.dados[app.total - 1].date} (R$ ${app.dados[app.total - 1].close.toFixed(2)})`;
  }
  document.getElementById('mTreino').textContent = (app.idxTrainIni >= 0)
    ? `${app.dados[app.idxTrainIni].date} até ${app.dados[app.idxTrainFim].date}` : '—';
  document.getElementById('mValidacao').textContent = (app.idxValIni >= 0)
    ? `${app.dados[app.idxValIni].date} até ${app.dados[app.idxValFim].date}` : '—';
}

// ── Gráfico ────────────────────────────────────────────────────
function drawChart(predicao) {
  const canvas = document.getElementById('chartCanvas');
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');
  const PAD = { l: 46, r: 16, t: 12, b: 24 };
  const plotW = W - PAD.l - PAD.r, plotH = H - PAD.t - PAD.b;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#081523'; ctx.fillRect(0, 0, W, H);

  if (!app.carregado || app.total === 0) return;

  const n = app.total + (predicao ? 1 : 0);
  const closes = app.dados.map(d => d.close);
  const allVals = predicao ? [...closes, predicao.yRaw] : closes;
  const minV = Math.min(...allVals) - 0.3;
  const maxV = Math.max(...allVals) + 0.3;

  const toX = idx => PAD.l + (idx / Math.max(1, n - 1)) * plotW;
  const toY = v => PAD.t + (maxV - v) / (maxV - minV) * plotH;

  // grid
  ctx.strokeStyle = '#0d1f32'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const v = minV + i * (maxV - minV) / 4;
    const y = toY(v);
    ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(W - PAD.r, y); ctx.stroke();
    ctx.fillStyle = '#86a8bf'; ctx.font = '9px Consolas'; ctx.textAlign = 'right';
    ctx.fillText(v.toFixed(1), PAD.l - 5, y + 3);
  }

  function segmentColor(idx) {
    if (app.idxTrainIni >= 0 && idx >= app.idxTrainIni && idx <= app.idxTrainFim) return '#50ff5a';
    if (app.idxValIni >= 0 && idx >= app.idxValIni && idx <= app.idxValFim) return '#00d39f';
    return '#86a8bf';
  }

  // linha do fechamento, colorida por segmento
  for (let i = 1; i < app.total; i++) {
    ctx.strokeStyle = segmentColor(i);
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    ctx.moveTo(toX(i - 1), toY(closes[i - 1]));
    ctx.lineTo(toX(i), toY(closes[i]));
    ctx.stroke();
  }

  // predição do próximo dia
  if (predicao) {
    const xLast = toX(app.total - 1), yLast = toY(closes[app.total - 1]);
    const xPred = toX(app.total), yPred = toY(predicao.yRaw);

    ctx.strokeStyle = '#ff3f9b';
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(xLast, yLast); ctx.lineTo(xPred, yPred); ctx.stroke();
    ctx.setLineDash([]);

    ctx.fillStyle = '#ff3f9b';
    ctx.beginPath(); ctx.arc(xPred, yPred, 4, 0, Math.PI * 2); ctx.fill();
  }

  document.getElementById('chartLabel').textContent =
    predicao ? `próximo dia previsto: R$ ${predicao.yRaw.toFixed(4)}` : `${app.total} registros`;
}

// ── Ações de UI ────────────────────────────────────────────────
function onLoad() {
  carregarDadosEmbutidos();
  addLog(`[OK] Dados embutidos carregados. Registros: ${app.total}`);
  addLog('Treino: 2025-01 até 2025-06 | Validação: 7 dias após 2025-06-30');

  if (definirJanelas()) calcularNormaTrain();
  mostrarResumoDados();
  drawChart(null);

  document.getElementById('trainBtn').disabled = false;
}

async function onTrain() {
  if (running || !app.carregado) return;
  running = true;
  document.getElementById('trainBtn').disabled = true;
  document.getElementById('validateBtn').disabled = true;
  document.getElementById('predictBtn').disabled = true;

  const hidden = Math.min(16, Math.max(2, parseInt(document.getElementById('hiddenInput').value) || 8));

  addLog('--- INICIANDO TREINAMENTO ---');
  const result = await treinarRede(hidden);

  if (!result.ok) {
    addLog(`[!] ${result.msg}`);
    document.getElementById('mStatusTreino').textContent = 'Falha';
    document.getElementById('mStatusTreino').className = 'metric-val err';
  } else {
    addLog('[OK] Rede treinada.');
    addLog(`Épocas: ${result.epocas}`);
    addLog(`MSE final (normalizado): ${result.erroFinal.toFixed(8)}`);
    addLog(`Treino: ${app.dados[app.idxTrainIni].date} até ${app.dados[app.idxTrainFim].date}`);
    addLog(`Validação: ${app.dados[app.idxValIni].date} até ${app.dados[app.idxValFim].date}`);

    document.getElementById('mStatusTreino').textContent = 'Treinada';
    document.getElementById('mStatusTreino').className = 'metric-val';
    document.getElementById('mEpocas').textContent = result.epocas;
    document.getElementById('mMse').textContent = result.erroFinal.toFixed(8);

    mostrarResumoDados();
    drawChart(null);

    document.getElementById('validateBtn').disabled = false;
    document.getElementById('predictBtn').disabled = false;
  }

  running = false;
  document.getElementById('trainBtn').disabled = false;
}

function onValidate() {
  if (!app.carregado || !app.rede.treinada) {
    addLog('[!] Carregue os dados e treine a rede antes de validar.');
    return;
  }

  addLog('--- VALIDAÇÃO (7 DIAS) ---');
  const { rows, mae, rmse, mape } = validar7Dias();

  document.getElementById('validationTableBody').innerHTML = rows.map(r => `
    <tr>
      <td>${r.date}</td>
      <td>${r.real.toFixed(4)}</td>
      <td>${r.pred.toFixed(4)}</td>
      <td>${r.eabs.toFixed(4)}</td>
    </tr>`).join('');

  document.getElementById('mMae').textContent = mae.toFixed(6);
  document.getElementById('mRmse').textContent = rmse.toFixed(6);
  document.getElementById('mMape').textContent = mape.toFixed(4) + ' %';

  rows.forEach(r => addLog(`${r.date}  real=${r.real.toFixed(4)}  pred=${r.pred.toFixed(4)}  erro=${r.eabs.toFixed(4)}`));
  addLog(`MAE=${mae.toFixed(6)}  RMSE=${rmse.toFixed(6)}  MAPE=${mape.toFixed(4)}%`);
}

function onPredict() {
  if (!app.carregado || !app.rede.treinada) {
    addLog('[!] Carregue os dados e treine a rede antes de prever.');
    return;
  }

  const { ultimaData, xRaw, yRaw } = preverProximoDia();
  document.getElementById('mUltimoFechamento').textContent = `${ultimaData} · R$ ${xRaw.toFixed(4)}`;
  document.getElementById('mPredicao').textContent = `R$ ${yRaw.toFixed(4)}`;

  addLog('--- PREDIÇÃO DO DIA SEGUINTE ---');
  addLog(`Último fechamento conhecido (${ultimaData}): R$ ${xRaw.toFixed(4)}`);
  addLog(`Predição para o próximo dia útil: R$ ${yRaw.toFixed(4)}`);

  drawChart({ yRaw });
}

// ── Inicialização ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  drawChart(null);
  document.getElementById('loadBtn').addEventListener('click', onLoad);
  document.getElementById('trainBtn').addEventListener('click', onTrain);
  document.getElementById('validateBtn').addEventListener('click', onValidate);
  document.getElementById('predictBtn').addEventListener('click', onPredict);
});
