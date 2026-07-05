// ── Trabalho 04 — MLP Backpropagation (ativação tanh) ──

const MAX_ENTRADA = 8;
const MAX_OCULTA = 8;
const MAX_SAIDA = 8;
const MAX_PADROES = 20;

const DEFAULT_ALPHA = 0.01;
const DEFAULT_MAX_CICLOS = 100000;
const DEFAULT_ERRO_ALVO = 0.01;

// ── Estado central da rede ───────────────────────────────────
const rede = {
  nEntrada: 3, nOculta: 2, nSaida: 3, nPadroes: 3,
  alpha: DEFAULT_ALPHA, erroAlvo: DEFAULT_ERRO_ALVO, maxCiclos: DEFAULT_MAX_CICLOS,

  X: [], T: [],
  v: [], v0: [], w: [], w0: [],

  ciclosExecutados: 0, erroFinal: 0, convergiu: false,
  Yfinal: [], Ylim: [],
};

let running = false;

function zeros2D(rows, cols) {
  return Array.from({ length: rows }, () => Array(cols).fill(0));
}

// ── Reset de configuração / padrões / pesos (mesma lógica do main.c) ──
function resetConfigPadrao() {
  rede.nEntrada = 3;
  rede.nOculta = 2;
  rede.nSaida = 3;
  rede.nPadroes = 3;
  rede.alpha = DEFAULT_ALPHA;
  rede.maxCiclos = DEFAULT_MAX_CICLOS;
  rede.erroAlvo = DEFAULT_ERRO_ALVO;
}

function resetPadroes() {
  if (rede.nPadroes === 3 && rede.nEntrada === 3 && rede.nSaida === 3) {
    rede.X = [[1, 0.5, -1], [0, 0.5, 1], [1, -0.5, -1]];
    rede.T = [[1, -1, -1], [-1, 1, -1], [-1, -1, 1]];
    return;
  }
  rede.X = zeros2D(rede.nPadroes, rede.nEntrada);
  rede.T = zeros2D(rede.nPadroes, rede.nSaida);
}

function resetPesosIniciais() {
  if (rede.nEntrada === 3 && rede.nOculta === 2 && rede.nSaida === 3) {
    rede.v = [[0.12, -0.03], [-0.04, 0.15], [0.31, -0.41]];
    rede.v0 = [-0.09, 0.18];
    rede.w = [[-0.05, -0.34, 0.21], [0.19, -0.09, 0.26]];
    rede.w0 = [0.18, -0.27, -0.12];
    return;
  }

  rede.v = Array.from({ length: rede.nEntrada }, () =>
    Array.from({ length: rede.nOculta }, () => Math.random() - 0.5));
  rede.v0 = Array.from({ length: rede.nOculta }, () => Math.random() - 0.5);
  rede.w = Array.from({ length: rede.nOculta }, () =>
    Array.from({ length: rede.nSaida }, () => Math.random() - 0.5));
  rede.w0 = Array.from({ length: rede.nSaida }, () => Math.random() - 0.5);
}

// ── Ativação ─────────────────────────────────────────────────
function fAtiv(x) { return Math.tanh(x); }
function dfAtiv(fx) { return (1.0 + fx) * (1.0 - fx); }

// ── Forward pass de um padrão ────────────────────────────────
function forward(Xp) {
  const zin = new Array(rede.nOculta);
  const z = new Array(rede.nOculta);
  for (let j = 0; j < rede.nOculta; j++) {
    zin[j] = rede.v0[j];
    for (let i = 0; i < rede.nEntrada; i++) zin[j] += Xp[i] * rede.v[i][j];
    z[j] = fAtiv(zin[j]);
  }
  const yin = new Array(rede.nSaida);
  const y = new Array(rede.nSaida);
  for (let k = 0; k < rede.nSaida; k++) {
    yin[k] = rede.w0[k];
    for (let j = 0; j < rede.nOculta; j++) yin[k] += z[j] * rede.w[j][k];
    y[k] = fAtiv(yin[k]);
  }
  return { z, y };
}

// ── Treinamento (mesma lógica do main.c: backprop online) ────
async function treinar() {
  rede.convergiu = false;
  let ciclo, et = 0;

  addLog('--- INICIANDO TREINAMENTO (Backpropagation) ---');
  addLog(`α=${rede.alpha} | erro alvo=${rede.erroAlvo} | máx. ciclos=${rede.maxCiclos}`);

  for (ciclo = 1; ciclo <= rede.maxCiclos; ciclo++) {
    et = 0.0;

    for (let p = 0; p < rede.nPadroes; p++) {
      const { z, y } = forward(rede.X[p]);

      for (let k = 0; k < rede.nSaida; k++) {
        et += 0.5 * (rede.T[p][k] - y[k]) * (rede.T[p][k] - y[k]);
      }

      // backward
      const dk = new Array(rede.nSaida);
      for (let k = 0; k < rede.nSaida; k++) dk[k] = (rede.T[p][k] - y[k]) * dfAtiv(y[k]);

      const dw = zeros2D(rede.nOculta, rede.nSaida);
      const dw0 = new Array(rede.nSaida);
      for (let j = 0; j < rede.nOculta; j++)
        for (let k = 0; k < rede.nSaida; k++) dw[j][k] = rede.alpha * dk[k] * z[j];
      for (let k = 0; k < rede.nSaida; k++) dw0[k] = rede.alpha * dk[k];

      const dinJ = new Array(rede.nOculta).fill(0);
      for (let j = 0; j < rede.nOculta; j++) {
        dinJ[j] = 0.0;
        for (let k = 0; k < rede.nSaida; k++) dinJ[j] += dk[k] * rede.w[j][k];
      }
      const dj = new Array(rede.nOculta);
      for (let j = 0; j < rede.nOculta; j++) dj[j] = dinJ[j] * dfAtiv(z[j]);

      const dv = zeros2D(rede.nEntrada, rede.nOculta);
      const dv0 = new Array(rede.nOculta);
      for (let i = 0; i < rede.nEntrada; i++)
        for (let j = 0; j < rede.nOculta; j++) dv[i][j] = rede.alpha * dj[j] * rede.X[p][i];
      for (let j = 0; j < rede.nOculta; j++) dv0[j] = rede.alpha * dj[j];

      // atualiza pesos
      for (let j = 0; j < rede.nOculta; j++)
        for (let k = 0; k < rede.nSaida; k++) rede.w[j][k] += dw[j][k];
      for (let k = 0; k < rede.nSaida; k++) rede.w0[k] += dw0[k];
      for (let i = 0; i < rede.nEntrada; i++)
        for (let j = 0; j < rede.nOculta; j++) rede.v[i][j] += dv[i][j];
      for (let j = 0; j < rede.nOculta; j++) rede.v0[j] += dv0[j];
    }

    rede.erroFinal = et;
    rede.ciclosExecutados = ciclo;

    if (ciclo === 1 || ciclo % 1000 === 0) addLog(`Ciclo ${ciclo}: erro=${et.toFixed(8)}`);

    if (et <= rede.erroAlvo) { rede.convergiu = true; break; }

    if (ciclo % 200 === 0) await new Promise(r => setTimeout(r, 0));
  }

  if (rede.convergiu) addLog(`[OK] Convergiu em ${rede.ciclosExecutados} ciclos! Erro final=${rede.erroFinal.toFixed(8)}`);
  else addLog(`[!] Não convergiu após ${rede.ciclosExecutados} ciclos. Erro final=${rede.erroFinal.toFixed(8)}`);

  // fase de operação
  rede.Yfinal = zeros2D(rede.nPadroes, rede.nSaida);
  rede.Ylim = zeros2D(rede.nPadroes, rede.nSaida);
  for (let p = 0; p < rede.nPadroes; p++) {
    const { y } = forward(rede.X[p]);
    for (let k = 0; k < rede.nSaida; k++) {
      rede.Yfinal[p][k] = y[k];
      rede.Ylim[p][k] = y[k] >= 0.0 ? 1.0 : -1.0;
    }
  }
}

// ── Renderização ─────────────────────────────────────────────
function addLog(msg) {
  const box = document.getElementById('logBox');
  box.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`;
  box.scrollTop = box.scrollHeight;
}

function renderConfigInputs() {
  document.getElementById('cfgEntrada').value = rede.nEntrada;
  document.getElementById('cfgOculta').value = rede.nOculta;
  document.getElementById('cfgSaida').value = rede.nSaida;
  document.getElementById('cfgPadroes').value = rede.nPadroes;
  document.getElementById('cfgAlpha').value = rede.alpha;
  document.getElementById('cfgErroAlvo').value = rede.erroAlvo;
  document.getElementById('cfgMaxCiclos').value = rede.maxCiclos;
}

function renderPatternsTable() {
  let html = '<table class="data-table"><thead><tr><th>Pad</th>';
  for (let i = 0; i < rede.nEntrada; i++) html += `<th>x${i + 1}</th>`;
  html += '<th class="divider">t1</th>';
  for (let k = 1; k < rede.nSaida; k++) html += `<th>t${k + 1}</th>`;
  html += '</tr></thead><tbody>';

  for (let p = 0; p < rede.nPadroes; p++) {
    html += `<tr><td>${p + 1}</td>`;
    for (let i = 0; i < rede.nEntrada; i++) {
      html += `<td><input type="number" step="0.1" value="${rede.X[p][i]}" data-p="${p}" data-i="${i}" class="xInput" /></td>`;
    }
    for (let k = 0; k < rede.nSaida; k++) {
      html += `<td class="${k === 0 ? 'divider' : ''}"><input type="number" step="0.1" value="${rede.T[p][k]}" data-p="${p}" data-k="${k}" class="tInput" /></td>`;
    }
    html += '</tr>';
  }
  html += '</tbody></table>';
  document.getElementById('patternsTable').innerHTML = html;

  document.querySelectorAll('.xInput').forEach(inp => {
    inp.addEventListener('change', () => {
      rede.X[+inp.dataset.p][+inp.dataset.i] = parseFloat(inp.value) || 0;
    });
  });
  document.querySelectorAll('.tInput').forEach(inp => {
    inp.addEventListener('change', () => {
      rede.T[+inp.dataset.p][+inp.dataset.k] = parseFloat(inp.value) || 0;
    });
  });
}

function renderWeightsTables() {
  // v [entrada][oculta] + bias v0
  let html = '<table class="data-table"><thead><tr><th>v</th>';
  for (let j = 0; j < rede.nOculta; j++) html += `<th>oculta ${j + 1}</th>`;
  html += '</tr></thead><tbody>';
  for (let i = 0; i < rede.nEntrada; i++) {
    html += `<tr><td>entrada ${i + 1}</td>`;
    for (let j = 0; j < rede.nOculta; j++) {
      html += `<td><input type="number" step="0.01" value="${rede.v[i][j].toFixed(4)}" data-i="${i}" data-j="${j}" class="vInput" /></td>`;
    }
    html += '</tr>';
  }
  html += '<tr><td>bias v0</td>';
  for (let j = 0; j < rede.nOculta; j++) {
    html += `<td><input type="number" step="0.01" value="${rede.v0[j].toFixed(4)}" data-j="${j}" class="v0Input" /></td>`;
  }
  html += '</tr></tbody></table>';
  document.getElementById('weightsVTable').innerHTML = html;

  // w [oculta][saida] + bias w0
  html = '<table class="data-table"><thead><tr><th>w</th>';
  for (let k = 0; k < rede.nSaida; k++) html += `<th>saída ${k + 1}</th>`;
  html += '</tr></thead><tbody>';
  for (let j = 0; j < rede.nOculta; j++) {
    html += `<tr><td>oculta ${j + 1}</td>`;
    for (let k = 0; k < rede.nSaida; k++) {
      html += `<td><input type="number" step="0.01" value="${rede.w[j][k].toFixed(4)}" data-j="${j}" data-k="${k}" class="wInput" /></td>`;
    }
    html += '</tr>';
  }
  html += '<tr><td>bias w0</td>';
  for (let k = 0; k < rede.nSaida; k++) {
    html += `<td><input type="number" step="0.01" value="${rede.w0[k].toFixed(4)}" data-k="${k}" class="w0Input" /></td>`;
  }
  html += '</tr></tbody></table>';
  document.getElementById('weightsWTable').innerHTML = html;

  document.querySelectorAll('.vInput').forEach(inp => {
    inp.addEventListener('change', () => { rede.v[+inp.dataset.i][+inp.dataset.j] = parseFloat(inp.value) || 0; });
  });
  document.querySelectorAll('.v0Input').forEach(inp => {
    inp.addEventListener('change', () => { rede.v0[+inp.dataset.j] = parseFloat(inp.value) || 0; });
  });
  document.querySelectorAll('.wInput').forEach(inp => {
    inp.addEventListener('change', () => { rede.w[+inp.dataset.j][+inp.dataset.k] = parseFloat(inp.value) || 0; });
  });
  document.querySelectorAll('.w0Input').forEach(inp => {
    inp.addEventListener('change', () => { rede.w0[+inp.dataset.k] = parseFloat(inp.value) || 0; });
  });
}

function renderReadOnlyWeights(containerIdV, containerIdW) {
  let html = '<table class="data-table"><thead><tr><th>v final</th>';
  for (let j = 0; j < rede.nOculta; j++) html += `<th>oculta ${j + 1}</th>`;
  html += '</tr></thead><tbody>';
  for (let i = 0; i < rede.nEntrada; i++) {
    html += `<tr><td>entrada ${i + 1}</td>`;
    for (let j = 0; j < rede.nOculta; j++) html += `<td>${rede.v[i][j].toFixed(5)}</td>`;
    html += '</tr>';
  }
  html += '<tr><td>bias v0</td>';
  for (let j = 0; j < rede.nOculta; j++) html += `<td>${rede.v0[j].toFixed(5)}</td>`;
  html += '</tr></tbody></table>';
  document.getElementById(containerIdV).innerHTML = html;

  html = '<table class="data-table"><thead><tr><th>w final</th>';
  for (let k = 0; k < rede.nSaida; k++) html += `<th>saída ${k + 1}</th>`;
  html += '</tr></thead><tbody>';
  for (let j = 0; j < rede.nOculta; j++) {
    html += `<tr><td>oculta ${j + 1}</td>`;
    for (let k = 0; k < rede.nSaida; k++) html += `<td>${rede.w[j][k].toFixed(5)}</td>`;
    html += '</tr>';
  }
  html += '<tr><td>bias w0</td>';
  for (let k = 0; k < rede.nSaida; k++) html += `<td>${rede.w0[k].toFixed(5)}</td>`;
  html += '</tr></tbody></table>';
  document.getElementById(containerIdW).innerHTML = html;
}

function renderResultsTable() {
  let html = '<table class="data-table"><thead><tr><th>Pad</th>';
  for (let k = 0; k < rede.nSaida; k++) html += `<th>y${k + 1}</th>`;
  html += `<th class="divider">lim1</th>`;
  for (let k = 1; k < rede.nSaida; k++) html += `<th>lim${k + 1}</th>`;
  html += `<th class="divider">alvo1</th>`;
  for (let k = 1; k < rede.nSaida; k++) html += `<th>alvo${k + 1}</th>`;
  html += '</tr></thead><tbody>';

  for (let p = 0; p < rede.nPadroes; p++) {
    html += `<tr><td>${p + 1}</td>`;
    for (let k = 0; k < rede.nSaida; k++) html += `<td>${rede.Yfinal[p][k].toFixed(4)}</td>`;
    for (let k = 0; k < rede.nSaida; k++) {
      const ok = rede.Ylim[p][k] === rede.T[p][k];
      html += `<td class="${k === 0 ? 'divider ' : ''}${ok ? 'ok' : 'err'}">${rede.Ylim[p][k] > 0 ? '+1' : '-1'}</td>`;
    }
    for (let k = 0; k < rede.nSaida; k++) html += `<td class="${k === 0 ? 'divider' : ''}">${rede.T[p][k] > 0 ? '+' : ''}${rede.T[p][k]}</td>`;
    html += '</tr>';
  }
  html += '</tbody></table>';
  document.getElementById('resultsTable').innerHTML = html;
}

function updateResultMetrics() {
  document.getElementById('mStatus').textContent = rede.convergiu ? 'CONVERGIU' : 'NÃO CONVERGIU';
  document.getElementById('mStatus').className = 'metric-val ' + (rede.convergiu ? '' : 'err');
  document.getElementById('mCiclos').textContent = rede.ciclosExecutados;
  document.getElementById('mErro').textContent = rede.erroFinal.toFixed(8);
  document.getElementById('mErroAlvo').textContent = rede.erroAlvo;
}

// ── Ações ──────────────────────────────────────────────────
function readConfigFromInputs() {
  rede.nEntrada = clamp(parseInt(document.getElementById('cfgEntrada').value) || 3, 1, MAX_ENTRADA);
  rede.nOculta = clamp(parseInt(document.getElementById('cfgOculta').value) || 2, 1, MAX_OCULTA);
  rede.nSaida = clamp(parseInt(document.getElementById('cfgSaida').value) || 3, 1, MAX_SAIDA);
  rede.nPadroes = clamp(parseInt(document.getElementById('cfgPadroes').value) || 3, 1, MAX_PADROES);

  let alpha = parseFloat(document.getElementById('cfgAlpha').value);
  rede.alpha = (!alpha || alpha <= 0) ? 0.01 : alpha;

  let erroAlvo = parseFloat(document.getElementById('cfgErroAlvo').value);
  rede.erroAlvo = (!erroAlvo || erroAlvo <= 0) ? 0.001 : erroAlvo;

  let maxCiclos = parseInt(document.getElementById('cfgMaxCiclos').value);
  rede.maxCiclos = (!maxCiclos || maxCiclos < 1) ? 1000 : maxCiclos;
}

function clamp(v, lo, hi) { return Math.max(lo, Math.min(hi, v)); }

function applyConfig() {
  readConfigFromInputs();
  renderConfigInputs();
  resetPadroes();
  resetPesosIniciais();
  renderPatternsTable();
  renderWeightsTables();
  addLog(`Configuração aplicada: entradas=${rede.nEntrada}, ocultas=${rede.nOculta}, saídas=${rede.nSaida}, padrões=${rede.nPadroes}`);
}

async function runTraining() {
  if (running) return;
  running = true;
  document.getElementById('trainBtn').disabled = true;
  document.getElementById('applyBtn').disabled = true;

  readConfigFromInputs();
  renderConfigInputs();

  await treinar();

  renderResultsTable();
  updateResultMetrics();
  renderReadOnlyWeights('finalVTable', 'finalWTable');

  running = false;
  document.getElementById('trainBtn').disabled = false;
  document.getElementById('applyBtn').disabled = false;
}

// ── Inicialização ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  resetConfigPadrao();
  renderConfigInputs();
  resetPadroes();
  resetPesosIniciais();
  renderPatternsTable();
  renderWeightsTables();

  document.getElementById('applyBtn').addEventListener('click', applyConfig);
  document.getElementById('resetConfigBtn').addEventListener('click', () => {
    resetConfigPadrao();
    renderConfigInputs();
    resetPadroes();
    resetPesosIniciais();
    renderPatternsTable();
    renderWeightsTables();
    addLog('Configuração restaurada para os valores padrão.');
  });

  document.getElementById('resetPatternsBtn').addEventListener('click', () => {
    resetPadroes();
    renderPatternsTable();
    addLog('Padrões de treinamento restaurados.');
  });

  document.getElementById('resetWeightsBtn').addEventListener('click', () => {
    resetPesosIniciais();
    renderWeightsTables();
    addLog('Pesos iniciais restaurados.');
  });

  document.getElementById('trainBtn').addEventListener('click', runTraining);
});
