// ── Trabalho 02 — Perceptron Simples — Reconhecimento de Letras A e B ──

const LINHAS = 7;
const COLUNAS = 7;
const TAMANHO = 49;

const letraA = [
  [-1, 1, 1, 1, -1, -1, -1],
  [1, -1, -1, -1, 1, -1, -1],
  [1, -1, -1, -1, 1, -1, -1],
  [1, 1, 1, 1, 1, -1, -1],
  [1, -1, -1, -1, 1, -1, -1],
  [1, -1, -1, -1, 1, -1, -1],
  [1, -1, -1, -1, 1, -1, -1],
];

const letraB = [
  [1, 1, 1, 1, -1, -1, -1],
  [1, -1, -1, -1, 1, -1, -1],
  [1, -1, -1, -1, 1, -1, -1],
  [1, 1, 1, 1, -1, -1, -1],
  [1, -1, -1, -1, 1, -1, -1],
  [1, -1, -1, -1, 1, -1, -1],
  [1, 1, 1, 1, -1, -1, -1],
];

const target = [-1, 1]; // A = -1, B = 1

function flatten(matriz) {
  const v = [];
  for (let i = 0; i < LINHAS; i++)
    for (let j = 0; j < COLUNAS; j++)
      v.push(matriz[i][j]);
  return v;
}

const entrada = [flatten(letraA), flatten(letraB)];

// ── Estado da rede (persiste entre treinar/testar, como no C) ──
let w = [];
let b = 0;
let trained = false;
let ciclosTreinados = 0;
let running = false;

function initNetwork() {
  w = Array.from({ length: TAMANHO }, () => Math.random() - 0.5);
  b = Math.random() - 0.5;
  trained = false;
  ciclosTreinados = 0;
}

function predict(entradaLetra, limiar) {
  let soma = 0;
  for (let col = 0; col < TAMANHO; col++) soma += entradaLetra[col] * w[col];
  soma += b;
  return soma >= limiar ? 1 : -1;
}

// ── Renderização ─────────────────────────────────────────────
function renderLetterGrid(elId, matriz) {
  const el = document.getElementById(elId);
  el.innerHTML = '';
  for (let i = 0; i < LINHAS; i++) {
    for (let j = 0; j < COLUNAS; j++) {
      const px = document.createElement('div');
      px.className = 'pixel' + (matriz[i][j] === 1 ? ' on' : '');
      el.appendChild(px);
    }
  }
}

function addLog(msg) {
  const box = document.getElementById('logBox');
  box.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`;
  box.scrollTop = box.scrollHeight;
}

function updateWeightsDisplay() {
  const first10 = w.slice(0, 10).map((v, i) => `w${i}=${v.toFixed(4)}`).join('  ');
  document.getElementById('weightsDisplay').textContent = first10 + '  ...';
}

function updateStateMetrics() {
  document.getElementById('mStatus').textContent = trained
    ? `Treinada em ${ciclosTreinados} ciclos`
    : 'Não treinada';
  document.getElementById('mCiclo').textContent = ciclosTreinados;
  document.getElementById('mBias').textContent = b.toFixed(4);
  updateWeightsDisplay();
}

function renderTestTable(results) {
  document.getElementById('testTableBody').innerHTML = results.map(r => `
    <tr>
      <td>${r.nome}</td>
      <td>${r.target}</td>
      <td>${r.pred} (${r.predLetra})</td>
      <td class="${r.ok ? 'ok' : 'err'}">${r.ok ? '[OK]' : '[ERRO]'}</td>
    </tr>`).join('');
}

function resetUI() {
  document.getElementById('logBox').textContent = 'Aguardando execução...';
  document.getElementById('testTableBody').innerHTML = '<tr><td colspan="4">—</td></tr>';
  updateStateMetrics();
}

// ── Treinamento (mesma lógica do main.c) ────────────────────
async function trainNetwork() {
  if (running) return;
  running = true;
  document.getElementById('trainBtn').disabled = true;
  document.getElementById('testBtn').disabled = true;
  document.getElementById('logBox').textContent = '';

  const alfa = parseFloat(document.getElementById('alfaInput').value) || 0.1;
  const limiar = parseFloat(document.getElementById('limiarInput').value) || 0;
  const maxCiclos = Math.max(1, parseInt(document.getElementById('maxCiclosInput').value) || 1000);

  addLog('--- INICIANDO TREINAMENTO ---');
  addLog(`Parâmetros: α=${alfa} | limiar=${limiar} | máx. ciclos=${maxCiclos}`);

  let contCiclo = 0;
  let condErro = true;

  while (condErro) {
    condErro = false;

    for (let lin = 0; lin < 2; lin++) {
      let yLiq = 0;
      for (let col = 0; col < TAMANHO; col++) yLiq += entrada[lin][col] * w[col];
      yLiq += b;

      const y = yLiq >= limiar ? 1 : -1;

      if (y !== target[lin]) {
        condErro = true;
        for (let col = 0; col < TAMANHO; col++) {
          w[col] = w[col] + (alfa * (target[lin] - y) * entrada[lin][col]);
        }
        b = b + alfa * (target[lin] - y);
      }
    }

    contCiclo++;

    if (contCiclo % 10 === 0) addLog(`Ciclo: ${contCiclo}`);

    if (contCiclo > maxCiclos) {
      addLog('[!] Limite de ciclos atingido!');
      break;
    }

    if (contCiclo % 25 === 0) await new Promise(r => setTimeout(r, 0));
  }

  ciclosTreinados = contCiclo;

  if (!condErro) {
    addLog(`[OK] Rede treinada com sucesso em ${contCiclo} ciclos!`);
    trained = true;
  } else {
    trained = false;
  }

  addLog('--- PESOS FINAIS ---');
  addLog(`Bias: ${b.toFixed(4)}`);
  for (let col = 0; col < 10; col++) addLog(`Peso[${col}]: ${w[col].toFixed(4)}`);
  addLog('...');

  updateStateMetrics();

  running = false;
  document.getElementById('trainBtn').disabled = false;
  document.getElementById('testBtn').disabled = false;
}

// ── Teste (mesma lógica do main.c) ──────────────────────────
function testNetwork() {
  const limiar = parseFloat(document.getElementById('limiarInput').value) || 0;

  addLog('--- TESTANDO A REDE ---');

  const results = [];
  for (let lin = 0; lin < 2; lin++) {
    const yTeste = predict(entrada[lin], limiar);
    const nome = lin === 0 ? 'A' : 'B';
    const predLetra = yTeste === -1 ? 'A' : 'B';
    const ok = yTeste === target[lin];

    addLog(`Entrada: Letra ${nome} | Target: ${target[lin]} | Predicao: ${yTeste} (Letra ${predLetra}) ${ok ? '[OK]' : '[ERRO]'}`);
    results.push({ nome, target: target[lin], pred: yTeste, predLetra, ok });
  }

  renderTestTable(results);
}

// ── Inicialização ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  renderLetterGrid('gridA', letraA);
  renderLetterGrid('gridB', letraB);

  initNetwork();
  updateStateMetrics();

  document.getElementById('trainBtn').addEventListener('click', trainNetwork);
  document.getElementById('testBtn').addEventListener('click', () => {
    if (!running) testNetwork();
  });
  document.getElementById('resetBtn').addEventListener('click', () => {
    if (!running) {
      initNetwork();
      resetUI();
    }
  });
});
