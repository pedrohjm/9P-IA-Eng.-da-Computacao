// ── Trabalho 03 — MADALINE — Reconhecimento de 13 Letras (A-M) ──

const LINHAS = 7;
const COLUNAS = 7;
const TAMANHO = 49;
const NUM_LETRAS = 13;

const letras = [
  // A
  [[-1, 1, 1, 1, -1, -1, -1], [1, -1, -1, -1, 1, -1, -1], [1, -1, -1, -1, 1, -1, -1],
   [1, 1, 1, 1, 1, -1, -1], [1, -1, -1, -1, 1, -1, -1], [1, -1, -1, -1, 1, -1, -1], [1, -1, -1, -1, 1, -1, -1]],
  // B
  [[1, 1, 1, 1, -1, -1, -1], [1, -1, -1, -1, 1, -1, -1], [1, -1, -1, -1, 1, -1, -1],
   [1, 1, 1, 1, -1, -1, -1], [1, -1, -1, -1, 1, -1, -1], [1, -1, -1, -1, 1, -1, -1], [1, 1, 1, 1, -1, -1, -1]],
  // C
  [[-1, 1, 1, 1, 1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1],
   [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [-1, 1, 1, 1, 1, -1, -1]],
  // D
  [[1, 1, 1, 1, -1, -1, -1], [1, -1, -1, -1, 1, -1, -1], [1, -1, -1, -1, 1, -1, -1],
   [1, -1, -1, -1, 1, -1, -1], [1, -1, -1, -1, 1, -1, -1], [1, -1, -1, -1, 1, -1, -1], [1, 1, 1, 1, -1, -1, -1]],
  // E
  [[1, 1, 1, 1, 1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1],
   [1, 1, 1, 1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, -1, -1]],
  // F
  [[1, 1, 1, 1, 1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1],
   [1, 1, 1, 1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1]],
  // G
  [[-1, 1, 1, 1, 1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1],
   [1, -1, 1, 1, 1, -1, -1], [1, -1, -1, -1, 1, -1, -1], [1, -1, -1, -1, 1, -1, -1], [-1, 1, 1, 1, -1, -1, -1]],
  // H
  [[1, -1, -1, -1, 1, -1, -1], [1, -1, -1, -1, 1, -1, -1], [1, -1, -1, -1, 1, -1, -1],
   [1, 1, 1, 1, 1, -1, -1], [1, -1, -1, -1, 1, -1, -1], [1, -1, -1, -1, 1, -1, -1], [1, -1, -1, -1, 1, -1, -1]],
  // I
  [[1, 1, 1, 1, 1, -1, -1], [-1, -1, 1, -1, -1, -1, -1], [-1, -1, 1, -1, -1, -1, -1],
   [-1, -1, 1, -1, -1, -1, -1], [-1, -1, 1, -1, -1, -1, -1], [-1, -1, 1, -1, -1, -1, -1], [1, 1, 1, 1, 1, -1, -1]],
  // J
  [[1, 1, 1, 1, 1, -1, -1], [-1, -1, -1, 1, -1, -1, -1], [-1, -1, -1, 1, -1, -1, -1],
   [-1, -1, -1, 1, -1, -1, -1], [1, -1, -1, 1, -1, -1, -1], [1, -1, -1, 1, -1, -1, -1], [-1, 1, 1, -1, -1, -1, -1]],
  // K
  [[1, -1, -1, -1, 1, -1, -1], [1, -1, -1, 1, -1, -1, -1], [1, -1, 1, -1, -1, -1, -1],
   [1, 1, -1, -1, -1, -1, -1], [1, -1, 1, -1, -1, -1, -1], [1, -1, -1, 1, -1, -1, -1], [1, -1, -1, -1, 1, -1, -1]],
  // L
  [[1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1],
   [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, -1, -1, -1, -1, -1, -1], [1, 1, 1, 1, 1, -1, -1]],
  // M
  [[1, -1, -1, -1, 1, -1, -1], [1, 1, -1, 1, 1, -1, -1], [1, -1, 1, -1, 1, -1, -1],
   [1, -1, -1, -1, 1, -1, -1], [1, -1, -1, -1, 1, -1, -1], [1, -1, -1, -1, 1, -1, -1], [1, -1, -1, -1, 1, -1, -1]],
];

const nomesLetras = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M'];

function flatten(matriz) {
  const v = [];
  for (let i = 0; i < LINHAS; i++)
    for (let j = 0; j < COLUNAS; j++)
      v.push(matriz[i][j]);
  return v;
}

const entradasLetras = letras.map(flatten);

// ── Estado da rede (persiste entre treinar/testar) ───────────
let pesos = [];
let bias = [];
let alfa = 0.05;
let redeTreinada = false;
let epocasTreinadas = 0;
let erroFinal = 0;
let running = false;

let matrizUsuario = Array.from({ length: LINHAS }, () => Array(COLUNAS).fill(-1));

function inicializarRede() {
  pesos = Array.from({ length: NUM_LETRAS }, () =>
    Array.from({ length: TAMANHO }, () => Math.random() * 0.2 - 0.1));
  bias = Array.from({ length: NUM_LETRAS }, () => Math.random() * 0.2 - 0.1);
  alfa = 0.05;
  redeTreinada = false;
  epocasTreinadas = 0;
  erroFinal = 0;
}

function ativacaoLinear(valor) {
  return valor >= 0 ? 1.0 : -1.0;
}

function reconhecerLetra(entrada) {
  let maxSaida = -999999.0;
  let letraReconhecida = -1;

  for (let neuronio = 0; neuronio < NUM_LETRAS; neuronio++) {
    let soma = bias[neuronio];
    for (let i = 0; i < TAMANHO; i++) soma += entrada[i] * pesos[neuronio][i];
    if (soma > maxSaida) {
      maxSaida = soma;
      letraReconhecida = neuronio;
    }
  }
  return letraReconhecida;
}

// ── Renderização ─────────────────────────────────────────────
function renderPixelGrid(container, matriz, interactive) {
  container.innerHTML = '';
  for (let i = 0; i < LINHAS; i++) {
    for (let j = 0; j < COLUNAS; j++) {
      const px = document.createElement('div');
      px.className = 'pixel' + (matriz[i][j] === 1 ? ' on' : '');
      if (interactive) {
        px.addEventListener('click', () => {
          matrizUsuario[i][j] = matrizUsuario[i][j] === 1 ? -1 : 1;
          px.classList.toggle('on', matrizUsuario[i][j] === 1);
        });
      }
      container.appendChild(px);
    }
  }
}

function renderLetterGallery() {
  const gallery = document.getElementById('letterGallery');
  gallery.innerHTML = '';
  letras.forEach((matriz, idx) => {
    const card = document.createElement('div');
    card.className = 'letter-card';
    card.innerHTML = `<div class="letter-title">${nomesLetras[idx]}</div>`;
    const grid = document.createElement('div');
    grid.className = 'pixel-grid';
    card.appendChild(grid);
    gallery.appendChild(card);
    renderPixelGrid(grid, matriz, false);
  });
}

function renderDrawGrid() {
  renderPixelGrid(document.getElementById('drawGrid'), matrizUsuario, true);
}

function addLog(msg) {
  const box = document.getElementById('logBox');
  box.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`;
  box.scrollTop = box.scrollHeight;
}

function updateStateMetrics() {
  document.getElementById('mStatus').textContent = redeTreinada
    ? `Treinada em ${epocasTreinadas} épocas`
    : 'Não treinada';
  document.getElementById('mEpoca').textContent = epocasTreinadas;
  document.getElementById('mErro').textContent = redeTreinada ? erroFinal.toFixed(4) : '—';
}

function renderTestTable(results) {
  document.getElementById('testTableBody').innerHTML = results.map(r => `
    <tr>
      <td>${r.real}</td>
      <td>${r.reconhecida}</td>
      <td class="${r.ok ? 'ok' : 'err'}">${r.ok ? '[OK]' : '[ERRO]'}</td>
    </tr>`).join('');
}

function resetUI() {
  document.getElementById('logBox').textContent = 'Aguardando execução...';
  document.getElementById('testTableBody').innerHTML = '<tr><td colspan="3">—</td></tr>';
  document.getElementById('mAcc').textContent = '—';
  document.getElementById('recognizedLetter').textContent = '—';
  document.getElementById('testBtn').disabled = true;
  document.getElementById('recognizeBtn').disabled = true;
  updateStateMetrics();
}

// ── Treinamento (mesma lógica do main.c) ────────────────────
async function treinarRede() {
  if (running) return;
  running = true;
  document.getElementById('trainBtn').disabled = true;
  document.getElementById('testBtn').disabled = true;
  document.getElementById('recognizeBtn').disabled = true;
  document.getElementById('logBox').textContent = '';

  alfa = parseFloat(document.getElementById('alfaInput').value) || 0.05;
  const maxEpocas = Math.max(1, parseInt(document.getElementById('maxEpocasInput').value) || 1000);

  addLog('--- INICIANDO TREINAMENTO MADALINE ---');
  addLog(`Taxa de aprendizagem: ${alfa.toFixed(4)}`);

  let epoca;
  let erroTotal = 0;

  for (epoca = 0; epoca < maxEpocas; epoca++) {
    erroTotal = 0.0;

    for (let letraIdx = 0; letraIdx < NUM_LETRAS; letraIdx++) {
      const entrada = entradasLetras[letraIdx];

      for (let neuronio = 0; neuronio < NUM_LETRAS; neuronio++) {
        const target = neuronio === letraIdx ? 1.0 : -1.0;

        let soma = bias[neuronio];
        for (let i = 0; i < TAMANHO; i++) soma += entrada[i] * pesos[neuronio][i];

        const saida = ativacaoLinear(soma);
        const erro = target - saida;
        erroTotal += Math.abs(erro);

        if (erro !== 0) {
          for (let i = 0; i < TAMANHO; i++) {
            pesos[neuronio][i] += alfa * erro * entrada[i];
          }
          bias[neuronio] += alfa * erro;
        }
      }
    }

    if ((epoca + 1) % 100 === 0) addLog(`Época ${epoca + 1} | Erro Total: ${erroTotal.toFixed(4)}`);

    if (erroTotal < 0.1) {
      addLog(`[OK] Convergência atingida na época ${epoca + 1}!`);
      break;
    }

    if (epoca % 20 === 0) await new Promise(r => setTimeout(r, 0));
  }

  if (epoca >= maxEpocas) addLog('[!] Limite de épocas atingido.');

  addLog('Treinamento concluído!');

  redeTreinada = true;
  epocasTreinadas = Math.min(epoca + 1, maxEpocas);
  erroFinal = erroTotal;

  updateStateMetrics();

  running = false;
  document.getElementById('trainBtn').disabled = false;
  document.getElementById('testBtn').disabled = false;
  document.getElementById('recognizeBtn').disabled = false;
}

// ── Teste (mesma lógica do main.c) ──────────────────────────
function testarRede() {
  if (!redeTreinada) {
    addLog('[!] A rede ainda não foi treinada!');
    return;
  }

  addLog('--- TESTANDO A REDE ---');

  let acertos = 0;
  const results = [];
  for (let letraIdx = 0; letraIdx < NUM_LETRAS; letraIdx++) {
    const resultado = reconhecerLetra(entradasLetras[letraIdx]);
    const ok = resultado === letraIdx;
    if (ok) acertos++;

    addLog(`Letra Real: ${nomesLetras[letraIdx]} | Reconhecida: ${nomesLetras[resultado]} ${ok ? '[OK]' : '[ERRO]'}`);
    results.push({ real: nomesLetras[letraIdx], reconhecida: nomesLetras[resultado], ok });
  }

  const acc = (acertos * 100.0) / NUM_LETRAS;
  addLog(`Acurácia: ${acertos}/${NUM_LETRAS} (${acc.toFixed(1)}%)`);

  renderTestTable(results);
  document.getElementById('mAcc').textContent = `${acertos}/${NUM_LETRAS} (${acc.toFixed(1)}%)`;
}

// ── Desenho e reconhecimento ─────────────────────────────────
function reconhecerDesenho() {
  if (!redeTreinada) {
    addLog('[!] A rede ainda não foi treinada!');
    return;
  }
  const entrada = flatten(matrizUsuario);
  const resultado = reconhecerLetra(entrada);
  document.getElementById('recognizedLetter').textContent = nomesLetras[resultado];
  addLog(`Desenho reconhecido como: Letra ${nomesLetras[resultado]}`);
}

function limparDesenho() {
  matrizUsuario = Array.from({ length: LINHAS }, () => Array(COLUNAS).fill(-1));
  renderDrawGrid();
  document.getElementById('recognizedLetter').textContent = '—';
}

// ── Inicialização ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  renderLetterGallery();
  renderDrawGrid();
  inicializarRede();
  updateStateMetrics();

  document.getElementById('trainBtn').addEventListener('click', treinarRede);
  document.getElementById('testBtn').addEventListener('click', () => {
    if (!running) testarRede();
  });
  document.getElementById('recognizeBtn').addEventListener('click', () => {
    if (!running) reconhecerDesenho();
  });
  document.getElementById('clearDrawBtn').addEventListener('click', limparDesenho);
  document.getElementById('resetBtn').addEventListener('click', () => {
    if (!running) {
      inicializarRede();
      resetUI();
    }
  });
});
