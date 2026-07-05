// ── Trabalho 01 — Regra de Hebb — Portas Lógicas ──────────────

const GATES = [
  {
    name: 'AND',
    description: 'Retorna 1 apenas quando ambas entradas sao 1',
    inputs: [[1, 1], [1, -1], [-1, 1], [-1, -1]],
    targets: [1, -1, -1, -1],
  },
  {
    name: 'OR',
    description: 'Retorna 1 quando pelo menos uma entrada e 1',
    inputs: [[1, 1], [1, -1], [-1, 1], [-1, -1]],
    targets: [1, 1, 1, -1],
  },
  {
    name: 'NAND',
    description: 'Negacao do AND - retorna -1 apenas quando ambas sao 1',
    inputs: [[1, 1], [1, -1], [-1, 1], [-1, -1]],
    targets: [-1, 1, 1, 1],
  },
  {
    name: 'NOR',
    description: 'Negacao do OR - retorna 1 apenas quando ambas sao -1',
    inputs: [[1, 1], [1, -1], [-1, 1], [-1, -1]],
    targets: [-1, -1, -1, 1],
  },
];

function initNetwork() {
  return { w1: 0, w2: 0, bias: 0 };
}

function train(net, gate, log) {
  gate.inputs.forEach((inp, i) => {
    const [x1, x2] = inp;
    const y = gate.targets[i];

    net.w1 += x1 * y;
    net.w2 += x2 * y;
    net.bias += y;

    log(`Amostra ${i + 1}: Pesos atualizados -> W1: ${net.w1.toFixed(1)}, W2: ${net.w2.toFixed(1)}, Bias: ${net.bias.toFixed(1)}`);
  });
}

function predict(net, x1, x2) {
  const soma = (x1 * net.w1) + (x2 * net.w2) + net.bias;
  return soma >= 0 ? 1 : -1;
}

function testNetwork(net, gate) {
  let acertos = 0;
  const rows = gate.inputs.map((inp, i) => {
    const [x1, x2] = inp;
    const target = gate.targets[i];
    const pred = predict(net, x1, x2);
    const ok = pred === target;
    if (ok) acertos++;
    return { x1, x2, target, pred, ok };
  });
  return { rows, acertos, total: gate.inputs.length };
}

// ── Renderização ───────────────────────────────────────────
function renderGateGrid() {
  document.getElementById('gateGrid').innerHTML = GATES.map(g => `
    <div class="gate-item">
      <span class="gi">${g.name}</span>
      <span class="desc">${g.description}</span>
    </div>
  `).join('');
}

function renderTruthTable(gate) {
  return `
    <table class="data-table">
      <thead><tr><th>X1</th><th>X2</th><th>Alvo</th></tr></thead>
      <tbody>
        ${gate.inputs.map((inp, i) => `<tr><td>${inp[0]}</td><td>${inp[1]}</td><td>${gate.targets[i]}</td></tr>`).join('')}
      </tbody>
    </table>`;
}

function renderTestTable(result) {
  return `
    <table class="data-table">
      <thead><tr><th>X1</th><th>X2</th><th>Alvo</th><th>Predição</th><th>Status</th></tr></thead>
      <tbody>
        ${result.rows.map(r => `
          <tr>
            <td>${r.x1}</td><td>${r.x2}</td><td>${r.target}</td><td>${r.pred}</td>
            <td class="${r.ok ? 'ok' : 'err'}">${r.ok ? '[OK]' : '[ERRO]'}</td>
          </tr>`).join('')}
      </tbody>
    </table>`;
}

function renderGateResult(gate, net, result) {
  const acc = ((result.acertos / result.total) * 100).toFixed(0);
  return `
    <div class="result-block">
      <div class="result-title">Porta ${gate.name}</div>
      <div class="result-desc">${gate.description}</div>

      <div class="section-label">Tabela verdade (bipolar)</div>
      ${renderTruthTable(gate)}

      <div class="result-metrics">
        <div class="metric"><span class="metric-label">W1</span><span class="metric-val blue">${net.w1.toFixed(1)}</span></div>
        <div class="metric"><span class="metric-label">W2</span><span class="metric-val blue">${net.w2.toFixed(1)}</span></div>
        <div class="metric"><span class="metric-label">Bias</span><span class="metric-val blue">${net.bias.toFixed(1)}</span></div>
        <div class="metric"><span class="metric-label">Acurácia</span><span class="metric-val pink">${result.acertos}/${result.total} (${acc}%)</span></div>
      </div>

      <div class="section-label">Teste final</div>
      ${renderTestTable(result)}
    </div>`;
}

// ── UI helpers ─────────────────────────────────────────────
function addLog(msg) {
  const box = document.getElementById('logBox');
  box.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`;
  box.scrollTop = box.scrollHeight;
}

function resetUI() {
  document.getElementById('resultsContainer').innerHTML = '<p class="info-text">Aguardando execução...</p>';
  document.getElementById('logBox').textContent = 'Aguardando execução...';
}

// ── Execução principal ──────────────────────────────────────
let running = false;

async function runAll() {
  if (running) return;
  running = true;

  const runBtn = document.getElementById('runBtn');
  runBtn.disabled = true;
  document.getElementById('logBox').textContent = '';
  document.getElementById('resultsContainer').innerHTML = '';

  addLog('Iniciando aprendizado de portas logicas pela Regra de Hebb...');

  for (const gate of GATES) {
    addLog(`--- Porta: ${gate.name} — ${gate.description} ---`);
    const net = initNetwork();

    addLog('Iniciando Treinamento (Regra de Hebb)...');
    train(net, gate, addLog);

    addLog(`Pesos Finais: W1=${net.w1.toFixed(1)}, W2=${net.w2.toFixed(1)}, Bias=${net.bias.toFixed(1)}`);

    const result = testNetwork(net, gate);
    const acc = ((result.acertos / result.total) * 100).toFixed(0);
    addLog(`Acuracia: ${result.acertos}/${result.total} (${acc}%)`);

    document.getElementById('resultsContainer')
      .insertAdjacentHTML('beforeend', renderGateResult(gate, net, result));

    await new Promise(r => setTimeout(r, 0));
  }

  addLog('─── Treinamento concluído para todas as portas ───');
  running = false;
  runBtn.disabled = false;
}

// ── Inicialização ────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', () => {
  renderGateGrid();

  document.getElementById('runBtn').addEventListener('click', runAll);
  document.getElementById('resetBtn').addEventListener('click', () => {
    if (!running) resetUI();
  });
});
