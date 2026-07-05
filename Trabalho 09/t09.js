  // ── Domínio e função ────────────────────────────────────────
  const X_MIN = 0, X_MAX = 512;

  function f(x) {
    return -(Math.abs(x * Math.sin(Math.sqrt(Math.abs(x)))));
  }

  // ── Cromossomo ──────────────────────────────────────────────
  function decode(bits) {
    let val = 0;
    for (let i = 0; i < bits.length; i++) val = val * 2 + bits[i];
    return X_MIN + val * (X_MAX - X_MIN) / (Math.pow(2, bits.length) - 1);
  }

  function randomChromosome(n) {
    return Array.from({length: n}, () => Math.random() < 0.5 ? 1 : 0);
  }

  // ── Seleção por Roleta ──────────────────────────────────────
  function gerarImagem(pop) { return pop.map(c => f(decode(c))); }

  function gerarProbabilidades(img) {
    const maxF = Math.max(...img);
    const shifted = img.map(v => maxF - v + 1e-9);
    const total = shifted.reduce((a, b) => a + b, 0);
    let cum = 0;
    return shifted.map(s => (cum += s / total));
  }

  function rouletteSelect(cumProbs, pop) {
    const r = Math.random();
    let idx = cumProbs.findIndex(p => r <= p);
    return [...pop[idx < 0 ? pop.length - 1 : idx]];
  }

  // ── Seleção por Torneio ─────────────────────────────────────
  // Escolhe k indivíduos aleatoriamente e retorna o mais apto (menor f(x))
  function tournamentSelect(pop, k) {
    let best = pop[Math.floor(Math.random() * pop.length)];
    for (let i = 1; i < k; i++) {
      const c = pop[Math.floor(Math.random() * pop.length)];
      if (f(decode(c)) < f(decode(best))) best = c;
    }
    return [...best];
  }

  // ── Elitismo ────────────────────────────────────────────────
  // Retorna cópias dos p melhores indivíduos (menor f(x) = melhor)
  function getElites(pop, p) {
    if (p <= 0) return [];
    return [...pop]
      .sort((a, b) => f(decode(a)) - f(decode(b)))
      .slice(0, p)
      .map(c => [...c]);
  }

  // ── Cruzamento em 1 ponto ───────────────────────────────────
  function cruzamento1pt(p1, p2) {
    const pt = Math.floor(Math.random() * (p1.length - 1)) + 1;
    return [
      [...p1.slice(0, pt), ...p2.slice(pt)],
      [...p2.slice(0, pt), ...p1.slice(pt)]
    ];
  }

  // ── Cruzamento em 2 pontos ──────────────────────────────────
  // c1 = A(p1) + B(p2) + C(p1)   c2 = A(p2) + B(p1) + C(p2)
  function cruzamento2pt(p1, p2) {
    const n = p1.length;
    if (n < 3) return cruzamento1pt(p1, p2);
    let pt1, pt2;
    do {
      pt1 = Math.floor(Math.random() * (n - 1)) + 1;
      pt2 = Math.floor(Math.random() * (n - 1)) + 1;
    } while (pt1 === pt2);
    if (pt1 > pt2) [pt1, pt2] = [pt2, pt1];
    return [
      [...p1.slice(0, pt1), ...p2.slice(pt1, pt2), ...p1.slice(pt2)],
      [...p2.slice(0, pt1), ...p1.slice(pt1, pt2), ...p2.slice(pt2)]
    ];
  }

  // ── Mutação (somente filhos) ────────────────────────────────
  function efetuarMutacao(filhos, mutProb) {
    return filhos.map(filho => {
      if (Math.random() < mutProb) {
        const m = [...filho];
        m[Math.floor(Math.random() * m.length)] ^= 1;
        return m;
      }
      return filho;
    });
  }

  function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  function findBest(pop) {
    let best = null;
    for (const c of pop) {
      const x = decode(c), fx = f(x);
      if (!best || fx < best.fx) best = {x, fx, chrom: c};
    }
    return best;
  }

  // ── Uma geração ─────────────────────────────────────────────
  function runGeneration(pop, cfg) {
    const {crossProb, mutProb, selMethod, k, crossType, elitism} = cfg;
    const n = pop.length;

    // 1. Salvar elites antes de modificar a população
    const elites = getElites(pop, elitism);

    // 2. Seleção (roleta ou torneio)
    let selected;
    if (selMethod === 'roleta') {
      const cum = gerarProbabilidades(gerarImagem(pop));
      selected = Array.from({length: n}, () => rouletteSelect(cum, pop));
    } else {
      const kEff = Math.min(k, n);
      selected = Array.from({length: n}, () => tournamentSelect(pop, kEff));
    }

    // 3. Separar quem vai cruzar (prob. de cruzamento)
    shuffle(selected);
    let numCross = Math.floor(n * crossProb);
    if (numCross % 2 !== 0) numCross = Math.max(0, numCross - 1);

    const toCross  = selected.slice(0, numCross);
    const noChange = selected.slice(numCross);

    // 4. Cruzamento — sortear casais e gerar filhos
    const filhos = [];
    for (let i = 0; i + 1 < toCross.length; i += 2) {
      const [c1, c2] = crossType === '1pt'
        ? cruzamento1pt(toCross[i], toCross[i + 1])
        : cruzamento2pt(toCross[i], toCross[i + 1]);
      filhos.push(c1, c2);
    }

    // 5. Mutação somente nos filhos
    const filhosMutados = efetuarMutacao(filhos, mutProb);

    // 6. Nova população: não cruzados + filhos mutados
    const newPop = [...noChange, ...filhosMutados];

    // 7. Aplicar elitismo: substituir os piores pelos elites
    newPop.sort((a, b) => f(decode(a)) - f(decode(b)));
    return [...elites, ...newPop.slice(0, Math.max(0, n - elitism))];
  }

  // ── Gráfico ─────────────────────────────────────────────────
  let gFmin = -500;

  function precomputeRange() {
    let mn = 0;
    const N = 3000;
    for (let i = 1; i <= N; i++) {
      const v = f(X_MIN + i * (X_MAX - X_MIN) / N);
      if (v < mn) mn = v;
    }
    gFmin = mn * 1.06;
  }

  function drawGraph(bestX) {
    const canvas = document.getElementById('graphCanvas');
    const W = canvas.width, H = canvas.height;
    const ctx = canvas.getContext('2d');
    const PAD = {l: 50, r: 14, t: 12, b: 28};
    const pW = W - PAD.l - PAD.r;
    const pH = H - PAD.t - PAD.b;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#081523';
    ctx.fillRect(0, 0, W, H);

    const toX = x  => PAD.l + (x - X_MIN) / (X_MAX - X_MIN) * pW;
    const toY = fy => PAD.t + (-fy / -gFmin) * pH;

    // Grid horizontal
    for (let i = 0; i <= 5; i++) {
      const fy = gFmin * (1 - i / 5);
      const cy = toY(fy);
      ctx.strokeStyle = i === 5 ? '#1e3f5f' : '#0d1f32';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(PAD.l, cy); ctx.lineTo(W - PAD.r, cy); ctx.stroke();
      ctx.fillStyle = '#86a8bf'; ctx.font = '9px Consolas'; ctx.textAlign = 'right';
      ctx.fillText(fy.toFixed(0), PAD.l - 4, cy + 3);
    }

    // Grid vertical + labels x
    for (let i = 0; i <= 4; i++) {
      const xv = X_MIN + i * (X_MAX - X_MIN) / 4;
      const cx = toX(xv);
      ctx.strokeStyle = '#0d1f32'; ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx, PAD.t); ctx.lineTo(cx, H - PAD.b); ctx.stroke();
      ctx.fillStyle = '#86a8bf'; ctx.font = '9px Consolas'; ctx.textAlign = 'center';
      ctx.fillText(xv.toFixed(0), cx, H - PAD.b + 10);
    }

    // Eixos
    ctx.strokeStyle = '#1e3f5f'; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(PAD.l, PAD.t); ctx.lineTo(PAD.l, H - PAD.b); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD.l, H - PAD.b); ctx.lineTo(W - PAD.r, H - PAD.b); ctx.stroke();

    // Curva f(x)
    ctx.strokeStyle = '#00d39f'; ctx.lineWidth = 1.5;
    ctx.beginPath();
    const steps = pW * 2;
    for (let i = 0; i <= steps; i++) {
      const x = X_MIN + i * (X_MAX - X_MIN) / steps;
      i === 0 ? ctx.moveTo(toX(x), toY(f(x))) : ctx.lineTo(toX(x), toY(f(x)));
    }
    ctx.stroke();

    // Marcador x*
    if (bestX != null) {
      const bx = toX(bestX), by = toY(f(bestX));
      ctx.strokeStyle = '#ff3f9b'; ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(bx, PAD.t); ctx.lineTo(bx, H - PAD.b); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = '#ff3f9b';
      ctx.beginPath(); ctx.arc(bx, by, 5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#0d1d2f';
      ctx.beginPath(); ctx.arc(bx, by, 2, 0, Math.PI * 2); ctx.fill();
    }

    // Label eixo x
    ctx.fillStyle = '#86a8bf'; ctx.font = '10px Consolas'; ctx.textAlign = 'center';
    ctx.fillText('x', W / 2, H - 2);
    ctx.save();
    ctx.translate(10, PAD.t + pH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.fillText('f(x)', 0, 0);
    ctx.restore();
  }

  // ── UI helpers ──────────────────────────────────────────────
  function addLog(msg) {
    const box = document.getElementById('logBox');
    box.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`;
    box.scrollTop = box.scrollHeight;
  }

  function updateMetrics(gen, best, cfg) {
    const {selMethod, k, crossType, elitism, bits} = cfg;
    document.getElementById('mGen').textContent   = gen;
    document.getElementById('mX').textContent     = best.x.toFixed(6);
    document.getElementById('mFx').textContent    = best.fx.toFixed(6);
    document.getElementById('mSel').textContent   =
      selMethod === 'roleta' ? 'Roleta' : `Torneio (k = ${k})`;
    document.getElementById('mCross').textContent =
      crossType === '1pt' ? '1 ponto' : '2 pontos';
    document.getElementById('mElite').textContent =
      elitism > 0 ? `${elitism} preservado(s)` : 'Desativado';
    document.getElementById('mBits').textContent  = bits;
    document.getElementById('bitsDisplay').innerHTML =
      best.chrom.map(b => `<span class="${b ? 'b1' : 'b0'}">${b}</span>`).join('');
    document.getElementById('graphLabel').textContent =
      `x* = ${best.x.toFixed(2)} | f(x*) = ${best.fx.toFixed(2)}`;
  }

  function resetUI() {
    ['mX','mFx','mSel','mCross','mElite','mBits'].forEach(
      id => (document.getElementById(id).textContent = '—')
    );
    document.getElementById('mGen').textContent = '0';
    document.getElementById('bitsDisplay').textContent = '——————————';
    document.getElementById('graphLabel').textContent  = '—';
    document.getElementById('logBox').textContent = 'Aguardando execução...';
    drawGraph(null);
  }

  // ── Execução principal ──────────────────────────────────────
  let running = false;

  async function runAG() {
    if (running) return;
    running = true;
    document.getElementById('runBtn').disabled = true;
    document.getElementById('logBox').textContent = '';

    const popSize   = Math.max(4,  parseInt(document.getElementById('popSize').value)    || 20);
    const gens      = Math.max(1,  parseInt(document.getElementById('generations').value) || 100);
    const bits      = Math.min(20, Math.max(4, parseInt(document.getElementById('bits').value) || 10));
    const crossProb = parseInt(document.getElementById('crossSlider').value) / 100;
    const mutProb   = parseInt(document.getElementById('mutSlider').value)   / 100;
    const elitism   = Math.max(0, Math.min(popSize - 1,
                        parseInt(document.getElementById('elitism').value) || 0));
    const selMethod = document.querySelector('input[name="selMethod"]:checked').value;
    const k         = Math.max(2, parseInt(document.getElementById('kTorneio').value) || 3);
    const crossType = document.querySelector('input[name="crossType"]:checked').value;

    const cfg = {crossProb, mutProb, selMethod, k, crossType, elitism, bits};

    const selLbl   = selMethod === 'roleta' ? 'Roleta' : `Torneio (k=${k})`;
    const crossLbl = crossType === '1pt' ? '1 ponto' : '2 pontos';

    addLog(`Pop: ${popSize} | Ger: ${gens} | Bits: ${bits} | Cruz: ${(crossProb*100).toFixed(0)}% | Mut: ${(mutProb*100).toFixed(0)}%`);
    addLog(`Seleção: ${selLbl} | Cruzamento: ${crossLbl} | Elitismo: p=${elitism}`);
    addLog('Inicializando população aleatória...');

    let pop = Array.from({length: popSize}, () => randomChromosome(bits));
    let overallBest = null;

    for (let gen = 1; gen <= gens; gen++) {
      pop = runGeneration(pop, cfg);
      const best = findBest(pop);
      if (!overallBest || best.fx < overallBest.fx) overallBest = best;

      updateMetrics(gen, overallBest, cfg);
      drawGraph(overallBest.x);

      if (gen === 1 || gen % 10 === 0 || gen === gens) {
        addLog(`Gen ${String(gen).padStart(4)}: x* = ${overallBest.x.toFixed(4).padStart(9)} | f(x*) = ${overallBest.fx.toFixed(4)}`);
      }

      if (gen % 5 === 0) await new Promise(r => setTimeout(r, 0));
    }

    addLog('─── AG Finalizado ───');
    addLog(`Melhor x*    = ${overallBest.x.toFixed(8)}`);
    addLog(`Melhor f(x*) = ${overallBest.fx.toFixed(8)}`);
    addLog(`Cromossomo   = [${overallBest.chrom.join('')}]`);

    running = false;
    document.getElementById('runBtn').disabled = false;
  }

  // ── Boot ─────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    precomputeRange();
    drawGraph(null);

    document.getElementById('crossSlider').addEventListener('input', function() {
      document.getElementById('crossVal').textContent = this.value + '%';
    });
    document.getElementById('mutSlider').addEventListener('input', function() {
      document.getElementById('mutVal').textContent = this.value + '%';
    });

    document.querySelectorAll('input[name="selMethod"]').forEach(r => {
      r.addEventListener('change', () => {
        document.getElementById('torneioCfg').style.display =
          document.getElementById('selTorneio').checked ? 'flex' : 'none';
      });
    });

    document.getElementById('runBtn').addEventListener('click', runAG);
    document.getElementById('resetBtn').addEventListener('click', () => {
      if (!running) resetUI();
    });
  });
