  // ── Intervalo e função ──────────────────────────────────────
  const X_MIN = 0;
  const X_MAX = 512;

  function f(x) {
    return -(Math.abs(x * Math.sin(Math.sqrt(Math.abs(x)))));
  }

  // ── Cromossomo ─────────────────────────────────────────────
  function decode(bits) {
    let val = 0;
    for (let i = 0; i < bits.length; i++) val = val * 2 + bits[i];
    const maxInt = (1 << bits.length) - 1;
    return X_MIN + val * (X_MAX - X_MIN) / maxInt;
  }

  function randomChromosome(n) {
    return Array.from({length: n}, () => Math.random() < 0.5 ? 1 : 0);
  }

  // ── Operadores do AG ───────────────────────────────────────

  // gerarImagem: aplica f(x) em cada indivíduo
  function gerarImagem(pop) {
    return pop.map(c => f(decode(c)));
  }

  // gerarProbabilidades: roleta adaptada para minimização
  // Inverte a fitness: individuo com menor f(x) tem maior probabilidade
  function gerarProbabilidades(imagemFuncao) {
    const maxF = Math.max(...imagemFuncao);
    const shifted = imagemFuncao.map(v => maxF - v + 1e-9);
    const total = shifted.reduce((a, b) => a + b, 0);
    const cumProbs = [];
    let cum = 0;
    for (const s of shifted) {
      cum += s / total;
      cumProbs.push(cum);
    }
    return cumProbs;
  }

  function rouletteSelect(cumProbs, pop) {
    const r = Math.random();
    let idx = cumProbs.findIndex(p => r <= p);
    if (idx < 0) idx = pop.length - 1;
    return [...pop[idx]];
  }

  // separarMelhores: seleciona n indivíduos pela roleta
  function separarMelhores(cumProbs, pop, n) {
    return Array.from({length: n}, () => rouletteSelect(cumProbs, pop));
  }

  function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  // gerarPontoDeCorte: ponto de corte aleatório entre 1 e BITS-1
  function gerarPontoDeCorte(n) {
    return Math.floor(Math.random() * (n - 1)) + 1;
  }

  // cruzamento: crossover de ponto único em cada par
  function cruzamento(casais) {
    const filhos = [];
    for (const [p1, p2] of casais) {
      const pt = gerarPontoDeCorte(p1.length);
      filhos.push([...p1.slice(0, pt), ...p2.slice(pt)]);
      filhos.push([...p2.slice(0, pt), ...p1.slice(pt)]);
    }
    return filhos;
  }

  // efetuarMutacao: mutação apenas nos filhos (flip de 1 bit por indivíduo)
  function efetuarMutacao(filhos, mutProb) {
    return filhos.map(filho => {
      if (Math.random() < mutProb) {
        const m = [...filho];
        const bit = Math.floor(Math.random() * m.length);
        m[bit] = 1 - m[bit];
        return m;
      }
      return filho;
    });
  }

  function findBest(pop) {
    let best = null;
    for (const c of pop) {
      const x = decode(c);
      const fx = f(x);
      if (!best || fx < best.fx) best = {x, fx, chrom: c};
    }
    return best;
  }

  // ── Uma geração ────────────────────────────────────────────
  function runGeneration(pop, crossProb, mutProb) {
    const n = pop.length;

    const imagemFuncao = gerarImagem(pop);
    const cumProbs = gerarProbabilidades(imagemFuncao);

    // Seleciona n indivíduos pela roleta
    const selected = separarMelhores(cumProbs, pop, n);
    shuffle(selected);

    // Separa os que vão cruzar (prob. de cruzamento)
    let numCross = Math.floor(n * crossProb);
    if (numCross % 2 !== 0) numCross--;   // precisa ser par para formar casais

    const toCross  = selected.slice(0, numCross);
    const noChange = selected.slice(numCross);

    // sortearCasais: forma pares entre os selecionados para cruzamento
    const casais = [];
    for (let i = 0; i + 1 < toCross.length; i += 2) {
      casais.push([toCross[i], toCross[i + 1]]);
    }

    // Cruzamento → filhos
    const filhos = cruzamento(casais);

    // Mutação apenas nos filhos
    const filhosMutados = efetuarMutacao(filhos, mutProb);

    // Nova população: não cruzados + filhos (com possível mutação)
    return [...noChange, ...filhosMutados];
  }

  // ── Gráfico ────────────────────────────────────────────────
  let gFmin = 0;
  let gFmax = 0;

  function precomputeRange() {
    const N = 2000;
    let mn = 0;
    for (let i = 0; i <= N; i++) {
      const x = X_MIN + i * (X_MAX - X_MIN) / N;
      const v = f(x);
      if (v < mn) mn = v;
    }
    gFmin = mn * 1.05;
    gFmax = 0;
  }

  function drawGraph(bestX) {
    const canvas = document.getElementById('graphCanvas');
    const W = canvas.width;
    const H = canvas.height;
    const ctx = canvas.getContext('2d');
    const PAD = {l: 50, r: 14, t: 12, b: 30};
    const plotW = W - PAD.l - PAD.r;
    const plotH = H - PAD.t - PAD.b;

    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#081523';
    ctx.fillRect(0, 0, W, H);

    const toCanX = x  => PAD.l + (x - X_MIN) / (X_MAX - X_MIN) * plotW;
    const toCanY = fy => PAD.t + (gFmax - fy) / (gFmax - gFmin) * plotH;

    // Grid horizontal
    const gridN = 5;
    for (let i = 0; i <= gridN; i++) {
      const fy = gFmin + i * (gFmax - gFmin) / gridN;
      const cy = toCanY(fy);
      ctx.strokeStyle = i === gridN ? '#1e3f5f' : '#0d1f32';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(PAD.l, cy); ctx.lineTo(W - PAD.r, cy); ctx.stroke();
      ctx.fillStyle = '#86a8bf';
      ctx.font = '9px Consolas';
      ctx.textAlign = 'right';
      ctx.fillText(fy.toFixed(0), PAD.l - 5, cy + 3);
    }

    // Grid vertical + labels x
    const gridX = 4;
    for (let i = 0; i <= gridX; i++) {
      const xv = X_MIN + i * (X_MAX - X_MIN) / gridX;
      const cx = toCanX(xv);
      ctx.strokeStyle = '#0d1f32';
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(cx, PAD.t); ctx.lineTo(cx, H - PAD.b); ctx.stroke();
      ctx.fillStyle = '#86a8bf';
      ctx.font = '9px Consolas';
      ctx.textAlign = 'center';
      ctx.fillText(xv.toFixed(0), cx, H - PAD.b + 12);
    }

    // Eixos
    ctx.strokeStyle = '#1e3f5f';
    ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.moveTo(PAD.l, PAD.t); ctx.lineTo(PAD.l, H - PAD.b); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD.l, H - PAD.b); ctx.lineTo(W - PAD.r, H - PAD.b); ctx.stroke();

    // Curva f(x)
    const N = plotW * 2;
    ctx.strokeStyle = '#00d39f';
    ctx.lineWidth = 1.5;
    ctx.beginPath();
    for (let i = 0; i <= N; i++) {
      const x  = X_MIN + i * (X_MAX - X_MIN) / N;
      const cx = toCanX(x);
      const cy = toCanY(f(x));
      i === 0 ? ctx.moveTo(cx, cy) : ctx.lineTo(cx, cy);
    }
    ctx.stroke();

    // Marcador do melhor x*
    if (bestX != null) {
      const bfx = f(bestX);
      const bx  = toCanX(bestX);
      const by  = toCanY(bfx);

      ctx.strokeStyle = '#ff3f9b';
      ctx.lineWidth = 1.5;
      ctx.setLineDash([5, 4]);
      ctx.beginPath(); ctx.moveTo(bx, PAD.t); ctx.lineTo(bx, H - PAD.b); ctx.stroke();
      ctx.setLineDash([]);

      // Ponto
      ctx.fillStyle = '#ff3f9b';
      ctx.beginPath(); ctx.arc(bx, by, 5, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = '#0d1d2f';
      ctx.beginPath(); ctx.arc(bx, by, 2, 0, Math.PI * 2); ctx.fill();
    }

    // Label eixo x
    ctx.fillStyle = '#86a8bf';
    ctx.font = '10px Consolas';
    ctx.textAlign = 'center';
    ctx.fillText('x', W / 2, H - 2);

    // Label eixo y
    ctx.save();
    ctx.translate(10, PAD.t + plotH / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.fillText('f(x)', 0, 0);
    ctx.restore();
  }

  // ── UI helpers ─────────────────────────────────────────────
  function addLog(msg) {
    const box = document.getElementById('logBox');
    const t = new Date().toLocaleTimeString();
    box.textContent += `[${t}] ${msg}\n`;
    box.scrollTop = box.scrollHeight;
  }

  function updateMetrics(gen, best, crossProb, mutProb) {
    document.getElementById('mGen').textContent = gen;
    document.getElementById('mX').textContent   = best.x.toFixed(6);
    document.getElementById('mFx').textContent  = best.fx.toFixed(6);
    document.getElementById('mCross').textContent = (crossProb * 100).toFixed(0) + '%';
    document.getElementById('mMut').textContent   = (mutProb   * 100).toFixed(0) + '%';

    document.getElementById('bitsDisplay').innerHTML =
      best.chrom.map(b => `<span class="${b ? 'b1' : 'b0'}">${b}</span>`).join('');

    document.getElementById('graphLabel').textContent =
      `x* = ${best.x.toFixed(2)} | f(x*) = ${best.fx.toFixed(2)}`;
  }

  function resetUI() {
    document.getElementById('mGen').textContent  = '0';
    document.getElementById('mX').textContent    = '—';
    document.getElementById('mFx').textContent   = '—';
    document.getElementById('mCross').textContent= '—';
    document.getElementById('mMut').textContent  = '—';
    document.getElementById('bitsDisplay').textContent = '——————————';
    document.getElementById('graphLabel').textContent  = '—';
    document.getElementById('logBox').textContent = 'Aguardando execução...';
    drawGraph(null);
  }

  // ── Execução principal ─────────────────────────────────────
  let running = false;

  async function runAG() {
    if (running) return;
    running = true;

    const runBtn = document.getElementById('runBtn');
    runBtn.disabled = true;
    document.getElementById('logBox').textContent = '';

    const popSize   = Math.max(4, parseInt(document.getElementById('popSize').value)    || 20);
    const gens      = Math.max(1, parseInt(document.getElementById('generations').value) || 100);
    const BITS      = 10;
    const crossProb = parseInt(document.getElementById('crossSlider').value) / 100;
    const mutProb   = parseInt(document.getElementById('mutSlider').value)   / 100;

    addLog(`Parâmetros: pop=${popSize} | ger=${gens} | bits=${BITS} | cruz=${(crossProb*100).toFixed(0)}% | mut=${(mutProb*100).toFixed(0)}%`);
    addLog('Inicializando população aleatória...');

    // popBin = gerarElementosBinarios → população inicial aleatória em binário
    let pop = Array.from({length: popSize}, () => randomChromosome(BITS));

    let overallBest = null;

    for (let gen = 1; gen <= gens; gen++) {
      pop = runGeneration(pop, crossProb, mutProb);

      const best = findBest(pop);
      if (!overallBest || best.fx < overallBest.fx) overallBest = best;

      updateMetrics(gen, overallBest, crossProb, mutProb);
      drawGraph(overallBest.x);

      if (gen === 1 || gen % 10 === 0 || gen === gens) {
        addLog(`Gen ${String(gen).padStart(4)}: x*=${overallBest.x.toFixed(5).padStart(10)} | f(x*)=${overallBest.fx.toFixed(5)}`);
      }

      if (gen % 5 === 0) await new Promise(r => setTimeout(r, 0));
    }

    addLog('─── AG finalizado ───');
    addLog(`Melhor x*    = ${overallBest.x.toFixed(8)}`);
    addLog(`Melhor f(x*) = ${overallBest.fx.toFixed(8)}`);
    addLog(`Cromossomo   = [${overallBest.chrom.join('')}]`);

    running = false;
    runBtn.disabled = false;
  }

  // ── Boot ───────────────────────────────────────────────────
  document.addEventListener('DOMContentLoaded', () => {
    precomputeRange();
    drawGraph(null);

    document.getElementById('crossSlider').addEventListener('input', function() {
      document.getElementById('crossVal').textContent = this.value + '%';
    });
    document.getElementById('mutSlider').addEventListener('input', function() {
      document.getElementById('mutVal').textContent = this.value + '%';
    });

    document.getElementById('runBtn').addEventListener('click', runAG);
    document.getElementById('resetBtn').addEventListener('click', () => {
      if (!running) resetUI();
    });
  });
