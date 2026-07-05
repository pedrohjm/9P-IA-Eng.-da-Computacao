// ── Trabalho 13 — AG Cromossomos Reais · Rastrigin ────────────

const A = 10;
const X_MIN = -5.12, X_MAX = 5.12;

function rastrigin(xs) {
  return A * xs.length + xs.reduce((s, x) => s + x*x - A * Math.cos(2 * Math.PI * x), 0);
}

function clamp(x) { return Math.max(X_MIN, Math.min(X_MAX, x)); }

function randomChromosome(n) {
  return Array.from({length: n}, () => X_MIN + Math.random() * (X_MAX - X_MIN));
}

function gaussian() {
  let u, v, s;
  do { u = 2*Math.random()-1; v = 2*Math.random()-1; s = u*u + v*v; } while (s >= 1 || s === 0);
  return u * Math.sqrt(-2 * Math.log(s) / s);
}

// ── Seleção ──────────────────────────────────────────────────────

function rouletteProbs(fitnesses) {
  const maxF = Math.max(...fitnesses);
  const shifted = fitnesses.map(f => maxF - f + 1e-9);
  const total = shifted.reduce((a, b) => a + b, 0);
  let cum = 0;
  return shifted.map(s => (cum += s / total));
}

function rouletteSelect(cum, pop) {
  const r = Math.random();
  const idx = cum.findIndex(v => r <= v);
  return [...pop[idx < 0 ? pop.length-1 : idx]];
}

function tournamentSelect(pop, fitnesses, k) {
  let bi = Math.floor(Math.random() * pop.length);
  for (let i = 1; i < k; i++) {
    const idx = Math.floor(Math.random() * pop.length);
    if (fitnesses[idx] < fitnesses[bi]) bi = idx;
  }
  return [...pop[bi]];
}

// ── Cruzamento ───────────────────────────────────────────────────

function arithmeticCross(p1, p2) {
  const a = Math.random();
  return [
    p1.map((g, i) =>  a*g + (1-a)*p2[i]),
    p2.map((g, i) =>  a*g + (1-a)*p1[i])
  ];
}

function blxCross(p1, p2) {
  const alpha = 0.5;
  function gene(a, b) {
    const lo = Math.min(a, b), hi = Math.max(a, b), I = hi - lo;
    return clamp(lo - alpha*I + Math.random() * (I + 2*alpha*I));
  }
  return [p1.map((g, i) => gene(g, p2[i])), p1.map((g, i) => gene(g, p2[i]))];
}

function sbxCross(p1, p2) {
  const eta = 2;
  function sbxGene(x1, x2) {
    if (Math.abs(x1 - x2) < 1e-10) return [x1, x2];
    const u = Math.random();
    const beta = u <= 0.5
      ? Math.pow(2*u, 1/(eta+1))
      : Math.pow(0.5/(1-u), 1/(eta+1));
    return [
      clamp(0.5*((1+beta)*x1 + (1-beta)*x2)),
      clamp(0.5*((1-beta)*x1 + (1+beta)*x2))
    ];
  }
  const c1 = [], c2 = [];
  for (let i = 0; i < p1.length; i++) { const [g1,g2] = sbxGene(p1[i],p2[i]); c1.push(g1); c2.push(g2); }
  return [c1, c2];
}

// ── Mutação ──────────────────────────────────────────────────────

function gaussMutate(chrom, prob, sigma) {
  return chrom.map(g => Math.random() < prob ? clamp(g + sigma * gaussian()) : g);
}

function uniformMutate(chrom, prob) {
  return chrom.map(g => Math.random() < prob ? X_MIN + Math.random()*(X_MAX-X_MIN) : g);
}

// ── Elitismo ─────────────────────────────────────────────────────

function getElites(pop, fitnesses, n) {
  if (n <= 0) return [];
  return pop
    .map((c, i) => ({c, f: fitnesses[i]}))
    .sort((a, b) => a.f - b.f)
    .slice(0, n)
    .map(e => [...e.c]);
}

// ── Uma geração ──────────────────────────────────────────────────

function runGeneration(pop, fitnesses, cfg) {
  const n = pop.length;
  const elites = getElites(pop, fitnesses, cfg.elitism);

  let selected;
  if (cfg.selMethod === 'roleta') {
    const cum = rouletteProbs(fitnesses);
    selected = Array.from({length: n}, () => rouletteSelect(cum, pop));
  } else {
    const kEff = Math.min(cfg.k, n);
    selected = Array.from({length: n}, () => tournamentSelect(pop, fitnesses, kEff));
  }

  for (let i = n-1; i > 0; i--) {
    const j = Math.floor(Math.random()*(i+1));
    [selected[i], selected[j]] = [selected[j], selected[i]];
  }

  let numCross = Math.floor(n * cfg.crossProb);
  if (numCross % 2 !== 0) numCross = Math.max(0, numCross-1);
  const toCross = selected.slice(0, numCross);
  const noChange = selected.slice(numCross);

  const filhos = [];
  for (let i = 0; i+1 < toCross.length; i += 2) {
    let c1, c2;
    if      (cfg.crossType === 'aritm') [c1,c2] = arithmeticCross(toCross[i], toCross[i+1]);
    else if (cfg.crossType === 'blx')   [c1,c2] = blxCross(toCross[i], toCross[i+1]);
    else                                [c1,c2] = sbxCross(toCross[i], toCross[i+1]);
    filhos.push(c1, c2);
  }

  const mutFn = cfg.mutType === 'gauss'
    ? c => gaussMutate(c, cfg.mutProb, cfg.sigma)
    : c => uniformMutate(c, cfg.mutProb);

  const newPop = [...elites, ...[...noChange, ...filhos].map(mutFn)].slice(0, n);
  while (newPop.length < n) newPop.push(randomChromosome(cfg.dims));
  return newPop;
}

// ── Visualização ─────────────────────────────────────────────────

let histBest = [];

function heatRGB(t) {
  if (t <= 0.5) {
    const s = t*2;
    return [Math.round(6-6*s), Math.round(20+160*s), Math.round(50+90*s)];
  }
  const s = (t-0.5)*2;
  return [Math.round(255*s), Math.round(180-117*s), Math.round(140+15*s)];
}

function drawHeatmap(bestChrom) {
  const canvas = document.getElementById('heatCanvas');
  if (!canvas) return;
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');
  const RES = 80, cw = W/RES, ch = H/RES;
  const FMAX = 80; // máximo aproximado de Rastrigin n=2 em [-5.12,5.12]²

  for (let row = 0; row < RES; row++) {
    for (let col = 0; col < RES; col++) {
      const x1 = X_MIN + (col+0.5)/RES*(X_MAX-X_MIN);
      const x2 = X_MIN + (row+0.5)/RES*(X_MAX-X_MIN);
      const t = Math.min(1, rastrigin([x1,x2]) / FMAX);
      const [r,g,b] = heatRGB(t);
      ctx.fillStyle = `rgb(${r},${g},${b})`;
      ctx.fillRect(col*cw, row*ch, cw+0.5, ch+0.5);
    }
  }

  if (bestChrom) {
    const px = ((bestChrom[0]-X_MIN)/(X_MAX-X_MIN)) * W;
    const py = ((bestChrom[1]-X_MIN)/(X_MAX-X_MIN)) * H;
    ctx.fillStyle = 'rgba(255,63,155,0.25)';
    ctx.beginPath(); ctx.arc(px, py, 11, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#ff3f9b';
    ctx.beginPath(); ctx.arc(px, py, 5, 0, Math.PI*2); ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.beginPath(); ctx.arc(px, py, 2, 0, Math.PI*2); ctx.fill();
  }
}

function drawConvergence() {
  const canvas = document.getElementById('convCanvas');
  if (!canvas || histBest.length === 0) return;
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');
  const PAD = {l:48, r:12, t:10, b:26};
  const pw = W-PAD.l-PAD.r, ph = H-PAD.t-PAD.b;

  ctx.clearRect(0,0,W,H); ctx.fillStyle='#081523'; ctx.fillRect(0,0,W,H);

  const maxV = Math.max(...histBest, 1);
  const toX = i => PAD.l + (i/Math.max(1,histBest.length-1))*pw;
  const toY = v => PAD.t + ph - (v/maxV)*ph;

  for (let i = 0; i <= 4; i++) {
    const v = maxV*i/4, y = toY(v);
    ctx.strokeStyle='#0d1f32'; ctx.lineWidth=1;
    ctx.beginPath(); ctx.moveTo(PAD.l,y); ctx.lineTo(W-PAD.r,y); ctx.stroke();
    ctx.fillStyle='#86a8bf'; ctx.font='9px Consolas'; ctx.textAlign='right';
    ctx.fillText(v.toFixed(1), PAD.l-4, y+3);
  }
  ctx.strokeStyle='#1e3f5f'; ctx.lineWidth=1.5;
  ctx.beginPath(); ctx.moveTo(PAD.l,PAD.t); ctx.lineTo(PAD.l,H-PAD.b); ctx.stroke();
  ctx.beginPath(); ctx.moveTo(PAD.l,H-PAD.b); ctx.lineTo(W-PAD.r,H-PAD.b); ctx.stroke();

  ctx.strokeStyle='#ff3f9b'; ctx.lineWidth=1.5; ctx.beginPath();
  histBest.forEach((v,i) => i===0 ? ctx.moveTo(toX(i),toY(v)) : ctx.lineTo(toX(i),toY(v)));
  ctx.stroke();

  ctx.fillStyle='#86a8bf'; ctx.font='9px Consolas'; ctx.textAlign='center';
  for (let i = 0; i <= 4; i++) {
    const idx = Math.round(i*(histBest.length-1)/4);
    ctx.fillText(idx, toX(idx), H-PAD.b+10);
  }
}

// ── UI helpers ────────────────────────────────────────────────────

function addLog(msg) {
  const box = document.getElementById('logBox');
  box.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`;
  box.scrollTop = box.scrollHeight;
}

function updateMetrics(gen, best, cfg) {
  document.getElementById('mGen').textContent   = gen;
  document.getElementById('mFit').textContent   = best.f.toFixed(6);
  document.getElementById('mSel').textContent   = cfg.selMethod==='roleta' ? 'Roleta' : `Torneio (k=${cfg.k})`;
  document.getElementById('mCross').textContent = {aritm:'Aritmético', blx:'BLX-α', sbx:'SBX'}[cfg.crossType];
  document.getElementById('mMut').textContent   = cfg.mutType==='gauss' ? `Gaussiana (σ=${cfg.sigma})` : 'Uniforme';
  document.getElementById('mElite').textContent = cfg.elitism>0 ? `${cfg.elitism} ind.` : 'Desativado';
  document.getElementById('chromDisplay').innerHTML =
    best.chrom.map((v,i) => `<span class="gene"><span class="gi">x${i+1}</span>${v.toFixed(4)}</span>`).join('');
  document.getElementById('graphLabel').textContent = `f* = ${best.f.toFixed(6)}`;
}

function resetUI() {
  document.getElementById('mGen').textContent = '0';
  ['mFit','mSel','mCross','mMut','mElite'].forEach(id => document.getElementById(id).textContent = '—');
  document.getElementById('chromDisplay').textContent = '—';
  document.getElementById('graphLabel').textContent = '—';
  document.getElementById('logBox').textContent = 'Aguardando execução...';
  histBest = [];
  const cc = document.getElementById('convCanvas');
  if (cc) { const ctx = cc.getContext('2d'); ctx.fillStyle='#081523'; ctx.fillRect(0,0,cc.width,cc.height); }
  drawHeatmap(null);
}

function readCfg() {
  return {
    dims:      Math.min(10, Math.max(1, parseInt(document.getElementById('dims').value) || 2)),
    popSize:   Math.max(4, parseInt(document.getElementById('popSize').value) || 60),
    gens:      Math.max(1, parseInt(document.getElementById('generations').value) || 300),
    crossProb: parseInt(document.getElementById('crossSlider').value)/100,
    mutProb:   parseInt(document.getElementById('mutSlider').value)/100,
    sigma:     parseFloat(document.getElementById('sigma').value) || 0.5,
    elitism:   Math.max(0, parseInt(document.getElementById('elitism').value) || 2),
    selMethod: document.querySelector('input[name="selMethod"]:checked').value,
    k:         Math.max(2, parseInt(document.getElementById('kTorneio').value) || 3),
    crossType: document.querySelector('input[name="crossType"]:checked').value,
    mutType:   document.querySelector('input[name="mutType"]:checked').value
  };
}

// ── Loop principal ────────────────────────────────────────────────

let running = false;
let overallBest = null;
let pop = [];

async function runAG() {
  if (running) return;
  running = true;
  document.getElementById('runBtn').disabled = true;
  document.getElementById('logBox').textContent = '';
  histBest = [];

  const cfg = readCfg();
  const crossLbl = {aritm:'Aritmético', blx:'BLX-α', sbx:'SBX'}[cfg.crossType];
  const selLbl   = cfg.selMethod==='roleta' ? 'Roleta' : `Torneio(k=${cfg.k})`;
  addLog(`Pop:${cfg.popSize} | Ger:${cfg.gens} | Dims:${cfg.dims} | Cruz:${(cfg.crossProb*100).toFixed(0)}% | Mut:${(cfg.mutProb*100).toFixed(0)}%`);
  addLog(`Seleção:${selLbl} | Cross:${crossLbl} | Mut:${cfg.mutType==='gauss'?`Gauss(σ=${cfg.sigma})`:'Unif'} | Elite:${cfg.elitism}`);
  addLog(`Min global: f(0,…,0) = 0`);

  document.getElementById('heatWrap').style.display = cfg.dims===2 ? 'block' : 'none';

  pop = Array.from({length: cfg.popSize}, () => randomChromosome(cfg.dims));
  overallBest = null;

  for (let gen = 1; gen <= cfg.gens; gen++) {
    if (!running) break;

    const fitnesses = pop.map(c => rastrigin(c));
    const minIdx = fitnesses.indexOf(Math.min(...fitnesses));
    const curr = {chrom: pop[minIdx], f: fitnesses[minIdx]};
    if (!overallBest || curr.f < overallBest.f)
      overallBest = {chrom: [...curr.chrom], f: curr.f};

    histBest.push(overallBest.f);
    pop = runGeneration(pop, fitnesses, cfg);

    updateMetrics(gen, overallBest, cfg);
    if (cfg.dims === 2) drawHeatmap(overallBest.chrom);
    drawConvergence();

    if (gen===1 || gen%20===0 || gen===cfg.gens)
      addLog(`Gen ${String(gen).padStart(4)}: f* = ${overallBest.f.toFixed(6)}`);
    if (gen%5===0) await new Promise(r => setTimeout(r, 0));
  }

  if (running) {
    addLog('─── Concluído ───');
    addLog(`Melhor f* = ${overallBest.f.toFixed(8)}`);
    addLog(`x* = [${overallBest.chrom.map(v=>v.toFixed(5)).join(', ')}]`);
  }
  running = false;
  document.getElementById('runBtn').disabled = false;
}

// ── Inicialização ─────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  drawHeatmap(null);
  resetUI();

  document.getElementById('crossSlider').addEventListener('input', function() {
    document.getElementById('crossVal').textContent = this.value + '%';
  });
  document.getElementById('mutSlider').addEventListener('input', function() {
    document.getElementById('mutVal').textContent = this.value + '%';
  });

  document.querySelectorAll('input[name="selMethod"]').forEach(r =>
    r.addEventListener('change', () => {
      document.getElementById('torneioCfg').style.display =
        document.getElementById('selTorneio').checked ? 'flex' : 'none';
    })
  );

  document.querySelectorAll('input[name="mutType"]').forEach(r =>
    r.addEventListener('change', () => {
      document.getElementById('sigmaCfg').style.display =
        document.getElementById('mutGauss').checked ? 'flex' : 'none';
    })
  );

  document.getElementById('dims').addEventListener('change', function() {
    const is2 = parseInt(this.value)===2;
    document.getElementById('heatWrap').style.display = is2 ? 'block' : 'none';
    if (is2) drawHeatmap(overallBest ? overallBest.chrom : null);
  });

  document.getElementById('runBtn').addEventListener('click', runAG);
  document.getElementById('resetBtn').addEventListener('click', () => {
    if (!running) { pop=[]; overallBest=null; resetUI(); }
  });
});
