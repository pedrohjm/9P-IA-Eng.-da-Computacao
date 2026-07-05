// ── Trabalho 14 — AG com Ranking · TSP (Triângulo Mineiro) ────

const CITIES = [
  {name:'Uberlândia',    short:'Uberlândia', x:336, y: 88},
  {name:'Uberaba',       short:'Uberaba ★',  x:417, y:179},
  {name:'Araguari',      short:'Araguari',   x:357, y: 58},
  {name:'Ituiutaba',     short:'Ituiutaba',  x: 60, y: 94},
  {name:'Patos de Minas',short:'P.Minas',    x:745, y: 50},
  {name:'Frutal',        short:'Frutal',     x:184, y:210},
  {name:'Araxá',         short:'Araxá',      x:647, y:162},
  {name:'Monte Carmelo', short:'M.Carmelo',  x:518, y: 67},
  {name:'Tupaciguara',   short:'Tupacig.',   x:237, y: 52},
  {name:'Campina Verde', short:'C.Verde',    x: 55, y:156},
];

const CITY_W = 800, CITY_H = 260;
const START = 1;

const DIST = [
//        0    1    2    3    4    5    6    7    8    9
/* 0 */ [  0, 106,  30, 138, 190, 211, 175,  89,  56, 266],
/* 1 */ [106,   0, 136, 244, 265, 105, 110, 195, 162, 160],
/* 2 */ [ 30, 136,   0, 117, 221, 241, 205,  66,  47, 296],
/* 3 */ [138, 244, 117,   0, 277, 265, 308, 163, 164, 186],
/* 4 */ [190, 265, 221, 277,   0, 322, 137, 114, 225, 397],
/* 5 */ [211, 105, 241, 265, 322,   0, 185, 300, 267,  75],
/* 6 */ [175, 110, 205, 308, 137, 185,   0, 145, 231, 260],
/* 7 */ [ 89, 195,  66, 163, 114, 300, 145,   0, 111, 349],
/* 8 */ [ 56, 162,  47, 164, 225, 267, 231, 111,   0, 322],
/* 9 */ [266, 160, 296, 186, 397,  75, 260, 349, 322,   0],
];

// ── Funções básicas ────────────────────────────────────────

function tourLength(perm) {
  let d = 0;
  for (let i = 0; i < perm.length; i++)
    d += DIST[perm[i]][perm[(i + 1) % perm.length]];
  return d;
}

function randomPerm() {
  const others = Array.from({length: CITIES.length}, (_, i) => i).filter(i => i !== START);
  for (let i = others.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [others[i], others[j]] = [others[j], others[i]];
  }
  return [START, ...others];
}

function findBest(pop) {
  let best = null;
  for (const p of pop) {
    const d = tourLength(p);
    if (!best || d < best.dist) best = {perm: [...p], dist: d};
  }
  return best;
}

// ── Seleção por Ranking ────────────────────────────────────
// pop ordenada por fitness; rank 1 = pior, rank N = melhor
// Linear:      P(i) = (2 - sp)/N + 2*(sp-1)*(i-1) / (N*(N-1))   sp ∈ (1,2]
// Exponencial: P(i) ∝ 1 - exp(-i)   normalizado

function buildRankingCumulative(pop, dists, method, sp) {
  const n = pop.length;
  // índices ordenados: pior primeiro (maior dist = rank 1), melhor último (rank N)
  const order = Array.from({length: n}, (_, i) => i)
    .sort((a, b) => dists[b] - dists[a]);

  const weights = new Array(n);
  for (let k = 0; k < n; k++) {
    const rank = k + 1; // rank 1 = pior, rank N = melhor
    if (method === 'linear') {
      weights[order[k]] = (2 - sp) / n + 2 * (sp - 1) * (rank - 1) / (n * (n - 1));
    } else {
      // exponencial: 1 - e^(-rank), rank cresce com melhor fitness
      weights[order[k]] = 1 - Math.exp(-rank);
    }
  }

  const total = weights.reduce((a, b) => a + b, 0);
  let cum = 0;
  const cumArr = weights.map(w => (cum += w / total));
  return {cumArr, order};
}

function rankingSelect(cumArr, pop) {
  const r = Math.random();
  const idx = cumArr.findIndex(v => r <= v);
  return [...pop[idx < 0 ? pop.length - 1 : idx]];
}

// ── Crossover OX (Order Crossover) ────────────────────────

function oxCrossover(p1, p2) {
  const tLen = p1.length - 1;
  const t1 = p1.slice(1), t2 = p2.slice(1);
  const pt1 = Math.floor(Math.random() * (tLen - 1));
  const pt2 = pt1 + 1 + Math.floor(Math.random() * (tLen - 1 - pt1));

  function makeChild(par1, par2) {
    const child = new Array(tLen).fill(-1);
    const inSeg = new Set();
    for (let i = pt1; i <= pt2; i++) { child[i] = par1[i]; inSeg.add(par1[i]); }
    const remaining = [];
    for (let k = 1; k <= tLen; k++) {
      const c = par2[(pt2 + k) % tLen];
      if (!inSeg.has(c)) remaining.push(c);
    }
    let ri = 0;
    for (let k = 1; k <= tLen; k++) {
      const pos = (pt2 + k) % tLen;
      if (child[pos] === -1) child[pos] = remaining[ri++];
    }
    return [START, ...child];
  }

  return [makeChild(t1, t2), makeChild(t2, t1)];
}

// ── Mutação por troca (swap) ───────────────────────────────

function swapMutate(perm, mutProb) {
  const p = [...perm];
  if (Math.random() < mutProb) {
    const i = 1 + Math.floor(Math.random() * (p.length - 1));
    let j;
    do { j = 1 + Math.floor(Math.random() * (p.length - 1)); } while (j === i);
    [p[i], p[j]] = [p[j], p[i]];
  }
  return p;
}

// ── Geração ────────────────────────────────────────────────

function runGeneration(pop, crossProb, mutProb, rankMethod, sp, elitism) {
  const n = pop.length;
  const dists = pop.map(p => tourLength(p));

  // elitismo
  const elite = [];
  if (elitism > 0) {
    const sorted = pop.map((p, i) => ({p, d: dists[i]})).sort((a, b) => a.d - b.d);
    for (let i = 0; i < Math.min(elitism, n); i++) elite.push([...sorted[i].p]);
  }

  const {cumArr} = buildRankingCumulative(pop, dists, rankMethod, sp);
  const selected = Array.from({length: n}, () => rankingSelect(cumArr, pop));

  // embaralha selecionados para parear aleatoriamente
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [selected[i], selected[j]] = [selected[j], selected[i]];
  }

  let numCross = Math.floor(n * crossProb);
  if (numCross % 2 !== 0) numCross = Math.max(0, numCross - 1);
  const toCross  = selected.slice(0, numCross);
  const noChange = selected.slice(numCross);

  const filhos = [];
  for (let i = 0; i + 1 < toCross.length; i += 2) {
    const [c1, c2] = oxCrossover(toCross[i], toCross[i + 1]);
    filhos.push(c1, c2);
  }

  let nextPop = [...noChange, ...filhos.map(f => swapMutate(f, mutProb))];

  // substitui os piores pelos elites
  if (elite.length > 0) {
    const nextDists = nextPop.map(p => tourLength(p));
    const worstIdx = nextDists
      .map((d, i) => ({d, i}))
      .sort((a, b) => b.d - a.d)
      .slice(0, elite.length)
      .map(x => x.i);
    worstIdx.forEach((idx, k) => { nextPop[idx] = elite[k]; });
  }

  return nextPop;
}

// ── Gráfico de convergência ────────────────────────────────

let convHistory = [];

function drawConvergence() {
  const canvas = document.getElementById('convCanvas');
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');
  const PAD = {l: 42, r: 14, t: 14, b: 28};
  const pw = W - PAD.l - PAD.r, ph = H - PAD.t - PAD.b;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#081523'; ctx.fillRect(0, 0, W, H);

  if (convHistory.length < 2) return;

  const minD = Math.min(...convHistory);
  const maxD = Math.max(...convHistory);
  const range = maxD - minD || 1;

  ctx.strokeStyle = '#0d2035'; ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = PAD.t + i * (ph / 4);
    ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(W - PAD.r, y); ctx.stroke();
    const val = maxD - i * (range / 4);
    ctx.fillStyle = '#455f74'; ctx.font = '9px Consolas'; ctx.textAlign = 'right';
    ctx.fillText(val.toFixed(0), PAD.l - 4, y + 3);
  }

  ctx.strokeStyle = '#ff3f9b'; ctx.lineWidth = 1.6; ctx.setLineDash([]);
  ctx.beginPath();
  convHistory.forEach((d, i) => {
    const x = PAD.l + (i / (convHistory.length - 1)) * pw;
    const y = PAD.t + ph - ((d - minD) / range) * ph;
    i === 0 ? ctx.moveTo(x, y) : ctx.lineTo(x, y);
  });
  ctx.stroke();
}

// ── Mapa ───────────────────────────────────────────────────

function drawMap(bestPerm) {
  const canvas = document.getElementById('mapCanvas');
  const W = canvas.width, H = canvas.height;
  const ctx = canvas.getContext('2d');
  const PAD = {l: 30, r: 20, t: 24, b: 20};
  const pw = W - PAD.l - PAD.r, ph = H - PAD.t - PAD.b;
  const toX = x => PAD.l + (x / CITY_W) * pw;
  const toY = y => PAD.t + (y / CITY_H) * ph;

  ctx.clearRect(0, 0, W, H);
  ctx.fillStyle = '#081523'; ctx.fillRect(0, 0, W, H);

  // grade
  ctx.strokeStyle = '#0d1f32'; ctx.lineWidth = 1;
  for (let i = 0; i <= 8; i++) {
    const x = PAD.l + i * (pw / 8);
    ctx.beginPath(); ctx.moveTo(x, PAD.t); ctx.lineTo(x, H - PAD.b); ctx.stroke();
  }
  for (let i = 0; i <= 4; i++) {
    const y = PAD.t + i * (ph / 4);
    ctx.beginPath(); ctx.moveTo(PAD.l, y); ctx.lineTo(W - PAD.r, y); ctx.stroke();
  }

  // rota
  if (bestPerm) {
    ctx.strokeStyle = '#ff3f9b'; ctx.lineWidth = 1.8; ctx.setLineDash([]);
    ctx.beginPath();
    ctx.moveTo(toX(CITIES[bestPerm[0]].x), toY(CITIES[bestPerm[0]].y));
    for (let i = 1; i < bestPerm.length; i++)
      ctx.lineTo(toX(CITIES[bestPerm[i]].x), toY(CITIES[bestPerm[i]].y));
    ctx.closePath(); ctx.stroke();
  }

  // cidades
  CITIES.forEach((c, i) => {
    const cx = toX(c.x), cy = toY(c.y);
    const isStart = (i === START);

    ctx.fillStyle = isStart ? 'rgba(255,200,0,0.18)' : 'rgba(0,211,159,0.12)';
    ctx.beginPath(); ctx.arc(cx, cy, 13, 0, Math.PI * 2); ctx.fill();

    ctx.fillStyle = isStart ? '#ffc800' : '#00d39f';
    ctx.beginPath(); ctx.arc(cx, cy, 8, 0, Math.PI * 2); ctx.fill();

    ctx.fillStyle = '#061422';
    ctx.font = 'bold 9px Consolas';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(i, cx, cy);

    const labelY = cy < 75 ? cy + 18 : cy - 14;
    ctx.fillStyle = isStart ? '#ffc800' : '#86a8bf';
    ctx.font = '7px Consolas';
    ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
    ctx.fillText(c.short, cx, labelY);
  });
}

// ── UI helpers ─────────────────────────────────────────────

function addLog(msg) {
  const box = document.getElementById('logBox');
  box.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`;
  box.scrollTop = box.scrollHeight;
}

function updateMetrics(gen, best, rankMethod, sp) {
  document.getElementById('mGen').textContent   = gen;
  document.getElementById('mDist').textContent  = best.dist.toFixed(0) + ' km';
  document.getElementById('mRank').textContent  = rankMethod === 'linear' ? `Linear  sp=${sp}` : 'Exponencial';
  document.getElementById('graphLabel').textContent = `dist* = ${best.dist.toFixed(0)} km`;
  document.getElementById('routeDisplay').textContent =
    best.perm.map(i => CITIES[i].name).join(' → ') + ' → ' + CITIES[START].name;
}

function updateCityGrid() {
  document.getElementById('cityGrid').innerHTML = CITIES.map((c, i) =>
    `<div class="city-item${i === START ? ' start' : ''}">
      <span class="ci">${i}</span>${i === START ? ' ★' : ''} ${c.name}
    </div>`
  ).join('');
}

function resetUI() {
  ['mDist', 'mRank'].forEach(id => document.getElementById(id).textContent = '—');
  document.getElementById('mGen').textContent = '0';
  document.getElementById('routeDisplay').textContent = '—';
  document.getElementById('graphLabel').textContent = '—';
  document.getElementById('logBox').textContent = 'Aguardando execução...';
  convHistory = [];
  drawMap(null);
  drawConvergence();
}

// ── Loop principal ─────────────────────────────────────────

let running = false;
let overallBest = null;
let pop = [];

async function runAG() {
  if (running) return;
  running = true;
  document.getElementById('runBtn').disabled = true;
  document.getElementById('logBox').textContent = '';
  convHistory = [];

  const popSize    = Math.max(4, parseInt(document.getElementById('popSize').value)    || 60);
  const gens       = Math.max(1, parseInt(document.getElementById('generations').value) || 300);
  const crossProb  = parseInt(document.getElementById('crossSlider').value) / 100;
  const mutProb    = parseInt(document.getElementById('mutSlider').value)   / 100;
  const elitism    = Math.max(0, parseInt(document.getElementById('elitism').value)    || 0);
  const rankMethod = document.querySelector('input[name="rankMethod"]:checked').value;
  const sp         = parseFloat(document.getElementById('spVal').value) || 1.5;

  addLog(`Pop: ${popSize} | Ger: ${gens} | Cruz: ${(crossProb*100).toFixed(0)}% | Mut: ${(mutProb*100).toFixed(0)}%`);
  addLog(`Ranking: ${rankMethod === 'linear' ? 'Linear  sp='+sp : 'Exponencial'} | Elitismo: ${elitism}`);
  addLog(`Partida/retorno: ${CITIES[START].name} · 9 cidades a visitar`);

  pop = Array.from({length: popSize}, () => randomPerm());
  overallBest = null;

  for (let gen = 1; gen <= gens; gen++) {
    if (!running) break;
    pop = runGeneration(pop, crossProb, mutProb, rankMethod, sp, elitism);
    const best = findBest(pop);
    if (!overallBest || best.dist < overallBest.dist) overallBest = best;
    convHistory.push(overallBest.dist);

    updateMetrics(gen, overallBest, rankMethod, sp);
    drawMap(overallBest.perm);
    drawConvergence();

    if (gen === 1 || gen % 10 === 0 || gen === gens)
      addLog(`Gen ${String(gen).padStart(4)}: dist=${overallBest.dist.toFixed(0)} km`);
    if (gen % 5 === 0) await new Promise(r => setTimeout(r, 0));
  }

  if (running) {
    addLog('─── Concluído ───');
    addLog(`Melhor rota: ${overallBest.dist.toFixed(0)} km`);
    addLog(overallBest.perm.map(i => CITIES[i].name).join(' → '));
    addLog('↩ ' + CITIES[START].name);
  }

  running = false;
  document.getElementById('runBtn').disabled = false;
}

// ── Init ───────────────────────────────────────────────────

document.addEventListener('DOMContentLoaded', () => {
  updateCityGrid();
  drawMap(null);
  drawConvergence();

  // mostrar/esconder pressão de seleção conforme método
  function toggleSP() {
    const isLinear = document.querySelector('input[name="rankMethod"]:checked').value === 'linear';
    document.getElementById('spCfg').style.display = isLinear ? 'flex' : 'none';
  }
  document.querySelectorAll('input[name="rankMethod"]').forEach(r => r.addEventListener('change', toggleSP));
  toggleSP();

  document.getElementById('crossSlider').addEventListener('input', function() {
    document.getElementById('crossVal').textContent = this.value + '%';
  });
  document.getElementById('mutSlider').addEventListener('input', function() {
    document.getElementById('mutVal').textContent = this.value + '%';
  });
  document.getElementById('spVal').addEventListener('input', function() {
    document.getElementById('spDisplay').textContent = parseFloat(this.value).toFixed(2);
  });

  document.getElementById('runBtn').addEventListener('click', runAG);
  document.getElementById('resetBtn').addEventListener('click', () => {
    if (!running) { running = false; pop = []; overallBest = null; resetUI(); }
  });
});
