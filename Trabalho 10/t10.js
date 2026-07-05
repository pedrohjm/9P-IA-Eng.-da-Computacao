// ── Data ──────────────────────────────────────────────────────────
const ALL_NAMES = [
  'Heloísa','Quintino','Custódio','Belátrio','Iracema','Patriano','Adauto',
  'Terciado','Alfácio','Sebastiana','Alaíde','Farinéia','Marcelus','Quíntia',
  'Gertrudes','Salomão','Bendita','Reginaldo','Flaviana','Tiburcio'
];
const ALL_SUBJS = [
  'Mat','Por','Fís','Quí','Bio','Geo','His','Ing','Art',
  'Fil','Soc','Edf','Inf','Esp','Mus','Eco','Cie','Red','Psic','Adm'
];
const DIA_LABELS = ['Seg','Ter','Qua','Qui','Sex','Sáb'];

const FITNESS_DEFS = {
  default: {
    cw:1, lw:18, uw:1,
    lines:[
      {c:'pos', t:'+1 por aula encadeada (mesma matéria 2+ consecutivas)'},
      {c:'neg', t:'−18 por choque (prof em 2 turmas no mesmo slot)'},
      {c:'neg', t:'−1 por matéria não-coberta numa turma'}
    ]
  },
  anticlash: {
    cw:0.5, lw:30, uw:2,
    lines:[
      {c:'pos', t:'+0.5 por aula encadeada'},
      {c:'neg', t:'−30 por choque (penalização severa)'},
      {c:'neg', t:'−2 por matéria não-coberta numa turma'}
    ]
  },
  coverage: {
    cw:1, lw:8, uw:3,
    lines:[
      {c:'pos', t:'+1 por aula encadeada'},
      {c:'neg', t:'−8 por choque (tolerância moderada)'},
      {c:'neg', t:'−3 por matéria não-coberta numa turma'}
    ]
  }
};

// ── State ─────────────────────────────────────────────────────────
let PROFS        = [];
let population   = [];
let bestInd      = null;
let currentGen   = 0;
let showNames    = true;
let running      = false;
let rafId        = null;

// ── Helpers ───────────────────────────────────────────────────────
function cfg() {
  return {
    numProfs:    +document.getElementById('numProfs').value,
    numTurmas:   +document.getElementById('numTurmas').value,
    aulasPorDia: +document.getElementById('aulasPorDia').value,
    numDias:     +document.getElementById('numDias').value,
    popSize:     +document.getElementById('popSize').value,
    maxGens:     +document.getElementById('maxGens').value,
    Pc:          +document.getElementById('Pc').value,
    Pm:          +document.getElementById('Pm').value,
    elitism:     +document.getElementById('elitism').value,
    mode:         document.getElementById('fitnessMode').value
  };
}

function totalSlots(c) { return c.aulasPorDia * c.numDias; }

// ── Professors ────────────────────────────────────────────────────
function generateProfs(n) {
  const names = [...ALL_NAMES].sort(()=>Math.random()-.5).slice(0,n);
  const subjs = [...ALL_SUBJS].sort(()=>Math.random()-.5).slice(0,n);
  return names.map((name,i)=>({name, subj:subjs[i]}));
}

function profBg(i)  { const h=(i*43+30)%360; return `hsl(${h},44%,17%)`; }
function profFg(i)  { const h=(i*43+30)%360; return `hsl(${h},70%,80%)`; }
function turmaHue(t){ return (t*55+140)%360; }

// ── Matrix operations ─────────────────────────────────────────────
function randomMatrix(c) {
  const s = totalSlots(c);
  return Array.from({length:s}, ()=>
    Array.from({length:c.numTurmas}, ()=>Math.floor(Math.random()*c.numProfs))
  );
}

function copyMatrix(m) { return m.map(r=>[...r]); }

// ── Fitness ───────────────────────────────────────────────────────
function evaluate(matrix, c) {
  const s = totalSlots(c);
  const fd = FITNESS_DEFS[c.mode];
  let chains=0, clashes=0, uncov=0;

  // Chained: same prof consecutive in same turma (not across day boundary)
  for (let t=0; t<c.numTurmas; t++) {
    for (let s2=0; s2<s-1; s2++) {
      if ((s2+1) % c.aulasPorDia === 0) continue;
      if (matrix[s2][t] === matrix[s2+1][t]) chains++;
    }
  }

  // Clashes: same prof in 2+ turmas at same slot
  for (let s2=0; s2<s; s2++) {
    const cnt = new Uint16Array(c.numProfs);
    for (let t=0; t<c.numTurmas; t++) cnt[matrix[s2][t]]++;
    for (let p=0; p<c.numProfs; p++) if (cnt[p]>1) clashes+=(cnt[p]-1);
  }

  // Uncovered: prof not present in turma at all
  for (let t=0; t<c.numTurmas; t++) {
    const seen = new Uint8Array(c.numProfs);
    for (let s2=0; s2<s; s2++) seen[matrix[s2][t]]=1;
    for (let p=0; p<c.numProfs; p++) if (!seen[p]) uncov++;
  }

  const score = chains*fd.cw - clashes*fd.lw - uncov*fd.uw;
  return {score, chains, clashes, uncov};
}

// ── GA operators ──────────────────────────────────────────────────
function crossover(m1, m2, c) {
  if (Math.random() > c.Pc) return [copyMatrix(m1), copyMatrix(m2)];
  const s = totalSlots(c);
  const pt = Math.floor(Math.random()*(s-1))+1;
  return [
    [...m1.slice(0,pt).map(r=>[...r]), ...m2.slice(pt).map(r=>[...r])],
    [...m2.slice(0,pt).map(r=>[...r]), ...m1.slice(pt).map(r=>[...r])]
  ];
}

function mutate(matrix, c) {
  return matrix.map(row =>
    row.map(cell => Math.random()<c.Pm ? Math.floor(Math.random()*c.numProfs) : cell)
  );
}

function tournamentSelect(pop) {
  let best = pop[Math.floor(Math.random()*pop.length)];
  for (let i=1; i<4; i++) {
    const cand = pop[Math.floor(Math.random()*pop.length)];
    if (cand.score > best.score) best = cand;
  }
  return best;
}

// ── One generation ────────────────────────────────────────────────
function runGen(c) {
  const n = population.length;

  // Evaluate all
  for (const ind of population) Object.assign(ind, evaluate(ind.matrix, c));

  // Sort: highest score first (maximization)
  population.sort((a,b)=>b.score-a.score);
  bestInd = population[0];

  // Elites
  const newPop = population.slice(0, c.elitism).map(ind=>({
    matrix: copyMatrix(ind.matrix), score:ind.score,
    chains:ind.chains, clashes:ind.clashes, uncov:ind.uncov
  }));

  // Children
  while (newPop.length < n) {
    const p1 = tournamentSelect(population);
    const p2 = tournamentSelect(population);
    const [c1,c2] = crossover(p1.matrix, p2.matrix, c);
    newPop.push({matrix: mutate(c1, c)});
    if (newPop.length < n) newPop.push({matrix: mutate(c2, c)});
  }

  population = newPop;
  currentGen++;
}

// ── Evolution loop ────────────────────────────────────────────────
const BATCH = 8;

function tick() {
  if (!running) return;
  const c = cfg();
  for (let i=0; i<BATCH && currentGen<c.maxGens; i++) runGen(c);
  updateStats();
  renderMatrix(c);
  if (currentGen >= c.maxGens) { stop(); return; }
  rafId = requestAnimationFrame(tick);
}

function start() {
  if (running) return;
  const c = cfg();
  if (!population.length) {
    PROFS = generateProfs(c.numProfs);
    population = Array.from({length:c.popSize}, ()=>({matrix:randomMatrix(c)}));
    currentGen = 0;
  }
  running = true;
  const btn = document.getElementById('btnRun');
  btn.textContent = '⏸ Pausar';
  btn.className = 'hdr-btn stop';
  rafId = requestAnimationFrame(tick);
}

function stop() {
  running = false;
  if (rafId) { cancelAnimationFrame(rafId); rafId=null; }
  const btn = document.getElementById('btnRun');
  btn.textContent = 'Evoluir';
  btn.className = 'hdr-btn ok';
}

function reset() {
  stop();
  population=[]; bestInd=null; currentGen=0;
  document.getElementById('statGen').textContent   = '0';
  document.getElementById('statFit').textContent   = '—';
  document.getElementById('statFitSub').textContent= '—';
  document.getElementById('statClash').textContent = '—';
  document.getElementById('schedTable').innerHTML  =
    '<tbody><tr><td><div class="empty-msg">Inicialize a população clicando em <strong style="color:var(--ok)">EVOLUIR</strong></div></td></tr></tbody>';
}

// ── UI ────────────────────────────────────────────────────────────
function refreshSlotInfo() {
  const c = cfg();
  const s = totalSlots(c);
  const chromo = s * c.numTurmas;
  document.getElementById('infoSlots').textContent  = s;
  document.getElementById('infoChromo').textContent = `${s} × ${c.numTurmas} = ${chromo}`;
}

function refreshFitDesc() {
  const mode = document.getElementById('fitnessMode').value;
  document.getElementById('fitDesc').innerHTML =
    FITNESS_DEFS[mode].lines.map(l=>`<div class="${l.c}">${l.t}</div>`).join('');
}

function updateStats() {
  if (!bestInd || bestInd.score===undefined) return;
  document.getElementById('statGen').textContent   = currentGen;
  document.getElementById('statFit').textContent   = bestInd.score.toFixed(1);
  document.getElementById('statFitSub').textContent= `encadeadas: ${bestInd.chains}`;
  document.getElementById('statClash').textContent = bestInd.clashes;
}

// ── Matrix render ─────────────────────────────────────────────────
function renderMatrix(c) {
  if (!bestInd) return;
  const mat = bestInd.matrix;
  const s   = totalSlots(c);

  // Pre-compute clashing slots
  const clashSet = new Set();
  for (let s2=0; s2<s; s2++) {
    const cnt = new Uint16Array(c.numProfs);
    for (let t=0; t<c.numTurmas; t++) cnt[mat[s2][t]]++;
    for (let p=0; p<c.numProfs; p++) if (cnt[p]>1) clashSet.add(s2);
  }

  // Header
  let html = '<thead><tr>';
  html += `<th colspan="2" style="min-width:54px;">Dia / #</th>`;
  for (let t=0; t<c.numTurmas; t++) {
    const h = turmaHue(t);
    html += `<th style="color:hsl(${h},70%,65%);border-top:3px solid hsl(${h},55%,45%)">` +
            `Turma ${String(t+1).padStart(2,'0')}</th>`;
  }
  html += '</tr></thead><tbody>';

  for (let s2=0; s2<s; s2++) {
    const dia   = Math.floor(s2 / c.aulasPorDia);
    const turno = s2 % c.aulasPorDia;
    const clash = clashSet.has(s2);

    html += '<tr>';
    if (turno===0) {
      html += `<td class="dia-lbl" rowspan="${c.aulasPorDia}">${DIA_LABELS[dia]||'D'+(dia+1)}</td>`;
    }
    html += `<td class="turno-lbl">${turno+1}</td>`;

    for (let t=0; t<c.numTurmas; t++) {
      const p    = mat[s2][t];
      const prof = PROFS[p] || {name:'?',subj:'?'};
      const bg   = profBg(p);
      const fg   = profFg(p);
      html += `<td class="${clash?'clash-cell':''}">` +
              `<div class="scell" style="background:${bg};color:${fg}">` +
              (showNames ? `<span class="scell-name">${prof.name}</span>` : '') +
              `<span class="scell-subj">${prof.subj}</span>` +
              `</div></td>`;
    }
    html += '</tr>';
  }

  html += '</tbody>';
  document.getElementById('schedTable').innerHTML = html;
}

// ── Boot ──────────────────────────────────────────────────────────
document.addEventListener('DOMContentLoaded', ()=>{
  refreshSlotInfo();
  refreshFitDesc();

  ['numProfs','numTurmas','aulasPorDia','numDias']
    .forEach(id => document.getElementById(id).addEventListener('change', refreshSlotInfo));

  document.getElementById('fitnessMode').addEventListener('change', refreshFitDesc);

  document.getElementById('btnRun').addEventListener('click', ()=>{
    if (running) stop(); else start();
  });

  document.getElementById('btnReset').addEventListener('click', reset);

  document.getElementById('btnNomes').addEventListener('click', ()=>{
    showNames = !showNames;
    document.getElementById('btnNomes').textContent = showNames ? '― Nomes' : '+ Nomes';
    if (bestInd) renderMatrix(cfg());
  });
});
