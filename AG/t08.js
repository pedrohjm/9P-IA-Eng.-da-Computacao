// ── Trabalho 08 — AG Função Matemática ────────────────────────
const T08 = (() => {
  'use strict';
  const $ = id => document.getElementById('a08-' + id);

  const X_MIN = 0, X_MAX = 512;
  let gFmin = 0;
  let running = false;
  let overallBest = null;
  let currentGen = 0;
  let pop = [];

  function f(x) { return -(Math.abs(x * Math.sin(Math.sqrt(Math.abs(x))))); }

  function decode(bits) {
    let val = 0;
    for (let i = 0; i < bits.length; i++) val = val * 2 + bits[i];
    return X_MIN + val * (X_MAX - X_MIN) / (Math.pow(2, bits.length) - 1);
  }

  function randomChromosome(n) {
    return Array.from({length: n}, () => Math.random() < 0.5 ? 1 : 0);
  }

  function gerarImagem(p) { return p.map(c => f(decode(c))); }

  function gerarProbabilidades(img) {
    const maxF = Math.max(...img);
    const shifted = img.map(v => maxF - v + 1e-9);
    const total = shifted.reduce((a, b) => a + b, 0);
    let cum = 0;
    return shifted.map(s => (cum += s / total));
  }

  function rouletteSelect(cum, p) {
    const r = Math.random();
    let idx = cum.findIndex(v => r <= v);
    return [...p[idx < 0 ? p.length - 1 : idx]];
  }

  function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  function cruzamento1pt(p1, p2) {
    const pt = Math.floor(Math.random() * (p1.length - 1)) + 1;
    return [[...p1.slice(0, pt), ...p2.slice(pt)], [...p2.slice(0, pt), ...p1.slice(pt)]];
  }

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

  function findBest(p) {
    let best = null;
    for (const c of p) {
      const x = decode(c), fx = f(x);
      if (!best || fx < best.fx) best = {x, fx, chrom: c};
    }
    return best;
  }

  function runGeneration(p, crossProb, mutProb) {
    const n = p.length;
    const img = gerarImagem(p);
    const cum = gerarProbabilidades(img);
    const selected = Array.from({length: n}, () => rouletteSelect(cum, p));
    shuffle(selected);
    let numCross = Math.floor(n * crossProb);
    if (numCross % 2 !== 0) numCross = Math.max(0, numCross - 1);
    const toCross = selected.slice(0, numCross);
    const noChange = selected.slice(numCross);
    const filhos = [];
    for (let i = 0; i + 1 < toCross.length; i += 2) {
      filhos.push(...cruzamento1pt(toCross[i], toCross[i + 1]));
    }
    return [...noChange, ...efetuarMutacao(filhos, mutProb)];
  }

  // ── Graph ────────────────────────────────────────────────────
  function precomputeRange() {
    let mn = 0;
    for (let i = 1; i <= 3000; i++) {
      const v = f(X_MIN + i * (X_MAX - X_MIN) / 3000);
      if (v < mn) mn = v;
    }
    gFmin = mn * 1.06;
  }

  function drawGraph(bestX) {
    const canvas = $('graphCanvas');
    if (!canvas) return;
    const W = canvas.width, H = canvas.height;
    const ctx = canvas.getContext('2d');
    const PAD = {l:50, r:14, t:12, b:28};
    const pW = W - PAD.l - PAD.r, pH = H - PAD.t - PAD.b;
    ctx.clearRect(0, 0, W, H);
    ctx.fillStyle = '#081523'; ctx.fillRect(0, 0, W, H);
    const toX = x  => PAD.l + (x - X_MIN) / (X_MAX - X_MIN) * pW;
    const toY = fy => PAD.t + (-fy / -gFmin) * pH;
    for (let i = 0; i <= 5; i++) {
      const fy = gFmin * (1 - i / 5), cy = toY(fy);
      ctx.strokeStyle = i===5?'#1e3f5f':'#0d1f32'; ctx.lineWidth=1;
      ctx.beginPath(); ctx.moveTo(PAD.l,cy); ctx.lineTo(W-PAD.r,cy); ctx.stroke();
      ctx.fillStyle='#86a8bf'; ctx.font='9px Consolas'; ctx.textAlign='right';
      ctx.fillText(fy.toFixed(0), PAD.l-4, cy+3);
    }
    for (let i = 0; i <= 4; i++) {
      const xv = X_MIN + i*(X_MAX-X_MIN)/4, cx = toX(xv);
      ctx.strokeStyle='#0d1f32'; ctx.lineWidth=1;
      ctx.beginPath(); ctx.moveTo(cx,PAD.t); ctx.lineTo(cx,H-PAD.b); ctx.stroke();
      ctx.fillStyle='#86a8bf'; ctx.font='9px Consolas'; ctx.textAlign='center';
      ctx.fillText(xv.toFixed(0), cx, H-PAD.b+10);
    }
    ctx.strokeStyle='#1e3f5f'; ctx.lineWidth=1.5;
    ctx.beginPath(); ctx.moveTo(PAD.l,PAD.t); ctx.lineTo(PAD.l,H-PAD.b); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(PAD.l,H-PAD.b); ctx.lineTo(W-PAD.r,H-PAD.b); ctx.stroke();
    ctx.strokeStyle='#00d39f'; ctx.lineWidth=1.5; ctx.beginPath();
    const steps = pW*2;
    for (let i = 0; i <= steps; i++) {
      const x = X_MIN + i*(X_MAX-X_MIN)/steps;
      i===0 ? ctx.moveTo(toX(x),toY(f(x))) : ctx.lineTo(toX(x),toY(f(x)));
    }
    ctx.stroke();
    if (bestX != null) {
      const bx=toX(bestX), by=toY(f(bestX));
      ctx.strokeStyle='#ff3f9b'; ctx.lineWidth=1.5; ctx.setLineDash([5,4]);
      ctx.beginPath(); ctx.moveTo(bx,PAD.t); ctx.lineTo(bx,H-PAD.b); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle='#ff3f9b'; ctx.beginPath(); ctx.arc(bx,by,5,0,Math.PI*2); ctx.fill();
      ctx.fillStyle='#0d1d2f'; ctx.beginPath(); ctx.arc(bx,by,2,0,Math.PI*2); ctx.fill();
    }
  }

  // ── UI ───────────────────────────────────────────────────────
  function addLog(msg) {
    const box = $('logBox');
    box.textContent += `[${new Date().toLocaleTimeString()}] ${msg}\n`;
    box.scrollTop = box.scrollHeight;
  }

  function updateMetrics(gen, best, crossProb, mutProb, bits) {
    $('mGen').textContent = gen;
    $('mX').textContent   = best.x.toFixed(6);
    $('mFx').textContent  = best.fx.toFixed(6);
    $('mCross').textContent = (crossProb*100).toFixed(0)+'%';
    $('mMut').textContent   = (mutProb*100).toFixed(0)+'%';
    $('mBits').textContent  = bits;
    $('bitsDisplay').innerHTML = best.chrom.map(b=>`<span class="${b?'b1':'b0'}">${b}</span>`).join('');
    $('graphLabel').textContent = `x* = ${best.x.toFixed(2)} | f(x*) = ${best.fx.toFixed(2)}`;
  }

  function resetUI() {
    ['mGen','mX','mFx','mCross','mMut','mBits'].forEach(id=>
      $(id).textContent = id==='mGen'?'0':'—'
    );
    $('bitsDisplay').textContent='——————————';
    $('graphLabel').textContent='—';
    $('logBox').textContent='Aguardando execução...';
    drawGraph(null);
  }

  // ── GA main ──────────────────────────────────────────────────
  async function runAG() {
    if (running) return;
    running = true;
    $('runBtn').disabled = true;
    $('logBox').textContent = '';

    const popSize   = Math.max(4,  parseInt($('popSize').value)    || 20);
    const gens      = Math.max(1,  parseInt($('generations').value) || 100);
    const bits      = 10;
    const crossProb = parseInt($('crossSlider').value) / 100;
    const mutProb   = parseInt($('mutSlider').value)   / 100;

    addLog(`Pop: ${popSize} | Ger: ${gens} | Bits: ${bits} | Cruz: ${(crossProb*100).toFixed(0)}% | Mut: ${(mutProb*100).toFixed(0)}%`);

    pop = Array.from({length: popSize}, () => randomChromosome(bits));
    overallBest = null;
    currentGen = 0;

    for (let gen = 1; gen <= gens; gen++) {
      if (!running) break;
      pop = runGeneration(pop, crossProb, mutProb);
      const best = findBest(pop);
      if (!overallBest || best.fx < overallBest.fx) overallBest = best;
      currentGen = gen;
      updateMetrics(gen, overallBest, crossProb, mutProb, bits);
      drawGraph(overallBest.x);
      if (gen===1 || gen%10===0 || gen===gens) {
        addLog(`Gen ${String(gen).padStart(4)}: x*=${overallBest.x.toFixed(4).padStart(10)} | f(x*)=${overallBest.fx.toFixed(4)}`);
      }
      if (gen % 5 === 0) await new Promise(r => setTimeout(r, 0));
    }
    if (running) {
      addLog('─── Concluído ───');
      addLog(`Melhor x*    = ${overallBest.x.toFixed(8)}`);
      addLog(`Melhor f(x*) = ${overallBest.fx.toFixed(8)}`);
    }
    running = false;
    $('runBtn').disabled = false;
  }

  function stop() {
    running = false;
    if ($('runBtn')) $('runBtn').disabled = false;
  }

  function redraw() { drawGraph(overallBest ? overallBest.x : null); }

  function init() {
    precomputeRange();
    drawGraph(null);
    $('crossSlider').addEventListener('input', function() { $('crossVal').textContent = this.value+'%'; });
    $('mutSlider').addEventListener('input',   function() { $('mutVal').textContent   = this.value+'%'; });
    $('runBtn').addEventListener('click',   runAG);
    $('resetBtn').addEventListener('click', () => { if (!running) { pop=[]; overallBest=null; currentGen=0; resetUI(); } });
  }

  return { init, stop, redraw };
})();

document.addEventListener('DOMContentLoaded', () => T08.init());
