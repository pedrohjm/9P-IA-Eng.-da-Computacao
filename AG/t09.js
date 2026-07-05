// ── Trabalho 09 — AG Parametrizável ───────────────────────────
const T09 = (() => {
  'use strict';
  const $ = id => document.getElementById('a09-' + id);

  const X_MIN = 0, X_MAX = 512;
  let gFmin = 0;
  let running = false;
  let overallBest = null;
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

  // ── Selection ────────────────────────────────────────────────
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

  function tournamentSelect(p, k) {
    let best = p[Math.floor(Math.random() * p.length)];
    for (let i = 1; i < k; i++) {
      const c = p[Math.floor(Math.random() * p.length)];
      if (f(decode(c)) < f(decode(best))) best = c;
    }
    return [...best];
  }

  // ── Elitism ──────────────────────────────────────────────────
  function getElites(p, elitism) {
    if (elitism <= 0) return [];
    return [...p].sort((a,b)=>f(decode(a))-f(decode(b))).slice(0,elitism).map(c=>[...c]);
  }

  function shuffle(arr) {
    for (let i = arr.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [arr[i], arr[j]] = [arr[j], arr[i]];
    }
  }

  // ── Crossover ────────────────────────────────────────────────
  function cruzamento1pt(p1, p2) {
    const pt = Math.floor(Math.random() * (p1.length - 1)) + 1;
    return [[...p1.slice(0,pt),...p2.slice(pt)],[...p2.slice(0,pt),...p1.slice(pt)]];
  }

  function cruzamento2pt(p1, p2) {
    const n = p1.length;
    if (n < 3) return cruzamento1pt(p1, p2);
    let pt1, pt2;
    do { pt1=Math.floor(Math.random()*(n-1))+1; pt2=Math.floor(Math.random()*(n-1))+1; } while(pt1===pt2);
    if (pt1>pt2) [pt1,pt2]=[pt2,pt1];
    return [
      [...p1.slice(0,pt1),...p2.slice(pt1,pt2),...p1.slice(pt2)],
      [...p2.slice(0,pt1),...p1.slice(pt1,pt2),...p2.slice(pt2)]
    ];
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
      const x=decode(c), fx=f(x);
      if (!best || fx<best.fx) best={x,fx,chrom:c};
    }
    return best;
  }

  // ── One generation ───────────────────────────────────────────
  function runGeneration(p, {crossProb,mutProb,selMethod,k,crossType,elitism}) {
    const n = p.length;
    const elites = getElites(p, elitism);
    let selected;
    if (selMethod==='roleta') {
      const cum = gerarProbabilidades(gerarImagem(p));
      selected = Array.from({length:n}, ()=>rouletteSelect(cum,p));
    } else {
      const kEff = Math.min(k,n);
      selected = Array.from({length:n}, ()=>tournamentSelect(p,kEff));
    }
    shuffle(selected);
    let numCross=Math.floor(n*crossProb);
    if(numCross%2!==0) numCross=Math.max(0,numCross-1);
    const toCross=selected.slice(0,numCross), noChange=selected.slice(numCross);
    const filhos=[];
    for(let i=0;i+1<toCross.length;i+=2) {
      const [c1,c2]=crossType==='1pt'?cruzamento1pt(toCross[i],toCross[i+1]):cruzamento2pt(toCross[i],toCross[i+1]);
      filhos.push(c1,c2);
    }
    const filhosMutados=efetuarMutacao(filhos,mutProb);
    const newPop=[...noChange,...filhosMutados];
    newPop.sort((a,b)=>f(decode(a))-f(decode(b)));
    return [...elites,...newPop.slice(0,Math.max(0,n-elitism))];
  }

  // ── Graph ────────────────────────────────────────────────────
  function precomputeRange() {
    let mn=0;
    for(let i=1;i<=3000;i++){const v=f(X_MIN+i*(X_MAX-X_MIN)/3000);if(v<mn)mn=v;}
    gFmin=mn*1.06;
  }

  function drawGraph(bestX) {
    const canvas=$('graphCanvas'); if(!canvas)return;
    const W=canvas.width,H=canvas.height,ctx=canvas.getContext('2d');
    const PAD={l:50,r:14,t:12,b:28},pW=W-PAD.l-PAD.r,pH=H-PAD.t-PAD.b;
    ctx.clearRect(0,0,W,H); ctx.fillStyle='#081523'; ctx.fillRect(0,0,W,H);
    const toX=x=>PAD.l+(x-X_MIN)/(X_MAX-X_MIN)*pW;
    const toY=fy=>PAD.t+(-fy/-gFmin)*pH;
    for(let i=0;i<=5;i++){
      const fy=gFmin*(1-i/5),cy=toY(fy);
      ctx.strokeStyle=i===5?'#1e3f5f':'#0d1f32';ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(PAD.l,cy);ctx.lineTo(W-PAD.r,cy);ctx.stroke();
      ctx.fillStyle='#86a8bf';ctx.font='9px Consolas';ctx.textAlign='right';
      ctx.fillText(fy.toFixed(0),PAD.l-4,cy+3);
    }
    for(let i=0;i<=4;i++){
      const xv=X_MIN+i*(X_MAX-X_MIN)/4,cx=toX(xv);
      ctx.strokeStyle='#0d1f32';ctx.lineWidth=1;
      ctx.beginPath();ctx.moveTo(cx,PAD.t);ctx.lineTo(cx,H-PAD.b);ctx.stroke();
      ctx.fillStyle='#86a8bf';ctx.font='9px Consolas';ctx.textAlign='center';
      ctx.fillText(xv.toFixed(0),cx,H-PAD.b+10);
    }
    ctx.strokeStyle='#1e3f5f';ctx.lineWidth=1.5;
    ctx.beginPath();ctx.moveTo(PAD.l,PAD.t);ctx.lineTo(PAD.l,H-PAD.b);ctx.stroke();
    ctx.beginPath();ctx.moveTo(PAD.l,H-PAD.b);ctx.lineTo(W-PAD.r,H-PAD.b);ctx.stroke();
    ctx.strokeStyle='#00d39f';ctx.lineWidth=1.5;ctx.beginPath();
    const steps=pW*2;
    for(let i=0;i<=steps;i++){
      const x=X_MIN+i*(X_MAX-X_MIN)/steps;
      i===0?ctx.moveTo(toX(x),toY(f(x))):ctx.lineTo(toX(x),toY(f(x)));
    }
    ctx.stroke();
    if(bestX!=null){
      const bx=toX(bestX),by=toY(f(bestX));
      ctx.strokeStyle='#ff3f9b';ctx.lineWidth=1.5;ctx.setLineDash([5,4]);
      ctx.beginPath();ctx.moveTo(bx,PAD.t);ctx.lineTo(bx,H-PAD.b);ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle='#ff3f9b';ctx.beginPath();ctx.arc(bx,by,5,0,Math.PI*2);ctx.fill();
      ctx.fillStyle='#0d1d2f';ctx.beginPath();ctx.arc(bx,by,2,0,Math.PI*2);ctx.fill();
    }
  }

  // ── UI ───────────────────────────────────────────────────────
  function addLog(msg) {
    const box=$('logBox');
    box.textContent+=`[${new Date().toLocaleTimeString()}] ${msg}\n`;
    box.scrollTop=box.scrollHeight;
  }

  function updateMetrics(gen, best, cfg) {
    const {selMethod,k,crossType,elitism,bits}=cfg;
    $('mGen').textContent=gen;
    $('mX').textContent=best.x.toFixed(6);
    $('mFx').textContent=best.fx.toFixed(6);
    $('mSel').textContent=selMethod==='roleta'?'Roleta':`Torneio (k=${k})`;
    $('mCross').textContent=crossType==='1pt'?'1 ponto':'2 pontos';
    $('mElite').textContent=elitism>0?`${elitism} preservado(s)`:'Desativado';
    $('mBits').textContent=bits;
    $('bitsDisplay').innerHTML=best.chrom.map(b=>`<span class="${b?'b1':'b0'}">${b}</span>`).join('');
    $('graphLabel').textContent=`x* = ${best.x.toFixed(2)} | f(x*) = ${best.fx.toFixed(2)}`;
  }

  function resetUI() {
    ['mX','mFx','mSel','mCross','mElite','mBits'].forEach(id=>$(id).textContent='—');
    $('mGen').textContent='0';
    $('bitsDisplay').textContent='——————————';
    $('graphLabel').textContent='—';
    $('logBox').textContent='Aguardando execução...';
    drawGraph(null);
  }

  function readCfg() {
    return {
      popSize:   Math.max(4,parseInt($('popSize').value)||20),
      gens:      Math.max(1,parseInt($('generations').value)||100),
      bits:      Math.min(20,Math.max(4,parseInt($('bits').value)||10)),
      crossProb: parseInt($('crossSlider').value)/100,
      mutProb:   parseInt($('mutSlider').value)/100,
      elitism:   Math.max(0,parseInt($('elitism').value)||0),
      selMethod: document.querySelector('input[name="a09-sel"]:checked').value,
      k:         Math.max(2,parseInt($('kTorneio').value)||3),
      crossType: document.querySelector('input[name="a09-cross"]:checked').value
    };
  }

  // ── GA main ──────────────────────────────────────────────────
  async function runAG() {
    if (running) return;
    running = true;
    $('runBtn').disabled = true;
    $('logBox').textContent = '';
    const cfg = readCfg();
    const selLbl=cfg.selMethod==='roleta'?'Roleta':`Torneio(k=${cfg.k})`;
    addLog(`Pop:${cfg.popSize} | Ger:${cfg.gens} | Bits:${cfg.bits} | Cruz:${(cfg.crossProb*100).toFixed(0)}% | Mut:${(cfg.mutProb*100).toFixed(0)}%`);
    addLog(`Seleção:${selLbl} | Cross:${cfg.crossType==='1pt'?'1pt':'2pt'} | Elite:${cfg.elitism}`);
    pop = Array.from({length:cfg.popSize}, ()=>randomChromosome(cfg.bits));
    overallBest = null;
    for (let gen=1; gen<=cfg.gens; gen++) {
      if (!running) break;
      pop = runGeneration(pop, cfg);
      const best=findBest(pop);
      if (!overallBest||best.fx<overallBest.fx) overallBest=best;
      updateMetrics(gen,overallBest,cfg);
      drawGraph(overallBest.x);
      if(gen===1||gen%10===0||gen===cfg.gens) {
        addLog(`Gen ${String(gen).padStart(4)}: x*=${overallBest.x.toFixed(4).padStart(10)} | f(x*)=${overallBest.fx.toFixed(4)}`);
      }
      if(gen%5===0) await new Promise(r=>setTimeout(r,0));
    }
    if(running){
      addLog('─── Concluído ───');
      addLog(`Melhor x*    = ${overallBest.x.toFixed(8)}`);
      addLog(`Melhor f(x*) = ${overallBest.fx.toFixed(8)}`);
    }
    running=false;
    $('runBtn').disabled=false;
  }

  function stop() {
    running=false;
    if($('runBtn')) $('runBtn').disabled=false;
  }

  function redraw() { drawGraph(overallBest?overallBest.x:null); }

  function init() {
    precomputeRange();
    drawGraph(null);
    $('crossSlider').addEventListener('input',function(){$('crossVal').textContent=this.value+'%';});
    $('mutSlider').addEventListener('input',function(){$('mutVal').textContent=this.value+'%';});
    document.querySelectorAll('input[name="a09-sel"]').forEach(r=>r.addEventListener('change',()=>{
      $('torneioCfg').style.display=document.getElementById('a09-selTorneio').checked?'flex':'none';
    }));
    $('runBtn').addEventListener('click', runAG);
    $('resetBtn').addEventListener('click',()=>{if(!running){pop=[];overallBest=null;resetUI();}});
  }

  return { init, stop, redraw };
})();

document.addEventListener('DOMContentLoaded', () => T09.init());
