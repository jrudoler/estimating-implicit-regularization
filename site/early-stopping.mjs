import {regressionProblem,regressionAt,randomRegressionStart} from './early-stopping-math.mjs';
export function initEarlyStopping({canvas,axes,line,colors,number}){
 const $=id=>document.getElementById(id);let problem=regressionProblem(),start=Array(10).fill(0),initSeed=0,playing=false,lastTime=0,frame=0;
 const step=()=>+$('early-step').value,family=()=>$('early-family').value;
 function stop(){playing=false;cancelAnimationFrame(frame);$('early-play').textContent='Play training';}
 function draw(){
  const t=step(),s=regressionAt(problem,t,start,family());
  $('early-step-value').textContent=t;$('early-fit').textContent=s.explained===null?'—':number(100*s.explained,1)+'%';$('early-weight-error').textContent=s.weightError.toExponential(2);$('early-residual').textContent=s.residual.toExponential(2);
  $('early-initialization').textContent=initSeed?`Random start (seed ${initSeed}); data unchanged.`:'Zero initialization.';
  $('early-status').textContent=s.unidentified?'A zero weight makes a penalty coefficient unidentifiable.':s.weightError<1e-8?'The fitted penalty reproduces the early-stopped weights.':family()==='scalar'?'One ridge coefficient cannot generally reproduce all ten weights.':'This start requires a negative penalty in some directions; a nonnegative quadratic cannot match it exactly.';
  const v=canvas('early-coefficients'),{ctx}=v;
  const ols=regressionAt(problem,2000).weights,limit=Math.max(.5,...ols.map(Math.abs),...s.weights.map(Math.abs),...s.fitted.map(Math.abs))*1.18;
  const {X,Y}=axes(v,{xmin:.5,xmax:10.5,ymin:-limit,ymax:limit,xlabel:'Coefficient index',ylabel:'Weight',xticks:Array.from({length:10},(_,i)=>i+1)});
  for(let j=0;j<10;j++){
   const x=X(j+1),y=Y(s.weights[j]),f=Y(s.fitted[j]);line(ctx,[[x,y],[x,f]],'#aaa',2);
   ctx.beginPath();ctx.arc(x,y,6,0,Math.PI*2);ctx.fillStyle=colors.ink;ctx.fill();
   ctx.strokeStyle=colors.reg;ctx.lineWidth=2;ctx.strokeRect(x-8,f-8,16,16);
  }
  const trace=canvas('early-penalties'),series=Array.from({length:400},(_,i)=>{const k=i+1;return {t:k,s:regressionAt(problem,k,start)};});
  const val=x=>Math.log10(1+(x??0));
  const max=Math.max(.1,...series.flatMap(p=>family()==='scalar'?[val(p.s.scalar.lambda)]:p.s.directional.map(val)))*1.08;
  const a=axes(trace,{xmin:0,xmax:Math.log10(400),ymin:0,ymax:max,xlabel:'Training step (log scale)',ylabel:'log₁₀(1 + penalty)',xticks:[1,5,20,100,400].map(Math.log10),formatX:x=>String(Math.round(10**x))});
  if(family()==='directional')for(let j=0;j<10;j++)line(trace.ctx,series.map(p=>[a.X(Math.log10(p.t)),a.Y(val(p.s.directional[j]))]),colors.reg,1.4,[5,4]);
  else line(trace.ctx,series.map(p=>[a.X(Math.log10(p.t)),a.Y(val(p.s.scalar.lambda))]),colors.total,2.5);
  line(trace.ctx,[[a.X(Math.log10(t)),a.Y(0)],[a.X(Math.log10(t)),a.Y(max)]],colors.ink,1,[3,4]);
  $('early-penalty-caption').textContent=family()==='directional'?'Ten inferred penalties, one per Hessian eigenvector. Vertical line: selected step.':'One inferred ridge coefficient shared by all directions. Vertical line: selected step.';
 }
 function tick(now){if(!playing)return;if(now-lastTime>65){lastTime=now;$('early-step').value=Math.min(400,step()+1);draw();}if(step()===400){stop();return;}frame=requestAnimationFrame(tick);}
 $('early-play').onclick=()=>{if(playing){stop();return;}if(step()===400)$('early-step').value=1;playing=true;lastTime=0;$('early-play').textContent='Pause training';frame=requestAnimationFrame(tick);};
 $('early-step').oninput=()=>{stop();draw();};$('early-family').onchange=draw;
 $('early-condition').onchange=()=>{stop();problem=regressionProblem(7,+$('early-condition').value);start=Array(10).fill(0);initSeed=0;draw();};
 $('early-reset').onclick=()=>{stop();start=Array(10).fill(0);initSeed=0;$('early-step').value=20;draw();};
 $('early-random').onclick=()=>{stop();initSeed=1+Math.floor(Math.random()*999999);start=randomRegressionStart(initSeed);draw();};
 draw();return draw;
}
