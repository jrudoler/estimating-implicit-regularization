import {regressionProblem,regressionAt,randomRegressionStart,penaltyMatrix,theoreticalPenalty,estimateRegularizer} from './early-stopping-math.mjs';
export function initEarlyStopping({canvas,axes,line,colors,number}){
 const $=id=>document.getElementById(id);let problem=regressionProblem(7,5),start=Array(10).fill(0),initSeed=0,playing=false,lastTime=0,frame=0;
 let pool=Array.from({length:20},(_,i)=>i?regressionProblem(7+i,5):problem);
 const step=()=>+$('early-step').value,family=()=>$('early-family').value;
 function stop(){playing=false;cancelAnimationFrame(frame);$('early-play').textContent='Play training';}
 function draw(){
  const t=step(),s=estimateRegularizer(problem,t,start,family(),pool.slice(0,+$('early-endpoints').value));
  $('early-endpoints-control').hidden=family()!=='full';
  $('early-endpoints-value').textContent=$('early-endpoints').value;
  $('early-step-value').textContent=t;$('early-weight-error').textContent=s.weightError===null?'—':s.weightError.toExponential(2);$('early-residual').textContent=s.residual.toExponential(2);
  $('early-initialization').textContent=initSeed?`Random start (seed ${initSeed}); data unchanged.`:'Zero initialization.';
  $('early-status').textContent=(family()==='full'?`Independent weight directions: ${s.rank}/10. `:'')+(s.unidentified?'A zero weight makes a coefficient unidentifiable.':!s.convex?'The fitted total objective is not strictly convex; gradient cancellation does not establish a unique minimum.':s.weightError<1e-8?'The explicit solution recovers the displayed endpoint weights. This alone does not identify the theoretical matrix.':'The fitted penalty does not exactly recover the displayed endpoint weights.');
  const estimated=s.A,reference=initSeed?null:penaltyMatrix(problem,theoreticalPenalty(problem,t));
  const difference=reference?estimated.map((row,i)=>row.map((value,j)=>value-reference[i][j])):null;
  const referenceNorm=reference?Math.hypot(...reference.flat()):0;
  $('early-matrix-error').textContent=difference&&referenceNorm>0?(Math.hypot(...difference.flat())/referenceNorm).toExponential(2):'—';
  const matrixLimit=Math.max(1e-12,...estimated.flat().map(Math.abs),...(reference??[]).flat().map(Math.abs),...(difference??[]).flat().map(Math.abs));
  const compact=x=>Math.abs(x)<.001&&x!==0?x.toExponential(2):number(x,3);
  function heatmap(id,matrix){
   const view=canvas(id),{ctx,w,h}=view,size=Math.min(w-52,h-44),cell=size/10,left=(w-size)/2+10,top=25;
   if(!matrix){ctx.fillStyle=colors.muted;ctx.textAlign='center';ctx.fillText('Reference requires zero initialization',w/2,h/2);return;}
   for(let i=0;i<10;i++)for(let j=0;j<10;j++){
    const fraction=Math.min(1,Math.abs(matrix[i][j])/matrixLimit),pale=Math.round(255*(1-fraction));
    ctx.fillStyle=matrix[i][j]<0?`rgb(255,${pale},${pale})`:`rgb(${pale},${pale},255)`;
    ctx.fillRect(left+j*cell,top+i*cell,cell,cell);ctx.strokeStyle='#ddd';ctx.lineWidth=.4;ctx.strokeRect(left+j*cell,top+i*cell,cell,cell);
   }
   ctx.fillStyle=colors.muted;ctx.textAlign='center';for(let j=0;j<10;j++)ctx.fillText(j+1,left+(j+.5)*cell,top-8);
   ctx.textAlign='right';for(let i=0;i<10;i++)ctx.fillText(i+1,left-8,top+(i+.5)*cell+3);
  }
  heatmap('early-theory',reference);heatmap('early-estimated',estimated);heatmap('early-difference',difference);
  $('early-matrix-scale').textContent=`Shared color scale: red = −${compact(matrixLimit)}, white = 0, blue = +${compact(matrixLimit)}. Rescaled at each selected step. `+(difference?'':'The zero-initialization theory is not a reference for this random start.');
  const v=canvas('early-coefficients'),{ctx}=v;
  const ols=regressionAt(problem,2000).weights,limit=Math.max(.5,...ols.map(Math.abs),...s.weights.map(Math.abs),...(s.fitted??[]).map(Math.abs))*1.18;
  const {X,Y}=axes(v,{xmin:-limit,xmax:limit,ymin:-limit,ymax:limit,xlabel:'Early-stopped weight',ylabel:'Recovered weight',equalAspect:true});
  line(ctx,[[X(-limit),Y(-limit)],[X(limit),Y(limit)]],colors.total,1.5,[5,4]);
  for(let j=0;s.fitted&&j<10;j++){
   const x=X(s.weights[j]),y=Y(s.fitted[j]);
   ctx.strokeStyle=colors.reg;ctx.lineWidth=2;ctx.strokeRect(x-4,y-4,8,8);
  }
  if(!s.fitted){ctx.fillStyle=colors.muted;ctx.textAlign='center';ctx.fillText('No unique minimum for this fitted objective',v.w/2,v.h/2);}
  const trace=canvas('early-penalties'),series=Array.from({length:400},(_,i)=>{const k=i+1;return {t:k,s:regressionAt(problem,k,start)};});
  const log=$('early-log').checked;
  const values=p=>[p.s.scalar.lambda];
  const maximum=Math.max(.1,...series.flatMap(values).filter(x=>x!==null));
  const positive=series.flatMap(values).filter(x=>x>0);
  const low=Math.floor(Math.log10(Math.min(...positive,maximum))),high=Math.ceil(Math.log10(maximum));
  const stride=Math.max(1,Math.ceil((high-low)/6));
  const logTicks=Array.from({length:Math.floor((high-low)/stride)+1},(_,i)=>high-i*stride);
  const ymax=log?high+.3:maximum*1.08, ymin=log?low-.5:0;
  const val=x=>log?Math.log10(x):x;
  const a=axes(trace,{xmin:0,xmax:400,ymin,ymax,xlabel:'Training step',ylabel:'Estimated penalty λ̂',left:65,xticks:[0,100,200,300,400],yticks:log?logTicks:null,formatY:x=>log?(x< -3?'10^'+x:String(10**x)):number(x,2)});
  for(let j=0;j<1;j++){
   let points=[];
   const flush=()=>{if(points.length)line(trace.ctx,points,colors.total,2.5,[]);points=[];};
   for(const p of series){const value=values(p)[j];if(value===null||(log&&value<=0)){flush();continue;}points.push([a.X(p.t),a.Y(val(value))]);}flush();
   const current=series[t-1].s.scalar.lambda;
   if(current!==null&&(!log||current>0)){trace.ctx.fillStyle=colors.ink;trace.ctx.beginPath();trace.ctx.arc(a.X(t),a.Y(val(current)),3,0,2*Math.PI);trace.ctx.fill();}
  }
  const firstZero=series.find(p=>p.s.scalar.lambda===0);
  if(log&&firstZero){
   const x=a.X(firstZero.t),y=a.Y(ymin);
   trace.ctx.strokeStyle=colors.total;trace.ctx.lineWidth=1.5;trace.ctx.beginPath();trace.ctx.arc(x,y,4,0,2*Math.PI);trace.ctx.stroke();
  }
  line(trace.ctx,[[a.X(t),a.Y(ymin)],[a.X(t),a.Y(ymax)]],colors.ink,1,[3,4]);
  $('early-penalty-caption').textContent=`A separate scalar ridge estimate at each step, independent of the selected matrix family. The dot marks step ${t}. `+(log?`Tick labels show penalty values on a logarithmic scale. ${firstZero?'The hollow marker at the bottom marks the first zero estimate (step '+firstZero.t+'); zero has no logarithmic coordinate. Values can reach numerical precision.':''}`:'');

 }
 function tick(now){if(!playing)return;if(now-lastTime>65){lastTime=now;$('early-step').value=Math.min(400,step()+1);draw();}if(step()===400){stop();return;}frame=requestAnimationFrame(tick);}
 $('early-play').onclick=()=>{if(playing){stop();return;}if(step()===400)$('early-step').value=1;playing=true;lastTime=0;$('early-play').textContent='Pause training';frame=requestAnimationFrame(tick);};
 $('early-log').onchange=draw;$('early-endpoints').oninput=()=>{stop();draw();};
 $('early-step').oninput=()=>{stop();draw();};$('early-family').onchange=draw;
 $('early-condition').onchange=()=>{stop();problem=regressionProblem(7,+$('early-condition').value);pool=Array.from({length:20},(_,i)=>i?regressionProblem(7+i,+$('early-condition').value):problem);start=Array(10).fill(0);initSeed=0;draw();};
 $('early-reset').onclick=()=>{stop();start=Array(10).fill(0);initSeed=0;$('early-step').value=20;draw();};
 $('early-random').onclick=()=>{stop();initSeed=1+Math.floor(Math.random()*999999);start=randomRegressionStart(initSeed);draw();};
 draw();return draw;
}
