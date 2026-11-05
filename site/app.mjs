import {balance,landscape,optimum2D,ridgeEstimate,dataset,createModel,predict,target} from './math.mjs';
const $=id=>document.getElementById(id);
const colors={loss:'#ff0000',reg:'#0000ff',total:'#555555',ink:'#222222',muted:'#707070',grid:'#e9e9e9'};
const number=(x,d=2)=>Number.isFinite(x)?x.toFixed(d):'—';
function canvas(id){const el=$(id),r=el.getBoundingClientRect(),dpr=window.devicePixelRatio||1;el.width=r.width*dpr;el.height=r.height*dpr;const ctx=el.getContext('2d');ctx.scale(dpr,dpr);ctx.font='11px system-ui';return {el,ctx,w:r.width,h:r.height};}
function axes(view,{xmin,xmax,ymin,ymax,xlabel='θ',ylabel='Objective',left=55,bottom=38}){
 const {ctx,w,h}=view,pad={l:left,r:25,t:24,b:bottom};
 const X=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),Y=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
 ctx.strokeStyle=colors.grid;ctx.fillStyle=colors.muted;ctx.lineWidth=1;
 for(let i=0;i<=4;i++){const y=ymin+(ymax-ymin)*i/4;ctx.beginPath();ctx.moveTo(X(xmin),Y(y));ctx.lineTo(X(xmax),Y(y));ctx.stroke();ctx.textAlign='right';ctx.fillText(number(y,1),pad.l-10,Y(y)+4);}
 for(let i=0;i<=4;i++){const x=xmin+(xmax-xmin)*i/4;ctx.textAlign='center';ctx.fillText(number(x,1),X(x),h-pad.b+20);}
 ctx.textAlign='right';ctx.fillText(xlabel,w-pad.r,h-5);ctx.textAlign='left';ctx.fillText(ylabel,pad.l,14);
 return {X,Y,unX:px=>xmin+(px-pad.l)/(w-pad.l-pad.r)*(xmax-xmin),unY:py=>ymin+(h-pad.b-py)/(h-pad.t-pad.b)*(ymax-ymin),pad};
}
function line(ctx,points,color,width=2,dash=[]){ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();ctx.setLineDash([]);}
function dot(ctx,x,y,color,r=5){ctx.beginPath();if(color===colors.reg){ctx.moveTo(x,y-r-1);ctx.lineTo(x+r+1,y+r);ctx.lineTo(x-r-1,y+r);ctx.closePath();}else if(color===colors.total){ctx.moveTo(x,y-r-1);ctx.lineTo(x+r+1,y);ctx.lineTo(x,y+r+1);ctx.lineTo(x-r-1,y);ctx.closePath();}else{ctx.arc(x,y,r,0,2*Math.PI);}ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='#ffffff';ctx.lineWidth=2;ctx.stroke();}
function arrow(ctx,from,to,color){const dx=to[0]-from[0],dy=to[1]-from[1],len=Math.hypot(dx,dy);if(len<1)return;line(ctx,[from,to],color,color===colors.total?3:2.5,color===colors.reg?[5,4]:[]);const a=Math.atan2(dy,dx);ctx.beginPath();ctx.moveTo(...to);ctx.lineTo(to[0]-8*Math.cos(a-.42),to[1]-8*Math.sin(a-.42));ctx.lineTo(to[0]-8*Math.cos(a+.42),to[1]-8*Math.sin(a+.42));ctx.closePath();ctx.fillStyle=color;ctx.fill();ctx.font='11px system-ui';ctx.textAlign='left';ctx.strokeStyle='#fff';ctx.lineWidth=3;const label=color===colors.loss?'∇L':color===colors.reg?'∇R':'Net';ctx.strokeText(label,to[0]+5,to[1]-6);ctx.fillText(label,to[0]+5,to[1]-6);}
let hidden=false,balAxes,balAnimation=0;
function drawBalance(){
 const v=canvas('balance-chart'),{ctx}=v,l=+$('lambda').value,t=+$('theta').value,b=balance(t,l);
 balAxes=axes(v,{xmin:-1,xmax:3,ymin:0,ymax:Math.max(5,9*l+1),ylabel:'Objective value'});
 const {X,Y}=balAxes;
 for(const [key,col] of [['loss',colors.loss],['penalty',colors.reg],['total',colors.total]]){
   if(hidden&&key!=='loss')continue;
   line(ctx,Array.from({length:201},(_,i)=>{const x=-1+i*.02;return[X(x),Y(balance(x,l)[key])];}),col,key==='total'?3.5:2,key==='penalty'?[7,5]:[]);
 }
 for(const [key,label,x,col] of [['loss','Loss',-.65,colors.loss],['penalty','Penalty',2.7,colors.reg],['total','Total',2.15,colors.total]]){
   if(hidden&&key!=='loss')continue;
   ctx.font='12px system-ui';ctx.textAlign='left';ctx.lineWidth=4;ctx.strokeStyle='#fff';ctx.fillStyle=col;
   const yy=Math.max(32,Y(balance(x,l)[key])-10);ctx.strokeText(label,X(x),yy);ctx.fillText(label,X(x),yy);
 }
 line(ctx,[[X(t),Y(0)],[X(t),Y(hidden?b.loss:b.total)]],colors.muted,1,[4,4]);
 dot(ctx,X(t),Y(b.loss),colors.loss);if(!hidden){dot(ctx,X(t),Y(b.penalty),colors.reg);dot(ctx,X(t),Y(b.total),colors.total);dot(ctx,X(b.optimum),Y(balance(b.optimum,l).total),colors.total,4);}
 const span=.45;line(ctx,[[X(t-span),Y(b.loss-b.gLoss*span)],[X(t+span),Y(b.loss+b.gLoss*span)]],colors.loss,4.5);
 if(!hidden)line(ctx,[[X(t-span),Y(b.penalty-b.gReg*span)],[X(t+span),Y(b.penalty+b.gReg*span)]],colors.reg,4.5,[7,4]);
 drawGradientBalance(b,t);
 $('lambda-value').textContent=hidden?'hidden':number(l);$('theta-value').textContent=number(t);
 $('g-loss').textContent=number(b.gLoss);$('g-reg').textContent=hidden?'?':number(b.gReg);$('g-net').textContent=hidden?'?':number(b.gLoss+b.gReg);
 $('balance-status').textContent=hidden?'What penalty could explain this point?':Math.abs(b.gLoss+b.gReg)<.025?'At the optimum, the slopes cancel.':'The net gradient is nonzero. Move toward the balance.';
 if(hidden){const est=ridgeEstimate([t],[b.gLoss]);$('inverse-result').textContent=est.lambda===null?'At θ = 0, ridge supplies no gradient. Its strength cannot be identified here.':`Assuming this point is stationary: estimated λ = ${number(est.lambda,3)}. Unexplained gradient = ${number(est.residual,3)}.${Math.abs(t-b.optimum)>.02?' Away from the true optimum, this need not recover the hidden penalty.':' This matches the hidden penalty at the true optimum.'}`;}
}
function drawGradientBalance(b,theta){
 const {ctx,w}=canvas('gradient-chart');
 const left=w<500?125:180,right=w-24,center=(left+right)/2;
 const max=Math.max(1,Math.abs(b.gLoss),hidden?0:Math.abs(b.gReg),hidden?0:Math.abs(b.gLoss+b.gReg));
 const scale=(right-left)/2/max*.9;
 const rows=[['Loss ∇L',b.gLoss,colors.loss],['Penalty ∇R',hidden?null:b.gReg,colors.reg],['Net',hidden?null:b.gLoss+b.gReg,colors.total]];
 line(ctx,[[center,10],[center,143]],'#bbbbbb',1,[2,3]);
 rows.forEach(([label,value,color],i)=>{
   const y=28+i*48;
   ctx.font='600 13px system-ui';ctx.textAlign='left';ctx.fillStyle=color;ctx.fillText(label,0,y-4);
   ctx.font='13px system-ui';ctx.fillText(value===null?'hidden':(value>0?'+':'')+number(value,2),0,y+14);
   if(value===null){ctx.fillStyle='#707070';ctx.fillText('?',center,y+4);return;}
   const end=center+value*scale;
   if(Math.abs(value)<.015){ctx.beginPath();ctx.arc(center,y,4,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();if(i===2){ctx.font='12px system-ui';ctx.fillText('balanced',center+10,y+4);}return;}
   line(ctx,[[center,y],[end,y]],color,i===2?3:4,color===colors.reg?[8,5]:[]);
   const direction=Math.sign(value);ctx.beginPath();ctx.moveTo(end,y);ctx.lineTo(end-direction*10,y-6);ctx.lineTo(end-direction*10,y+6);ctx.closePath();ctx.fillStyle=color;ctx.fill();
   ctx.beginPath();ctx.arc(center,y,3,0,Math.PI*2);ctx.fill();
 });
}
for(const id of ['lambda','theta'])$(id).addEventListener('input',()=>{balAnimation++;drawBalance();});
$('find-balance').onclick=()=>{const token=++balAnimation,start=+$('theta').value,end=balance(0,+$('lambda').value).optimum,time=performance.now(),duration=matchMedia('(prefers-reduced-motion: reduce)').matches?0:650;function step(now){if(token!==balAnimation)return;const u=duration?Math.min(1,(now-time)/duration):1;$('theta').value=start+(end-start)*(1-(1-u)**3);drawBalance();if(u<1)requestAnimationFrame(step);}requestAnimationFrame(step);};
$('hide-penalty').onclick=()=>{hidden=!hidden;$('hide-penalty').textContent=hidden?'Reveal the penalty':'Hide λ & infer it from θ';$('hide-penalty').setAttribute('aria-pressed',hidden);$('lambda').disabled=hidden;$('inverse-result').hidden=!hidden;drawBalance();};
$('balance-chart').addEventListener('pointerdown',e=>{balAnimation++;const r=e.currentTarget.getBoundingClientRect();$('theta').value=Math.max(-1,Math.min(3,balAxes.unX(e.clientX-r.left)));drawBalance();e.currentTarget.setPointerCapture(e.pointerId);});
$('balance-chart').addEventListener('pointermove',e=>{if(e.buttons){const r=e.currentTarget.getBoundingClientRect();$('theta').value=Math.max(-1,Math.min(3,balAxes.unX(e.clientX-r.left)));drawBalance();}});
let point=[1,.5],followOptimum=true,contourAxes,yaw=-.61,pitch=.7;
function config(){return {l:+$('lambda2').value,a:+$('angle').value*Math.PI/180,s:+$('stretch').value};}
function drawLandscape(){
 const {l,a,s}=config(),opt=optimum2D(l,a,s);if(followOptimum)point=opt;
 const v=canvas('contour-chart'),{ctx}=v;contourAxes=axes(v,{xmin:-2,xmax:3,ymin:-2,ymax:3,xlabel:'θ₁',ylabel:'θ₂'});const {X,Y}=contourAxes;
 ctx.save();ctx.beginPath();ctx.rect(X(-2),Y(3),X(3)-X(-2),Y(-2)-Y(3));ctx.clip();
 for(const level of [.1,.35,.8,1.5,2.7,4.5,7,10,15,22]){
   const pts=Array.from({length:161},(_,i)=>{const t=i/160*Math.PI*2,u=Math.sqrt(2*level/(1+2*l))*Math.cos(t),vv=Math.sqrt(2*level/(s+2*l))*Math.sin(t);return[X(opt[0]+Math.cos(a)*u-Math.sin(a)*vv),Y(opt[1]+Math.sin(a)*u+Math.cos(a)*vv)];});
   line(ctx,pts,'#b9b9b9',1);
 }
 const b=landscape(...point,l,a,s),sum=b.gLoss.map((n,i)=>n+b.gReg[i]);
 const from=[X(point[0]),Y(point[1])],scale=.25;
 for(const [g,color] of [[b.gLoss,colors.loss],[b.gReg,colors.reg],[sum,colors.total]])arrow(ctx,from,[X(point[0]+scale*g[0]),Y(point[1]+scale*g[1])],color);
 dot(ctx,X(1.5),Y(.9),colors.loss,4);dot(ctx,X(0),Y(0),colors.reg,4);dot(ctx,X(opt[0]),Y(opt[1]),colors.total,5);dot(ctx,...from,colors.ink,4);ctx.restore();
 $('lambda2-value').textContent=number(l);$('stretch-value').textContent=number(s,1)+'×';$('angle-value').textContent=number(a*180/Math.PI,0)+'°';$('point-x').value=point[0];$('point-y').value=point[1];
 $('vector-status').textContent=`‖∇L + ∇R‖ = ${number(Math.hypot(...sum),3)}`;
 $('vector-detail').textContent=Math.hypot(...sum)<.001?'The two gradients cancel at the combined minimum.':'The arrows do not cancel here. Descent follows the negative net gradient.';
 drawSurface();
}
function drawSurface(){
 const v=canvas('surface-chart'),{ctx,w,h}=v,{l,a,s}=config(),range=5,scale=Math.min(w/7.8,h/6.2);
 const maxZ=24;const project=(x,y,z)=>{x-=.5;y-=.5;const xx=Math.cos(yaw)*x-Math.sin(yaw)*y,yy=Math.sin(yaw)*x+Math.cos(yaw)*y;return[w*.5+scale*xx,h*.66+scale*(Math.sin(pitch)*yy-Math.cos(pitch)*z*.13)];};
 const z=(x,y)=>Math.min(maxZ,landscape(x,y,l,a,s).total);
 const cells=[];const n=25;
 for(let i=0;i<n;i++)for(let j=0;j<n;j++){const x=-2+range*i/n,y=-2+range*j/n,d=range/n;const vertices=[[x,y],[x+d,y],[x+d,y+d],[x,y+d]];cells.push({vertices,depth:Math.sin(yaw)*(x+d/2)+Math.cos(yaw)*(y+d/2),z:z(x+d/2,y+d/2)});}
 cells.sort((u,v)=>u.depth-v.depth);
 for(const cell of cells){ctx.beginPath();cell.vertices.forEach(([x,y],i)=>{const p=project(x,y,z(x,y));i?ctx.lineTo(...p):ctx.moveTo(...p);});ctx.closePath();ctx.fillStyle=`hsl(0 0% ${94-28*Math.min(cell.z/maxZ,1)}%)`;ctx.fill();ctx.strokeStyle='#ffffff90';ctx.lineWidth=.6;ctx.stroke();}
 const opt=optimum2D(l,a,s);dot(ctx,...project(opt[0],opt[1],z(...opt)),colors.total,5);dot(ctx,...project(point[0],point[1],z(...point)),colors.ink,4);
 // Gradients live in parameter space, so draw them on the z=0 plane.
 const b=landscape(...point,l,a,s),base=project(...point,0);line(ctx,[base,project(...point,z(...point))],colors.muted,1,[3,3]);
 for(const [g,c] of [[b.gLoss,colors.loss],[b.gReg,colors.reg],[b.gLoss.map((x,i)=>x+b.gReg[i]),colors.total]])arrow(ctx,base,project(point[0]+g[0]*.25,point[1]+g[1]*.25,0),c);
 ctx.fillStyle=colors.muted;ctx.textAlign='left';ctx.fillText('Height = loss + penalty · view clipped at 24',20,h-13);
}
for(const id of ['lambda2','stretch','angle'])$(id).addEventListener('input',drawLandscape);
$('snap2').onclick=()=>{followOptimum=true;drawLandscape();};
$('contour-chart').onclick=e=>{const r=e.currentTarget.getBoundingClientRect();point=[Math.max(-2,Math.min(3,contourAxes.unX(e.clientX-r.left))),Math.max(-2,Math.min(3,contourAxes.unY(e.clientY-r.top)))];followOptimum=false;drawLandscape();};
for(const id of ['point-x','point-y'])$(id).oninput=()=>{point=[+$('point-x').value,+$('point-y').value];followOptimum=false;drawLandscape();};
$('camera').oninput=()=>{yaw=+$('camera').value*Math.PI/180;drawSurface();};
let drag=null;$('surface-chart').onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,yaw,pitch};e.currentTarget.setPointerCapture(e.pointerId);};
$('surface-chart').onpointermove=e=>{if(!drag)return;yaw=drag.yaw+(e.clientX-drag.x)*.009;pitch=Math.max(.25,Math.min(1.3,drag.pitch+(e.clientY-drag.y)*.005));$('camera').value=((yaw*180/Math.PI+180)%360+360)%360-180;drawSurface();};
$('surface-chart').onpointerup=$('surface-chart').onpointercancel=()=>{drag=null;};
let data=dataset(7),curve=[],worker=null,state='ready',revealed=false,lastStats=null;
function drawNetwork(){
 const v=canvas('network-chart'),{ctx}=v,{X,Y}=axes(v,{xmin:-1.6,xmax:1.6,ymin:-1.2,ymax:1.2,xlabel:'Input x',ylabel:'Prediction f(x)'});
 line(ctx,Array.from({length:151},(_,i)=>{const x=-1.5+i/50;return[X(x),Y(target(x))];}),colors.muted,1.5,[5,4]);
 for(const p of data)dot(ctx,X(p.x),Y(p.y),'#777777',3);
 if(curve.length)line(ctx,curve.map(p=>[X(p.x),Y(p.y)]),colors.total,2.5);
}
function initialCurve(){const model=createModel(+$('seed').value);curve=Array.from({length:151},(_,i)=>({x:-1.5+i/50,y:predict(model,-1.5+i/50)}));}
function updateStats(stats){
 lastStats=stats;curve=stats.curve;drawNetwork();
 $('estimate').textContent=number(stats.estimate.lambda,4);$('explained').textContent=stats.estimate.explained===null?'—':number(stats.estimate.explained*100,1)+'%';
 $('explained-meter').style.width=(Math.max(0,stats.estimate.explained??0)*100)+'%';$('train-loss').textContent=number(stats.loss,4);$('stationarity').textContent=stats.stationary.toExponential(2);
 $('training-bar').style.width=stats.epoch/stats.limit*100+'%';$('reveal').disabled=false;
 if(state==='running')$('train-status').textContent=`Training · ${stats.epoch.toLocaleString()} / ${stats.limit.toLocaleString()} full-batch steps`;
 if(stats.done){state='done';$('train').textContent='Train again';$('train-status').textContent=`Finished ${stats.epoch.toLocaleString()} steps. Change the seed or penalty to try another run.`;$('penalty-choice').disabled=$('seed').disabled=false;}
 if(revealed)showReveal();
}
function showReveal(){revealed=true;const truth=+$('penalty-choice').value;$('true-lambda').textContent=number(truth,4);$('reveal').textContent='True penalty revealed';$('reveal').disabled=true;if(lastStats?.estimate.lambda!==null&&lastStats){const error=Math.abs(lastStats.estimate.lambda-truth)/truth*100;$('recovery-note').textContent=`Relative recovery error: ${number(error,1)}%. ${state==='done'?'Finite training and ReLU kinks can leave a residual.':'Keep training to see whether the estimate improves.'}`;}}
function reset(){worker?.terminate();worker=null;state='ready';revealed=false;lastStats=null;$('seed').disabled=$('penalty-choice').disabled=false;data=dataset(+$('seed').value);initialCurve();for(const id of ['estimate','explained','train-loss','stationarity'])$(id).textContent='—';$('true-lambda').textContent='hidden';$('reveal').disabled=true;$('reveal').textContent='Reveal the true penalty';$('train').textContent='Train the network →';$('train-status').textContent='Ready. Training runs locally; training data stays in your browser.';$('training-bar').style.width=$('explained-meter').style.width='0%';$('recovery-note').textContent='Watch the estimate settle as the network approaches a stationary point.';drawNetwork();}
$('train').onclick=()=>{
 if(state==='running'){worker.postMessage({type:'pause'});state='paused';$('train').textContent='Resume training';$('train-status').textContent='Paused. Inspect the current estimate or resume training.';return;}
 if(state==='paused'){state='running';worker.postMessage({type:'resume'});$('train').textContent='Pause training';return;}
 reset();
 try{worker=new Worker(new URL('./train-worker.mjs',import.meta.url),{type:'module'});worker.onmessage=({data})=>updateStats(data);worker.onerror=()=>{state='error';worker?.terminate();$('train-status').textContent='Training could not start. Try a modern browser with module-worker support, served over HTTP(S).';$('train').textContent='Try again';$('seed').disabled=$('penalty-choice').disabled=false;};state='running';$('seed').disabled=$('penalty-choice').disabled=true;$('train').textContent='Pause training';$('train-status').textContent='Starting local training…';worker.postMessage({type:'start',seed:+$('seed').value,lambda:+$('penalty-choice').value});}
 catch(error){state='error';$('train-status').textContent=`Training unavailable: ${error.message}`;}
};
$('reset').onclick=reset;$('reveal').onclick=showReveal;$('seed').onchange=$('penalty-choice').onchange=reset;
let resizeFrame;new ResizeObserver(()=>{cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(()=>{drawBalance();drawLandscape();drawNetwork();});}).observe(document.querySelector('main'));
initialCurve();drawBalance();drawLandscape();drawNetwork();
