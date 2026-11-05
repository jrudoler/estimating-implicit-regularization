import {initEarlyStopping} from './early-stopping.mjs';
import {balance,landscape,optimum2D,ridgeEstimate,dataset,createModel,predict,target} from './math.mjs?v=2';
const $=id=>document.getElementById(id);
const randomBetween=(min,max)=>min+(max-min)*Math.random();
const colors={loss:'#ff0000',reg:'#0000ff',total:'#555555',ink:'#222222',muted:'#707070',grid:'#e9e9e9'};
const number=(x,d=2)=>Number.isFinite(x)?(Math.abs(x)<0.5*10**(-d)?0:x).toFixed(d):'—';
function canvas(id){const el=$(id),r=el.getBoundingClientRect(),dpr=window.devicePixelRatio||1;el.width=r.width*dpr;el.height=r.height*dpr;const ctx=el.getContext('2d');ctx.scale(dpr,dpr);ctx.font='11px system-ui';return {el,ctx,w:r.width,h:r.height};}
function axes(view,{xmin,xmax,ymin,ymax,xlabel='θ',ylabel='Objective',left=55,bottom=38,equalAspect=false,xticks=null,formatX=null,yticks=null,formatY=null}){
 const {ctx,w,h}=view,pad={l:left,r:25,t:24,b:bottom};
 if(equalAspect){const unit=Math.min((w-pad.l-pad.r)/(xmax-xmin),(h-pad.t-pad.b)/(ymax-ymin)),dx=(w-pad.l-pad.r-unit*(xmax-xmin))/2,dy=(h-pad.t-pad.b-unit*(ymax-ymin))/2;pad.l+=dx;pad.r+=dx;pad.t+=dy;pad.b+=dy;}
 const X=x=>pad.l+(x-xmin)/(xmax-xmin)*(w-pad.l-pad.r),Y=y=>h-pad.b-(y-ymin)/(ymax-ymin)*(h-pad.t-pad.b);
 ctx.strokeStyle=colors.grid;ctx.fillStyle=colors.muted;ctx.lineWidth=1;
 for(const y of (yticks??Array.from({length:5},(_,i)=>ymin+(ymax-ymin)*i/4))){ctx.beginPath();ctx.moveTo(X(xmin),Y(y));ctx.lineTo(X(xmax),Y(y));ctx.stroke();ctx.textAlign='right';ctx.fillText(formatY?formatY(y):number(y,1),pad.l-10,Y(y)+4);}
 for(const x of (xticks??Array.from({length:5},(_,i)=>xmin+(xmax-xmin)*i/4))){ctx.textAlign='center';ctx.fillText(formatX?formatX(x):number(x,Number.isInteger(x)?0:1),X(x),h-pad.b+20);}
 ctx.textAlign='right';ctx.fillText(xlabel,w-pad.r,h-5);ctx.textAlign='left';ctx.fillText(ylabel,pad.l,14);
 return {X,Y,unX:px=>xmin+(px-pad.l)/(w-pad.l-pad.r)*(xmax-xmin),unY:py=>ymin+(h-pad.b-py)/(h-pad.t-pad.b)*(ymax-ymin),pad};
}
function line(ctx,points,color,width=2,dash=[]){ctx.beginPath();ctx.strokeStyle=color;ctx.lineWidth=width;ctx.setLineDash(dash);points.forEach(([x,y],i)=>i?ctx.lineTo(x,y):ctx.moveTo(x,y));ctx.stroke();ctx.setLineDash([]);}
function dot(ctx,x,y,color,r=5){ctx.beginPath();if(color===colors.reg){ctx.moveTo(x,y-r-1);ctx.lineTo(x+r+1,y+r);ctx.lineTo(x-r-1,y+r);ctx.closePath();}else if(color===colors.total){ctx.moveTo(x,y-r-1);ctx.lineTo(x+r+1,y);ctx.lineTo(x,y+r+1);ctx.lineTo(x-r-1,y);ctx.closePath();}else{ctx.arc(x,y,r,0,2*Math.PI);}ctx.fillStyle=color;ctx.fill();ctx.strokeStyle='#ffffff';ctx.lineWidth=2;ctx.stroke();}
function arrow(ctx,from,to,color){const dx=to[0]-from[0],dy=to[1]-from[1],len=Math.hypot(dx,dy);if(len<1)return;line(ctx,[from,to],color,color===colors.total?3:2.5,color===colors.reg?[5,4]:[]);const a=Math.atan2(dy,dx);ctx.beginPath();ctx.moveTo(...to);ctx.lineTo(to[0]-8*Math.cos(a-.42),to[1]-8*Math.sin(a-.42));ctx.lineTo(to[0]-8*Math.cos(a+.42),to[1]-8*Math.sin(a+.42));ctx.closePath();ctx.fillStyle=color;ctx.fill();ctx.font='11px system-ui';ctx.textAlign=dx<0?'right':'left';ctx.strokeStyle='#fff';ctx.lineWidth=3;const label=color===colors.loss?'∇L':color===colors.reg?'∇R':'Net';const labelX=to[0]+(dx<0?-6:6),labelY=to[1]+(dy>0?14:-6);ctx.strokeText(label,labelX,labelY);ctx.fillText(label,labelX,labelY);}
// Keep the computed parameter independent of the range input's 0.01 snapping.
let balanceTheta=+$('theta').value;
let balAxes,balAnimation=0;
function drawBalance(){
 const v=canvas('balance-chart'),{ctx}=v,l=+$('lambda').value,t=balanceTheta,b=balance(t,l);
 balAxes=axes(v,{xmin:-1,xmax:3,ymin:0,ymax:Math.max(5,9*l+1),ylabel:'Objective value'});
 const {X,Y}=balAxes;
 for(const [key,col] of [['loss',colors.loss],['penalty',colors.reg],['total',colors.total]]){
   line(ctx,Array.from({length:201},(_,i)=>{const x=-1+i*.02;return[X(x),Y(balance(x,l)[key])];}),col,key==='total'?3.5:2,key==='penalty'?[7,5]:[]);
 }
 for(const [key,label,x,col] of [['loss','Loss',-.65,colors.loss],['penalty','Penalty',2.7,colors.reg],['total','Total',2.15,colors.total]]){
   ctx.font='12px system-ui';ctx.textAlign='left';ctx.lineWidth=4;ctx.strokeStyle='#fff';ctx.fillStyle=col;
   const yy=Math.max(32,Y(balance(x,l)[key])-10);ctx.strokeText(label,X(x),yy);ctx.fillText(label,X(x),yy);
 }
 line(ctx,[[X(t),Y(0)],[X(t),Y(b.total)]],colors.muted,1,[4,4]);
 dot(ctx,X(t),Y(b.loss),colors.loss);dot(ctx,X(t),Y(b.penalty),colors.reg);dot(ctx,X(t),Y(b.total),colors.total);dot(ctx,X(b.optimum),Y(balance(b.optimum,l).total),colors.total,4);
 const span=.45;line(ctx,[[X(t-span),Y(b.loss-b.gLoss*span)],[X(t+span),Y(b.loss+b.gLoss*span)]],colors.loss,4.5);
 line(ctx,[[X(t-span),Y(b.penalty-b.gReg*span)],[X(t+span),Y(b.penalty+b.gReg*span)]],colors.reg,4.5,[7,4]);
 drawGradientBalance(b,t);
 $('lambda-value').textContent=number(l);$('theta-value').textContent=number(t,4);
 $('g-loss').textContent=number(b.gLoss);$('g-reg').textContent=number(b.gReg);$('g-net').textContent=number(b.gLoss+b.gReg);
 $('balance-status').textContent=Math.abs(b.gLoss+b.gReg)<1e-10?'At the optimum, the slopes cancel.':'The net gradient is nonzero. Move toward the balance.';
}
function drawGradientBalance(b,theta,canvasId='gradient-chart',fixedScale=null){
 const {ctx,w}=canvas(canvasId);
 const left=w<500?125:180,right=w-24,center=(left+right)/2;
 const max=fixedScale??Math.max(1,Math.abs(b.gLoss),Math.abs(b.gReg),Math.abs(b.gLoss+b.gReg));
 const scale=(right-left)/2/max*.9;
 const rows=[['Loss ∇L',b.gLoss,colors.loss],['Penalty ∇R',b.gReg,colors.reg],['Net',b.gLoss+b.gReg,colors.total]];
 line(ctx,[[center,10],[center,143]],'#bbbbbb',1,[2,3]);
 rows.forEach(([label,value,color],i)=>{
   const y=28+i*48;
   ctx.font='600 13px system-ui';ctx.textAlign='left';ctx.fillStyle=color;ctx.fillText(label,0,y-4);
   ctx.font='13px system-ui';ctx.fillText((value>0?'+':'')+number(value,2),0,y+14);
   const end=center+value*scale;
   if(Math.abs(value)<1e-10){ctx.beginPath();ctx.arc(center,y,4,0,Math.PI*2);ctx.fillStyle=color;ctx.fill();if(i===2){ctx.font='12px system-ui';ctx.fillText('balanced',center+10,y+4);}return;}
   line(ctx,[[center,y],[end,y]],color,i===2?3:4,color===colors.reg?[8,5]:[]);
   const direction=Math.sign(value);ctx.beginPath();ctx.moveTo(end,y);ctx.lineTo(end-direction*10,y-6);ctx.lineTo(end-direction*10,y+6);ctx.closePath();ctx.fillStyle=color;ctx.fill();
   ctx.beginPath();ctx.arc(center,y,3,0,Math.PI*2);ctx.fill();
 });
}
$('lambda').addEventListener('input',()=>{balAnimation++;drawBalance();});
$('theta').addEventListener('input',()=>{balAnimation++;balanceTheta=+$('theta').value;drawBalance();});
$('find-balance').onclick=()=>{const token=++balAnimation,start=balanceTheta,end=balance(0,+$('lambda').value).optimum,time=performance.now(),duration=matchMedia('(prefers-reduced-motion: reduce)').matches?0:650;function step(now){if(token!==balAnimation)return;const u=duration?Math.min(1,(now-time)/duration):1;balanceTheta=u===1?end:start+(end-start)*(1-(1-u)**3);$('theta').value=balanceTheta;drawBalance();if(u<1)requestAnimationFrame(step);}requestAnimationFrame(step);};
$('balance-chart').addEventListener('pointerdown',e=>{balAnimation++;const r=e.currentTarget.getBoundingClientRect();balanceTheta=Math.max(-1,Math.min(3,balAxes.unX(e.clientX-r.left)));$('theta').value=balanceTheta;drawBalance();e.currentTarget.setPointerCapture(e.pointerId);});
$('balance-chart').addEventListener('pointermove',e=>{if(e.buttons){const r=e.currentTarget.getBoundingClientRect();balanceTheta=Math.max(-1,Math.min(3,balAxes.unX(e.clientX-r.left)));$('theta').value=balanceTheta;drawBalance();}});
$('random-balance').onclick=()=>{balAnimation++;balanceTheta=randomBetween(-1,3);$('theta').value=balanceTheta;drawBalance();};
// In the inverse problem theta is observed; only lambda moves during estimation.
let observedTheta=+$('observed-theta').value,inverseLambda=+$('inverse-lambda').value,inverseAnimation=0,inverseLimit=1.2;
function inverseEstimate(){return ridgeEstimate([observedTheta],[observedTheta-2]);}
function drawInverse(){
 const t=observedTheta,l=inverseLambda,b=balance(t,l),fit=inverseEstimate();
 const limit=Math.max(inverseLimit,(fit.lambda??0)*1.5);
 $('inverse-lambda').max=limit;
 const v=canvas('inverse-chart'),{ctx}=v;
 const {X,Y}=axes(v,{xmin:-1,xmax:3,ymin:0,ymax:Math.max(5,9*limit+1),ylabel:'Objective value'});
 for(const [key,col] of [['loss',colors.loss],['penalty',colors.reg],['total',colors.total]])
   line(ctx,Array.from({length:201},(_,i)=>{const x=-1+i*.02;return[X(x),Y(balance(x,l)[key])];}),col,key==='total'?3.5:2,key==='penalty'?[7,5]:[]);
 line(ctx,[[X(t),Y(0)],[X(t),Y(Math.max(5,9*limit+1))]],colors.ink,1.5,[3,3]);
 const opt=b.optimum;
 line(ctx,[[X(opt),Y(0)],[X(opt),Y(balance(opt,l).total)]],colors.total,1,[6,4]);
 dot(ctx,X(opt),Y(balance(opt,l).total),colors.total,6);
 dot(ctx,X(t),Y(b.loss),colors.loss);dot(ctx,X(t),Y(b.penalty),colors.reg);
 for(const [height,gradient,color,dash] of [[b.loss,b.gLoss,colors.loss,[]],[b.penalty,b.gReg,colors.reg,[7,4]]]){
   line(ctx,[[X(t-.35),Y(height-gradient*.35)],[X(t+.35),Y(height+gradient*.35)]],color,4,dash);
 }
 ctx.font='12px system-ui';ctx.fillStyle=colors.ink;ctx.textAlign=t>2?'right':'left';ctx.fillText(`Fixed θ̂ = ${number(t,2)}`,X(t)+(t>2?-6:6),40);
 ctx.textAlign='left';ctx.fillStyle=colors.total;ctx.fillText('Total minimum',Math.min(v.w-100,Math.max(55,X(opt)+8)),Y(balance(opt,l).total)-12);
 const e=canvas('inverse-error-chart'),error=x=>(t-2+2*x*t)**2;
 const ea=axes(e,{xmin:0,xmax:limit,ymin:0,ymax:Math.max(1,error(0),error(limit))*1.08,xlabel:'λ',ylabel:'Squared net gradient'});
 line(e.ctx,Array.from({length:151},(_,i)=>{const x=limit*i/150;return[ea.X(x),ea.Y(error(x))];}),colors.ink,2);
 if(fit.lambda!==null)dot(e.ctx,ea.X(fit.lambda),ea.Y(error(fit.lambda)),colors.total,6);
 dot(e.ctx,ea.X(l),ea.Y(error(l)),colors.reg,6);
 e.ctx.fillStyle=colors.reg;e.ctx.font='12px system-ui';e.ctx.textAlign='left';e.ctx.fillText('Current λ',Math.max(55,Math.min(e.w-80,ea.X(l)+8)),Math.max(35,ea.Y(error(l))-12));
 const gradientScale=Math.max(1,Math.abs(b.gLoss),2*limit*Math.abs(t),Math.abs(b.gLoss+2*limit*t));
 drawGradientBalance(b,t,'inverse-gradient-chart',gradientScale);
 $('observed-value').textContent=number(t,3);$('inverse-lambda-value').textContent=number(l,4);
 $('inverse-current').textContent=number(l,4);$('inverse-error').textContent=error(l)<1e-20?'0':error(l).toExponential(2);$('inverse-minimum').textContent=number(opt,4);
 $('estimate-lambda').disabled=fit.lambda===null;
 $('inverse-status').textContent=fit.lambda===null?'At θ̂ = 0 the penalty gradient is zero for every λ. The coefficient is unidentifiable; the loss gradient cannot be canceled.':Math.abs(b.gLoss+b.gReg)<1e-10?'The gradients cancel: the total minimum now matches the fixed observed weight.':Math.abs(l-fit.lambda)<1e-10?'Best nonnegative fit reached, but a residual remains. Ridge cannot explain this observed weight.':'Adjust λ or select “Estimate λ”. The observed weight and its loss gradient remain fixed.';
}
$('observed-theta').oninput=()=>{inverseAnimation++;observedTheta=+$('observed-theta').value;inverseLimit=Math.max(1.2,inverseLambda);drawInverse();};
$('inverse-lambda').oninput=()=>{inverseAnimation++;inverseLambda=+$('inverse-lambda').value;drawInverse();};
$('inverse-extended').onchange=()=>{
 inverseAnimation++;const extended=$('inverse-extended').checked;
 $('observed-theta').min=extended?-1:.25;$('observed-theta').max=extended?3:2;
 observedTheta=Math.max(extended?-1:.25,Math.min(extended?3:2,observedTheta));$('observed-theta').value=observedTheta;drawInverse();
};
$('estimate-lambda').onclick=()=>{
 const end=inverseEstimate().lambda;if(end===null)return;
 const token=++inverseAnimation,start=inverseLambda,time=performance.now(),duration=matchMedia('(prefers-reduced-motion: reduce)').matches?0:900;
 function step(now){if(token!==inverseAnimation)return;const u=duration?Math.min(1,(now-time)/duration):1;inverseLambda=u===1?end:start+(end-start)*(1-(1-u)**3);$('inverse-lambda').value=inverseLambda;drawInverse();if(u<1)requestAnimationFrame(step);}
 requestAnimationFrame(step);
};
$('random-inverse').onclick=()=>{inverseAnimation++;inverseLambda=randomBetween(0,+$('inverse-lambda').max||4);$('inverse-lambda').value=inverseLambda;drawInverse();};

let point=[1.5,.9],landscapeAnimation=0,contourAxes,yaw=-.61,pitch=.7;
function config(){return {l:+$('lambda2').value,a:+$('angle').value*Math.PI/180,s:+$('stretch').value};}
function visibleLandscapes(){return [
 {key:'loss',label:'Loss L',color:colors.loss,dash:[],center:[1.5,.9]},
 {key:'penalty',label:'Penalty R',color:colors.reg,dash:[5,4],center:[0,0]},
 {key:'total',label:'Total L + R',color:colors.total,dash:[],center:optimum2D(config().l,config().a,config().s)}
].filter(layer=>$('show-'+layer.key).checked);}
function drawLandscape(){
 const {l,a,s}=config(),opt=optimum2D(l,a,s);
 const v=canvas('contour-chart'),{ctx}=v;contourAxes=axes(v,{xmin:-2,xmax:3,ymin:-2,ymax:3,xlabel:'θ₁',ylabel:'θ₂',equalAspect:true});const {X,Y}=contourAxes;
 ctx.save();ctx.beginPath();ctx.rect(X(-2),Y(3),X(3)-X(-2),Y(-2)-Y(3));ctx.clip();
 const layers=visibleLandscapes(),alpha=+$('landscape-alpha').value;
 $('landscape-alpha-value').textContent=number(alpha,2);
 ctx.globalAlpha=alpha;
 for(const layer of layers)for(const level of [.1,.35,.8,1.5,2.7,4.5,7,10,15,22]){
   const h1=layer.key==='penalty'?2*l:1+(layer.key==='total'?2*l:0),h2=layer.key==='penalty'?2*l:s+(layer.key==='total'?2*l:0);
   if(h1===0)continue; // A zero penalty is flat: no nonzero contour levels.
   const pts=Array.from({length:161},(_,i)=>{const t=i/160*Math.PI*2,u=Math.sqrt(2*level/h1)*Math.cos(t),vv=Math.sqrt(2*level/h2)*Math.sin(t);return[X(layer.center[0]+Math.cos(a)*u-Math.sin(a)*vv),Y(layer.center[1]+Math.sin(a)*u+Math.cos(a)*vv)];});
   line(ctx,pts,layer.color,layer.key==='total'?1.5:1,layer.dash);
 }
 ctx.globalAlpha=1;
 const b=landscape(...point,l,a,s),sum=b.gLoss.map((n,i)=>n+b.gReg[i]);
 const from=[X(point[0]),Y(point[1])],scale=.6;
 for(const [g,color] of [[b.gLoss,colors.loss],[b.gReg,colors.reg],[sum,colors.total]])arrow(ctx,from,[X(point[0]+scale*g[0]),Y(point[1]+scale*g[1])],color);
 dot(ctx,X(1.5),Y(.9),colors.loss,4);dot(ctx,X(0),Y(0),colors.reg,4);dot(ctx,X(opt[0]),Y(opt[1]),colors.total,5);dot(ctx,...from,colors.ink,4);ctx.restore();
 $('lambda2-value').textContent=number(l);$('stretch-value').textContent=number(s,1)+'×';$('angle-value').textContent=number(a*180/Math.PI,0)+'°';$('point-x').value=point[0];$('point-y').value=point[1];
 $('vector-status').textContent=`‖∇L + ∇R‖ = ${number(Math.hypot(...sum),3)}`;
 const atOptimum=Math.hypot(...sum)<1e-10;
 $('snap2').disabled=atOptimum;$('snap2').textContent=atOptimum?'At the optimum':'Go to optimum';
 $('point-status').textContent=`Selected: (${number(point[0],3)}, ${number(point[1],3)}) · Optimum: (${number(opt[0],3)}, ${number(opt[1],3)})`;
 $('vector-detail').textContent=atOptimum?'The two gradients cancel at the combined minimum.':'The arrows do not cancel here. Descent follows the negative net gradient.';
 drawSurface();
}
function drawSurface(){
 const {ctx,w,h}=canvas('surface-chart'),{l,a,s}=config(),layers=visibleLandscapes();
 const z=(x,y,key='total')=>landscape(x,y,l,a,s)[key];
 // Keep one height scale across layers and toggles so overlays are comparable.
 const maxZ=Math.max(...[[-2,-2],[-2,3],[3,-2],[3,3]].map(p=>z(...p)));
 const raw=(x,y,height)=>{x-=.5;y-=.5;const xx=Math.cos(yaw)*x-Math.sin(yaw)*y,yy=Math.sin(yaw)*x+Math.cos(yaw)*y;return [xx,Math.sin(pitch)*yy-Math.cos(pitch)*height/maxZ*5];};
 const grid=[];for(let i=0;i<=24;i++)for(let j=0;j<=24;j++){const x=-2+5*i/24,y=-2+5*j/24;grid.push(raw(x,y,z(x,y)));}
 const floor=[[-2,-2],[-2,3],[3,3],[3,-2],[-2,-2]];
 const bounds=grid.concat(floor.map(p=>raw(...p,0)),[raw(-2,-2,maxZ)]);
 const xs=bounds.map(p=>p[0]),ys=bounds.map(p=>p[1]),xmin=Math.min(...xs),xmax=Math.max(...xs),ymin=Math.min(...ys),ymax=Math.max(...ys);
 const scale=Math.min((w-80)/(xmax-xmin),(h-70)/(ymax-ymin));
 const project=(x,y,height)=>{const p=raw(x,y,height);return [w/2+(p[0]-(xmin+xmax)/2)*scale,32+(p[1]-ymin)*scale];};
 line(ctx,floor.map(p=>project(...p,0)),'#b0b0b0',1);
 ctx.globalAlpha=+$('landscape-alpha').value;
 for(const layer of layers)for(let i=0;i<=12;i++){
   const q=-2+5*i/12;
   for(const transpose of [false,true])line(ctx,Array.from({length:65},(_,j)=>{const t=-2+5*j/64,x=transpose?t:q,y=transpose?q:t;return project(x,y,z(x,y,layer.key));}),layer.color,layer.key==='total'?1.3:1,layer.dash);
 }
 ctx.globalAlpha=1;
 line(ctx,[project(-2,-2,0),project(-2,-2,maxZ)],'#888',1);
 ctx.fillStyle=colors.muted;ctx.textAlign='left';ctx.font='11px system-ui';
 for(const [label,p] of [['θ₁',project(3,-2,0)],['θ₂',project(-2,3,0)],['Value',project(-2,-2,maxZ)]])ctx.fillText(label,p[0]+5,p[1]);
 for(const [index,layer] of layers.entries()){
   const selected=project(...point,z(...point,layer.key)),minimum=project(...layer.center,z(...layer.center,layer.key));
   line(ctx,[project(...point,0),selected],layer.color,1,[4,4]);
   if(layer.key!=='penalty'||l>0)dot(ctx,...minimum,layer.color,5);
   dot(ctx,...selected,colors.ink,4);
   ctx.font='11px system-ui';ctx.strokeStyle='#fff';ctx.lineWidth=4;ctx.fillStyle=layer.color;
   const readout=`${layer.label}: ${number(z(...point,layer.key),3)}`;ctx.strokeText(readout,12,18+index*16);ctx.fillText(readout,12,18+index*16);
 }
 ctx.font='11px system-ui';ctx.fillStyle=colors.muted;ctx.fillText(layers.length?'Black dots: selected weights on each surface':'Select a landscape above to show it.',12,h-10);
}
for(const id of ['show-loss','show-penalty','show-total'])$(id).onchange=drawLandscape;
$('landscape-alpha').oninput=drawLandscape;
for(const id of ['lambda2','stretch','angle'])$(id).addEventListener('input',()=>{landscapeAnimation++;drawLandscape();});
$('snap2').onclick=()=>{
 const token=++landscapeAnimation,start=[...point],{l,a,s}=config(),end=optimum2D(l,a,s),time=performance.now();
 const duration=matchMedia('(prefers-reduced-motion: reduce)').matches?0:800;
 function step(now){if(token!==landscapeAnimation)return;const u=duration?Math.min(1,(now-time)/duration):1,ease=1-(1-u)**3;point=u===1?end:start.map((v,i)=>v+(end[i]-v)*ease);drawLandscape();if(u<1)requestAnimationFrame(step);}
 requestAnimationFrame(step);
};
$('random-landscape').onclick=()=>{landscapeAnimation++;point=[randomBetween(-2,3),randomBetween(-2,3)];drawLandscape();};
$('reset2').onclick=()=>{landscapeAnimation++;point=[1.5,.9];drawLandscape();};
$('contour-chart').onclick=e=>{const r=e.currentTarget.getBoundingClientRect();point=[Math.max(-2,Math.min(3,contourAxes.unX(e.clientX-r.left))),Math.max(-2,Math.min(3,contourAxes.unY(e.clientY-r.top)))];landscapeAnimation++;drawLandscape();};
for(const id of ['point-x','point-y'])$(id).oninput=()=>{point=[+$('point-x').value,+$('point-y').value];landscapeAnimation++;drawLandscape();};
$('camera').oninput=()=>{yaw=+$('camera').value*Math.PI/180;drawSurface();};
let drag=null;$('surface-chart').onpointerdown=e=>{drag={x:e.clientX,y:e.clientY,yaw,pitch};e.currentTarget.setPointerCapture(e.pointerId);};
$('surface-chart').onpointermove=e=>{if(!drag)return;yaw=drag.yaw+(e.clientX-drag.x)*.009;pitch=Math.max(.25,Math.min(1.3,drag.pitch+(e.clientY-drag.y)*.005));$('camera').value=((yaw*180/Math.PI+180)%360+360)%360-180;drawSurface();};
$('surface-chart').onpointerup=$('surface-chart').onpointercancel=()=>{drag=null;};
let initSeed=7;
function networkConfig(){return {width:+$('network-width').value,activation:$('network-activation').value};}
function lockNetworkControls(locked){for(const id of ['seed','penalty-choice','network-width','network-activation'])$(id).disabled=locked;}
let data=dataset(7),curve=[],worker=null,state='ready',lastStats=null;
function drawNetwork(){
 const v=canvas('network-chart'),{ctx}=v,{X,Y}=axes(v,{xmin:-1.6,xmax:1.6,ymin:-1.2,ymax:1.2,xlabel:'Input x',ylabel:'Prediction f(x)'});
 line(ctx,Array.from({length:151},(_,i)=>{const x=-1.5+i/50;return[X(x),Y(target(x))];}),colors.muted,1.5,[5,4]);
 for(const p of data)dot(ctx,X(p.x),Y(p.y),'#777777',3);
 if(curve.length)line(ctx,curve.map(p=>[X(p.x),Y(p.y)]),colors.total,2.5);
}
function initialCurve(){const {width,activation}=networkConfig(),model=createModel(initSeed,width,activation);$('network-summary').textContent=`1 input → ${width} ${activation==='relu'?'ReLUs':'tanh units'} → 1 output · ${3*width+1} parameters`;$('init-seed-value').textContent=initSeed;curve=Array.from({length:151},(_,i)=>({x:-1.5+i/50,y:predict(model,-1.5+i/50)}));}
function updateStats(stats){
 lastStats=stats;curve=stats.curve;drawNetwork();
 $('estimate').textContent=number(stats.estimate.lambda,4);$('explained').textContent=stats.estimate.explained===null?'—':number(stats.estimate.explained*100,1)+'%';
 $('explained-meter').style.width=(Math.max(0,stats.estimate.explained??0)*100)+'%';$('train-loss').textContent=number(stats.loss,4);$('stationarity').textContent=stats.stationary.toExponential(2);
 $('training-bar').style.width=stats.epoch/stats.limit*100+'%';
 if(state==='running')$('train-status').textContent=`Training · ${stats.epoch.toLocaleString()} / ${stats.limit.toLocaleString()} full-batch steps`;
 if(stats.done){state='done';$('train').textContent='Train again';$('train-status').textContent=`Finished ${stats.epoch.toLocaleString()} steps.`;lockNetworkControls(false);}
 updateRecovery();
}
function updateRecovery(){const truth=+$('penalty-choice').value;$('true-lambda').textContent=number(truth,4);if(lastStats?.estimate.lambda!=null){const error=Math.abs(lastStats.estimate.lambda-truth)/truth*100;$('recovery-note').textContent=`Relative recovery error: ${number(error,1)}%.`;}}
function reset(){worker?.terminate();worker=null;state='ready';lastStats=null;lockNetworkControls(false);data=dataset(+$('seed').value);initialCurve();for(const id of ['estimate','explained','train-loss','stationarity'])$(id).textContent='—';updateRecovery();$('train').textContent='Train the network';$('train-status').textContent='';$('training-bar').style.width=$('explained-meter').style.width='0%';$('recovery-note').textContent='';drawNetwork();}
$('train').onclick=()=>{
 if(state==='running'){worker.postMessage({type:'pause'});state='paused';$('train').textContent='Resume training';$('train-status').textContent='Paused.';return;}
 if(state==='paused'){state='running';worker.postMessage({type:'resume'});$('train').textContent='Pause training';return;}
 reset();
 try{worker=new Worker(new URL('./train-worker.mjs?v=2',import.meta.url),{type:'module'});worker.onmessage=({data})=>updateStats(data);worker.onerror=()=>{state='error';worker?.terminate();$('train-status').textContent='Training could not start. Try a modern browser with module-worker support, served over HTTP(S).';$('train').textContent='Try again';lockNetworkControls(false);};state='running';lockNetworkControls(true);$('train').textContent='Pause training';$('train-status').textContent='Starting training…';worker.postMessage({type:'start',seed:+$('seed').value,initSeed,...networkConfig(),lambda:+$('penalty-choice').value});}
 catch(error){state='error';lockNetworkControls(false);$('train-status').textContent=`Training unavailable: ${error.message}`;}
};
$('reset').onclick=reset;for(const id of ['seed','penalty-choice','network-width','network-activation'])$(id).onchange=reset;
$('random-network').onclick=()=>{initSeed=(initSeed+1+Math.floor(Math.random()*999999))%1000000;reset();};
const drawEarlyStopping=initEarlyStopping({canvas,axes,line,colors,number});
let resizeFrame;new ResizeObserver(()=>{cancelAnimationFrame(resizeFrame);resizeFrame=requestAnimationFrame(()=>{drawBalance();drawInverse();drawLandscape();drawNetwork();drawEarlyStopping();});}).observe(document.querySelector('main'));
initialCurve();updateRecovery();drawBalance();drawInverse();drawLandscape();drawNetwork();
