import {rng,ridgeEstimate} from './math.mjs?v=2';
const dot=(a,b)=>a.reduce((s,v,i)=>s+v*b[i],0);
// Construct an actual full-rank regression dataset with a known eigensystem.
// L = ||X theta - y||²/(2n), so Hessian eigenvalues are h (no factor of two).
export function regressionProblem(seed=7,condition=25){
 const p=10,n=160,random=rng(seed),normal=()=>Math.sqrt(-2*Math.log(Math.max(random(),1e-12)))*Math.cos(2*Math.PI*random());
 const V=Array.from({length:p},(_,j)=>Array.from({length:p},(_,k)=>Math.sqrt((k===0?1:2)/p)*Math.cos(Math.PI*(j+.5)*k/p)));
 const h=Array.from({length:p},(_,k)=>2/condition**(k/(p-1))),beta=Array.from({length:p},()=>normal());
 const X=Array.from({length:n},(_,i)=>Array.from({length:p},(_,j)=>h.reduce((sum,hk,k)=>sum+Math.sqrt(2*hk)*Math.cos(Math.PI*(i+.5)*(k+1)/n)*V[j][k],0)));
 const y=X.map(row=>dot(row,beta)+.3*normal()),b=Array.from({length:p},(_,j)=>X.reduce((sum,row,i)=>sum+row[j]*y[i]/n,0));
 const target=h.map((hk,k)=>V.reduce((sum,row,j)=>sum+row[k]*b[j],0)/hk);
 return {p,n,X,y,V,h,target,rate:.2};
}
export function fromEigen(problem,z){return problem.V.map(row=>dot(row,z));}
export function regressionAt(problem,step,start=Array(10).fill(0),family='directional'){
 const {h,target,rate}=problem;
 const z=h.map((hk,k)=>target[k]+(start[k]-target[k])*(1-rate*hk)**step),g=h.map((hk,k)=>hk*(z[k]-target[k]));
 const scalar=ridgeEstimate(z,g),directional=z.map((v,k)=>Math.abs(v)<1e-12?null:Math.max(0,-g[k]/(2*v)));
 const penalties=family==='scalar'?z.map(()=>scalar.lambda??0):directional.map(v=>v??0);
 // Independently solve the explicitly regularized normal equations.
 const recovered=h.map((hk,k)=>hk*target[k]/(hk+2*penalties[k]));
 const residual=g.map((v,k)=>v+2*penalties[k]*z[k]),g2=dot(g,g);
 const weights=fromEigen(problem,z),fitted=fromEigen(problem,recovered);
 return {z,g,weights,fitted,scalar,directional,penalties,explained:g2<1e-20?null:1-dot(residual,residual)/g2,residual:Math.hypot(...residual),weightError:Math.hypot(...weights.map((v,k)=>v-fitted[k])),unidentified:family==='directional'?directional.some(v=>v===null):scalar.lambda===null};
}
export function randomRegressionStart(seed){const random=rng(seed);return Array.from({length:10},()=>1.2*(random()-.5));}
