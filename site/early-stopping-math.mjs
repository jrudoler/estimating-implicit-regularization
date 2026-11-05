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

// R(theta) = theta^T A theta, expressed in the original feature coordinates.
export function penaltyMatrix(problem,coefficients){
 return problem.V.map(row=>problem.V.map(other=>row.reduce((sum,v,k)=>sum+v*coefficients[k]*other[k],0)));
}
// Independent theoretical reference for full-batch GD from zero initialization.
export function theoreticalPenalty(problem,step){
 return problem.h.map(h=>{const q=(1-problem.rate*h)**step;return h*q/(2*(1-q));});
}

const eye=n=>Array.from({length:n},(_,i)=>Array.from({length:n},(_,j)=>+(i===j)));
const mv=(A,x)=>A.map(row=>dot(row,x));
// Jacobi eigensolver for small real symmetric matrices; no theoretical basis is supplied.
export function symmetricEigen(matrix){
 const A=matrix.map(row=>row.slice()),n=A.length,V=eye(n);
 for(let sweep=0;sweep<100*n*n;sweep++){
  let p=0,q=1,max=0;
  for(let i=0;i<n;i++)for(let j=i+1;j<n;j++)if(Math.abs(A[i][j])>max){max=Math.abs(A[i][j]);p=i;q=j;}
  if(max<1e-14*Math.max(1,...A.map((row,i)=>Math.abs(row[i]))))break;
  const angle=.5*Math.atan2(2*A[p][q],A[q][q]-A[p][p]),c=Math.cos(angle),s=Math.sin(angle),ap=A[p][p],aq=A[q][q],off=A[p][q];
  for(let k=0;k<n;k++)if(k!==p&&k!==q){const x=A[k][p],y=A[k][q];A[k][p]=A[p][k]=c*x-s*y;A[k][q]=A[q][k]=s*x+c*y;}
  A[p][p]=c*c*ap-2*s*c*off+s*s*aq;A[q][q]=s*s*ap+2*s*c*off+c*c*aq;A[p][q]=A[q][p]=0;
  for(let k=0;k<n;k++){const x=V[k][p],y=V[k][q];V[k][p]=c*x-s*y;V[k][q]=s*x+c*y;}
 }
 return {values:A.map((row,i)=>row[i]),V};
}
export function fitSymmetric(endpoints){
 const n=endpoints[0].weights.length,C=eye(n).map(row=>row.map(()=>0)),D=C.map(row=>row.slice());
 for(const {weights:w,gradient:g} of endpoints)for(let i=0;i<n;i++)for(let j=0;j<n;j++){C[i][j]+=w[i]*w[j];D[i][j]-=(g[i]*w[j]+w[i]*g[j])/2;}
 const {values:d,V}=symmetricEigen(C),tol=Math.max(...d)*1e-10;
 const rotated=V.map((_,i)=>V.map((_,j)=>D.reduce((sum,row,k)=>sum+V[k][i]*dot(row,V.map(r=>r[j])),0)));
 const B=rotated.map((row,i)=>row.map((v,j)=>d[i]+d[j]>tol?v/(d[i]+d[j]):0));
 const A=V.map(row=>V.map(other=>row.reduce((sum,v,i)=>sum+v*dot(B[i],other),0)));
 return {A,rank:d.filter(v=>v>tol).length};
}
export function endpointAt(problem,step,start=Array(10).fill(0)){
 const z=problem.h.map((h,k)=>problem.target[k]+(start[k]-problem.target[k])*(1-problem.rate*h)**step);
 return {weights:fromEigen(problem,z),gradient:fromEigen(problem,z.map((v,k)=>problem.h[k]*(v-problem.target[k])))};
}
export function estimateRegularizer(problem,step,start,family='diagonal',pool=[problem]){
 const endpoint=endpointAt(problem,step,start),{weights,gradient}=endpoint;
 let A,rank=null,unidentified=false;
 if(family==='full')({A,rank}=fitSymmetric(pool.map(p=>endpointAt(p,step,start))));
 else {
  const scalar=ridgeEstimate(weights,gradient);
  const coefficients=family==='scalar'?weights.map(()=>scalar.lambda??0):weights.map((w,i)=>Math.abs(w)<1e-12?null:-gradient[i]/(2*w));
  unidentified=coefficients.some(x=>x===null)||(family==='scalar'&&scalar.lambda===null);
  A=coefficients.map((v,i)=>coefficients.map((_,j)=>i===j?(v??0):0));
 }
 const H=penaltyMatrix(problem,problem.h),normal=H.map((row,i)=>row.map((v,j)=>v+2*A[i][j])),eig=symmetricEigen(normal),tol=Math.max(...eig.values.map(Math.abs))*1e-10;
 const convex=eig.values.every(v=>v>tol),b=fromEigen(problem,problem.h.map((h,k)=>h*problem.target[k]));
 const fitted=convex?eig.V.map(row=>row.reduce((sum,v,k)=>sum+v*dot(eig.V.map(r=>r[k]),b)/eig.values[k],0)):null;
 const residual=gradient.map((g,i)=>g+2*dot(A[i],weights)),g2=dot(gradient,gradient);
 return {A,rank,unidentified,convex,weights,fitted,residual:Math.hypot(...residual),explained:g2<1e-20?null:1-dot(residual,residual)/g2,weightError:fitted?Math.hypot(...weights.map((v,i)=>v-fitted[i])):null};
}
