import test from 'node:test';import assert from 'node:assert/strict';
import {regressionProblem,regressionAt,fromEigen,randomRegressionStart,penaltyMatrix,theoreticalPenalty} from '../early-stopping-math.mjs';
const near=(a,b,tol=1e-9)=>assert.ok(Math.abs(a-b)<tol,`${a} != ${b}`);
test('spectral training matches actual unregularized GD on the generated regression data',()=>{
 for(const condition of [1,25,100]){
  const p=regressionProblem(7,condition),start=randomRegressionStart(19);let w=fromEigen(p,start);
  for(let t=1;t<=40;t++){
   const g=Array(10).fill(0);
   p.X.forEach((row,i)=>{const e=row.reduce((sum,x,j)=>sum+x*w[j],0)-p.y[i];row.forEach((x,j)=>g[j]+=x*e/p.n);});
   w=w.map((v,j)=>v-p.rate*g[j]);const s=regressionAt(p,t,start);s.weights.forEach((v,j)=>near(v,w[j]));
  }
 }
});
test('zero-start directional estimates recover weights and match early-stopping theory',()=>{
 for(const condition of [1,25,100])for(const seed of [7,19,42]){
  const p=regressionProblem(seed,condition);
  for(const t of [1,2,20,100,400]){
   const s=regressionAt(p,t);assert.ok(s.weightError<1e-9);assert.ok(s.residual<1e-9);
   s.directional.forEach((lambda,k)=>{const q=(1-p.rate*p.h[k])**t;near(lambda,p.h[k]*q/(2*(1-q)));});
  }
 }
});
test('scalar ridge recovers isotropic stopping but misses anisotropic stopping',()=>{
 assert.ok(regressionAt(regressionProblem(7,1),20,undefined,'scalar').weightError<1e-9);
 const s=regressionAt(regressionProblem(7,25),20,undefined,'scalar');assert.ok(s.weightError>.01);assert.ok(s.explained<.99);
});
test('zero-start penalties decrease and arbitrary starts can be infeasible',()=>{
 const p=regressionProblem();let prev=regressionAt(p,1);
 for(const t of [2,10,20,100,400]){const next=regressionAt(p,t);next.directional.forEach((v,j)=>assert.ok(v<=prev.directional[j]+1e-12));prev=next;}
 const s=regressionAt(p,1,p.target.map(v=>2*v));assert.ok(s.weightError>.01);assert.ok(s.directional.every(v=>v===0));
 assert.equal(regressionAt(p,0).unidentified,true);
});

test('penalty matrices preserve gradients in original coordinates and agree with independent theory',()=>{
 const p=regressionProblem();
 for(const t of [1,20,400]){
  const s=regressionAt(p,t),A=penaltyMatrix(p,s.penalties),expected=penaltyMatrix(p,theoreticalPenalty(p,t));
  A.forEach((row,i)=>row.forEach((value,j)=>{near(value,expected[i][j]);near(value,A[j][i]);}));
  const gradient=A.map(row=>2*row.reduce((sum,a,j)=>sum+a*s.weights[j],0));
  fromEigen(p,s.g).forEach((value,i)=>near(value+gradient[i],0));
 }
 const ridge=penaltyMatrix(p,Array(10).fill(.2));ridge.forEach((row,i)=>row.forEach((v,j)=>near(v,i===j?.2:0)));
});

test('original-coordinate diagonal recovers weights without recovering theory',async()=>{
 const {estimateRegularizer}=await import('../early-stopping-math.mjs');
 const p=regressionProblem(7,5),s=estimateRegularizer(p,20,Array(10).fill(0),'diagonal'),theory=penaltyMatrix(p,theoreticalPenalty(p,20));
 assert.ok(s.convex);assert.ok(s.weightError<1e-9);
 assert.ok(Math.hypot(...s.A.flatMap((row,i)=>row.map((v,j)=>v-theory[i][j])))>1e-5);
 s.A.forEach((row,i)=>row.forEach((v,j)=>{if(i!==j)near(v,0);}));
 const nonconvex=estimateRegularizer(regressionProblem(7,25),20,Array(10).fill(0),'diagonal');
 assert.ok(nonconvex.residual<1e-9);assert.equal(nonconvex.convex,false);assert.equal(nonconvex.fitted,null);
});
test('full symmetric estimator learns the theoretical matrix from independent endpoints',async()=>{
 const {estimateRegularizer}=await import('../early-stopping-math.mjs');
 for(const condition of [1,5,25,100]){
  const pool=Array.from({length:12},(_,i)=>regressionProblem(7+i,condition)),p=pool[0];
  for(const t of [1,20,100,400]){
   const s=estimateRegularizer(p,t,Array(10).fill(0),'full',pool),theory=penaltyMatrix(p,theoreticalPenalty(p,t));
   assert.equal(s.rank,10);assert.ok(s.weightError<1e-7);
   s.A.forEach((row,i)=>row.forEach((v,j)=>near(v,theory[i][j],1e-7)));
  }
 }
});
test('insufficient endpoints match observations but do not identify the full penalty',async()=>{
 const {estimateRegularizer}=await import('../early-stopping-math.mjs');
 const p=regressionProblem(7,5),s=estimateRegularizer(p,20,Array(10).fill(0),'full',[p]),theory=penaltyMatrix(p,theoreticalPenalty(p,20));
 assert.equal(s.rank,1);assert.ok(s.residual<1e-8);
 assert.ok(Math.hypot(...s.A.flatMap((row,i)=>row.map((v,j)=>v-theory[i][j])))>1e-5);
});
