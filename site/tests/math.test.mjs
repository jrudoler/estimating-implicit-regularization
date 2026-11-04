import test from 'node:test';
import assert from 'node:assert/strict';
import {balance,landscape,optimum2D,ridgeEstimate,createModel,dataset,lossGradient,trainSteps,diagnostics} from '../math.mjs';
const near=(actual,expected,tol=1e-8)=>assert.ok(Math.abs(actual-expected)<tol,`${actual} != ${expected}`);
test('the analytic 1D optimum cancels gradients and recovers the penalty',()=>{
 for(const lambda of [0,.01,.5,2]){const t=balance(0,lambda).optimum,b=balance(t,lambda);near(b.gLoss+b.gReg,0);near(ridgeEstimate([t],[b.gLoss]).lambda,lambda);}
});
test('rotated anisotropic quadratics have the advertised optimum',()=>{
 for(const lambda of [0,.3,2])for(const angle of [0,.6,2.8])for(const stretch of [1,3,6]){
 const p=optimum2D(lambda,angle,stretch),b=landscape(...p,lambda,angle,stretch);
 b.gLoss.forEach((x,i)=>near(x+b.gReg[i],0));
 for(const delta of [[.01,0],[0,.01],[-.01,-.01]])assert.ok(landscape(p[0]+delta[0],p[1]+delta[1],lambda,angle,stretch).total>b.total);
 }
});
test('ridge projection handles boundaries and nonidentifiable inputs',()=>{
 near(ridgeEstimate([1,2],[-2,-4]).lambda,1);near(ridgeEstimate([1,2],[-2,-4]).explained,1);
 near(ridgeEstimate([1],[1]).lambda,0);near(ridgeEstimate([1],[1]).explained,0);
 assert.equal(ridgeEstimate([0],[1]).lambda,null);assert.equal(ridgeEstimate([1],[0]).explained,null);near(ridgeEstimate([1],[0]).lambda,0);
});
test('manual MLP backprop agrees with central finite differences',()=>{
 const m=createModel(7),d=dataset(7,13),analytic=lossGradient(m,d).gradient,eps=1e-6;
 for(let i=0;i<m.weights.length;i++){const old=m.weights[i];m.weights[i]=old+eps;const plus=lossGradient(m,d).loss;m.weights[i]=old-eps;const minus=lossGradient(m,d).loss;m.weights[i]=old;near(analytic[i],(plus-minus)/(2*eps),1e-7);}
});
test('seeded data and network initialization reproduce exactly',()=>{
 assert.deepEqual(dataset(19),dataset(19));assert.deepEqual(createModel(19),createModel(19));assert.notDeepEqual(dataset(19),dataset(7));
});
test('all exposed seed and penalty presets recover the explicit coefficient',()=>{
 for(const seed of [7,19,42])for(const lambda of [.005,.015,.05]){
 const m=createModel(seed),d=dataset(seed),before=lossGradient(m,d).loss;
 trainSteps(m,d,lambda,20000);const result=diagnostics(m,d,lambda);
 assert.ok(result.loss<before,`loss did not decrease: seed ${seed}`);
 assert.ok(Math.abs(result.estimate.lambda-lambda)/lambda<.05,`poor recovery: seed ${seed}, lambda ${lambda}, estimate ${result.estimate.lambda}`);
 assert.ok(result.estimate.explained>.95,`poor gradient fit: seed ${seed}, lambda ${lambda}`);
 }
});
