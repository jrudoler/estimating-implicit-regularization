// The teaching examples use R(theta) = lambda * ||theta||^2 throughout.
export function balance(theta, lambda) {
  const loss = 0.5 * (theta - 2) ** 2;
  return { loss, penalty: lambda * theta ** 2, total: loss + lambda * theta ** 2,
    gLoss: theta - 2, gReg: 2 * lambda * theta, optimum: 2 / (1 + 2 * lambda) };
}
export function landscape(x, y, lambda, angle = 0.6, stretch = 3) {
  const c = Math.cos(angle), s = Math.sin(angle), dx = x - 1.5, dy = y - 0.9;
  const u = c * dx + s * dy, v = -s * dx + c * dy;
  const loss = 0.5 * (u * u + stretch * v * v);
  return { loss, penalty: lambda * (x*x+y*y), total: loss + lambda*(x*x+y*y),
    gLoss: [c*u-s*stretch*v,s*u+c*stretch*v], gReg: [2*lambda*x,2*lambda*y] };
}
export function optimum2D(lambda, angle = 0.6, stretch = 3) {
  const c = Math.cos(angle), s = Math.sin(angle);
  const a=c*c+stretch*s*s, b=c*s*(1-stretch), d=s*s+stretch*c*c;
  const hx=a*1.5+b*0.9, hy=b*1.5+d*0.9, det=(a+2*lambda)*(d+2*lambda)-b*b;
  return [((d+2*lambda)*hx-b*hy)/det, ((a+2*lambda)*hy-b*hx)/det];
}
export function ridgeEstimate(weights, gradient) {
  let dot=0, norm=0, gnorm=0;
  for(let i=0;i<weights.length;i++){ dot+=weights[i]*gradient[i]; norm+=weights[i]**2; gnorm+=gradient[i]**2; }
  if(norm < 1e-16) return { lambda: null, explained: null, residual: Math.sqrt(gnorm), gnorm: Math.sqrt(gnorm) };
  const lambda=Math.max(0,-dot/(2*norm));
  let residual=0;
  for(let i=0;i<weights.length;i++) residual+=(gradient[i]+2*lambda*weights[i])**2;
  return { lambda, explained: gnorm < 1e-20 ? null : 1-residual/gnorm, residual: Math.sqrt(residual), gnorm: Math.sqrt(gnorm) };
}
export function rng(seed) {
  let state=seed>>>0;
  return () => { state+=0x6D2B79F5; let t=state; t=Math.imul(t^(t>>>15),t|1); t^=t+Math.imul(t^(t>>>7),t|61); return ((t^(t>>>14))>>>0)/4294967296; };
}
export const target = x => 0.65*Math.sin(2.8*x)+0.25*x;
export function dataset(seed=7,n=64) {
  const random=rng(seed);
  return Array.from({length:n},(_,i)=>{ const x=-1.5+3*i/(n-1); return {x,y:target(x)+(random()+random()+random()-1.5)*0.16}; });
}
// A 1 -> width -> 1 MLP with ReLU or tanh activation. Layout: input weights, hidden biases,
// output weights, output bias. All parameters, including biases, are penalized.
export function createModel(seed=7,width=16,activation='relu') {
  if(!['relu','tanh'].includes(activation)) throw new Error('Unsupported activation');
  const random=rng(seed+1000), weights=new Float64Array(3*width+1);
  for(let j=0;j<width;j++){ weights[j]=(random()*2-1)*0.8; weights[width+j]=(random()*2-1)*0.8; weights[2*width+j]=(random()*2-1)*0.3; }
  return {width,weights,activation};
}
export function predict(model,x) {
  const {width:h,weights:w}=model; let y=w[3*h];
  for(let j=0;j<h;j++) y+=w[2*h+j]*(model.activation==='tanh'?Math.tanh(w[j]*x+w[h+j]):Math.max(0,w[j]*x+w[h+j]));
  return y;
}
export function lossGradient(model,data) {
  const {width:h,weights:w}=model, gradient=new Float64Array(w.length); let loss=0;
  for(const {x,y} of data){
    const error=predict(model,x)-y; loss+=error*error/data.length;
    const delta=2*error/data.length; gradient[3*h]+=delta;
    for(let j=0;j<h;j++) { const z=w[j]*x+w[h+j],a=model.activation==='tanh'?Math.tanh(z):Math.max(0,z),derivative=model.activation==='tanh'?1-a*a:(z>0?1:0); gradient[2*h+j]+=delta*a; gradient[j]+=delta*w[2*h+j]*derivative*x; gradient[h+j]+=delta*w[2*h+j]*derivative; }
  }
  return {loss,gradient};
}
export function trainSteps(model,data,lambda,steps,rate=0.04*16/model.width) {
  for(let step=0;step<steps;step++) {
    const {gradient}=lossGradient(model,data);
    for(let i=0;i<model.weights.length;i++) model.weights[i]-=rate*(gradient[i]+2*lambda*model.weights[i]);
  }
}
export function diagnostics(model,data,lambda) {
  const {loss,gradient}=lossGradient(model,data), estimate=ridgeEstimate(model.weights,gradient);
  let stationary=0; for(let i=0;i<gradient.length;i++) stationary+=(gradient[i]+2*lambda*model.weights[i])**2;
  const curve=Array.from({length:151},(_,i)=>{ const x=-1.5+i/50; return {x,y:predict(model,x)}; });
  return { loss, estimate, stationary:Math.sqrt(stationary), curve };
}
