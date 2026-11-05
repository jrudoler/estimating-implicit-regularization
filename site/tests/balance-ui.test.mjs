import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {balance,ridgeEstimate,landscape,optimum2D} from '../math.mjs';

// Exercise the real Figure 1 event handlers with an HTML-range-like value setter.
// Range inputs snap programmatic assignments to their step, too.
function fixture(withLandscape=false) {
  const nodes=new Map(),frames=[];
  const context=new Proxy({}, {get:(obj,key)=>obj[key]??(()=>{}),set:(obj,key,value)=>(obj[key]=value,true)});
  function element(id) {
    if(nodes.has(id))return nodes.get(id);
    let value=({lambda:.5,lambda2:.5,stretch:3,angle:34,'observed-theta':.8,'inverse-lambda':.15})[id]??1;
    const node={textContent:'',style:{},listeners:{},hidden:false,
      addEventListener(event,fn){this.listeners[event]=fn;},
      getBoundingClientRect(){return {width:800,height:350,left:0,top:0};},
      getContext(){return context;},setAttribute(){},setPointerCapture(){},
      get value(){return String(value);},
      set value(input){value=(['theta','lambda','inverse-lambda','observed-theta'].includes(id))?Math.round(Number(input)*100)/100:Number(input);},
    };
    nodes.set(id,node);return node;
  }
  const source=readFileSync(new URL('../app.mjs',import.meta.url),'utf8');
  const controller=source.slice(source.indexOf('const $='),source.indexOf(withLandscape?'let data=':'let point='));
  const scope=vm.createContext({document:{getElementById:element},window:{devicePixelRatio:1},balance,ridgeEstimate,landscape,optimum2D,
    performance:{now:()=>0},matchMedia:()=>({matches:false}),requestAnimationFrame:fn=>frames.push(fn)});
  vm.runInContext(controller,scope);
  vm.runInContext('drawInverse()',scope);
  if(withLandscape)vm.runInContext('drawLandscape()',scope);
  return {element,flush(){while(frames.length)frames.shift()(1000);}};
}

test('Find the balance cancels gradients across every penalty slider value despite range snapping',()=>{
  const ui=fixture();
  for(let step=0;step<=200;step++){
    const lambda=step/100;
    ui.element('lambda').value=lambda;
    ui.element('lambda').listeners.input();
    ui.element('find-balance').onclick();ui.flush();
    assert.equal(ui.element('g-net').textContent,'0.00',`lambda=${lambda}`);
    assert.equal(ui.element('balance-status').textContent,'At the optimum, the slopes cancel.');
    assert.equal(Number(ui.element('g-loss').textContent)+Number(ui.element('g-reg').textContent),0);
    assert.equal(ui.element('theta-value').textContent,(2/(1+2*lambda)).toFixed(4));
  }
});

test('manual slider movement replaces the exact optimum and reports an actual residual',()=>{
  const ui=fixture();
  ui.element('lambda').value=.3;ui.element('lambda').listeners.input();
  ui.element('find-balance').onclick();ui.flush();
  assert.equal(ui.element('theta-value').textContent,'1.2500');
  ui.element('theta').value=1.26;ui.element('theta').listeners.input();
  assert.equal(ui.element('g-net').textContent,'0.02');
  assert.match(ui.element('balance-status').textContent,/nonzero/);
});

test('landscape starts away from optimum, moves exactly, and re-enables after objective changes',()=>{
 const ui=fixture(true),button=ui.element('snap2');
 assert.equal(button.disabled,false);
 assert.match(ui.element('point-status').textContent,/Selected: \(1.500, 0.900\)/);
 button.onclick();ui.flush();
 assert.equal(button.disabled,true);
 assert.equal(button.textContent,'At the optimum');
 assert.equal(ui.element('vector-status').textContent,'‖∇L + ∇R‖ = 0.000');
 const previous=ui.element('point-status').textContent.split(' · ')[0];
 ui.element('lambda2').value=1.23;ui.element('lambda2').listeners.input();
 assert.equal(button.disabled,false);
 assert.equal(ui.element('point-status').textContent.split(' · ')[0],previous);
 button.onclick();ui.flush();assert.equal(button.disabled,true);
 ui.element('reset2').onclick();assert.equal(button.disabled,false);
});

test('inverse estimation holds observed weights fixed and reaches exact gradient cancellation',()=>{
 const ui=fixture();
 for(const theta of [.25,.43,.8,1.27,1.91,2]){
  ui.element('observed-theta').value=theta;ui.element('observed-theta').oninput();
  ui.element('inverse-lambda').value=.18;ui.element('inverse-lambda').oninput();
  const observed=ui.element('observed-value').textContent;
  ui.element('estimate-lambda').onclick();ui.flush();
  assert.equal(ui.element('observed-value').textContent,observed);
  assert.equal(ui.element('inverse-error').textContent,'0');
  assert.equal(ui.element('inverse-minimum').textContent,theta.toFixed(4));
  assert.match(ui.element('inverse-status').textContent,/gradients cancel/);
 }
});

test('inverse estimation reports infeasibility and nonidentifiability without false success',()=>{
 const ui=fixture();ui.element('inverse-extended').checked=true;ui.element('inverse-extended').onchange();
 for(const theta of [-1,2.5]){
  ui.element('observed-theta').value=theta;ui.element('observed-theta').oninput();
  ui.element('estimate-lambda').onclick();ui.flush();
  assert.equal(ui.element('inverse-current').textContent,'0.0000');
  assert.match(ui.element('inverse-status').textContent,/residual remains/);
 }
 ui.element('observed-theta').value=0;ui.element('observed-theta').oninput();
 assert.equal(ui.element('estimate-lambda').disabled,true);
 assert.match(ui.element('inverse-status').textContent,/unidentifiable/);
});

test('random initialization preserves the fixed quantities and cancels pending animations',()=>{
 const ui=fixture(true);
 ui.element('find-balance').onclick();ui.element('random-balance').onclick();
 const theta=ui.element('theta-value').textContent;ui.flush();assert.equal(ui.element('theta-value').textContent,theta);
 assert.equal(ui.element('lambda').value,'0.5');
 const observed=ui.element('observed-value').textContent;
 ui.element('estimate-lambda').onclick();ui.element('random-inverse').onclick();
 const lambda=ui.element('inverse-current').textContent;ui.flush();assert.equal(ui.element('inverse-current').textContent,lambda);
 assert.equal(ui.element('observed-value').textContent,observed);
 ui.element('snap2').onclick();ui.element('random-landscape').onclick();
 const point=ui.element('point-status').textContent;ui.flush();assert.equal(ui.element('point-status').textContent,point);
 assert.equal(ui.element('lambda2').value,'0.5');
});
