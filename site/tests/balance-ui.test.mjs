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
    let value=({lambda:.5,lambda2:.5,stretch:3,angle:34})[id]??1;
    const node={textContent:'',style:{},listeners:{},hidden:false,
      addEventListener(event,fn){this.listeners[event]=fn;},
      getBoundingClientRect(){return {width:800,height:350,left:0,top:0};},
      getContext(){return context;},setAttribute(){},setPointerCapture(){},
      get value(){return String(value);},
      set value(input){value=(id==='theta'||id==='lambda')?Math.round(Number(input)*100)/100:Number(input);},
    };
    nodes.set(id,node);return node;
  }
  const source=readFileSync(new URL('../app.mjs',import.meta.url),'utf8');
  const controller=source.slice(source.indexOf('const $='),source.indexOf(withLandscape?'let data=':'let point='));
  const scope=vm.createContext({document:{getElementById:element},window:{devicePixelRatio:1},balance,ridgeEstimate,landscape,optimum2D,
    performance:{now:()=>0},matchMedia:()=>({matches:false}),requestAnimationFrame:fn=>frames.push(fn)});
  vm.runInContext(controller,scope);
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
