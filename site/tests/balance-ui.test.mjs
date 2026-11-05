import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import vm from 'node:vm';
import {balance,ridgeEstimate} from '../math.mjs';

// Exercise the real Figure 1 event handlers with an HTML-range-like value setter.
// Range inputs snap programmatic assignments to their step, too.
function fixture() {
  const nodes=new Map(),frames=[];
  const context=new Proxy({}, {get:(obj,key)=>obj[key]??(()=>{}),set:(obj,key,value)=>(obj[key]=value,true)});
  function element(id) {
    if(nodes.has(id))return nodes.get(id);
    let value=id==='lambda'?.5:1;
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
  const controller=source.slice(source.indexOf('const $='),source.indexOf('let point='));
  const scope=vm.createContext({document:{getElementById:element},window:{devicePixelRatio:1},balance,ridgeEstimate,
    performance:{now:()=>0},matchMedia:()=>({matches:false}),requestAnimationFrame:fn=>frames.push(fn)});
  vm.runInContext(controller,scope);
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
