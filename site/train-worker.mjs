import {createModel,dataset,trainSteps,diagnostics} from './math.mjs';
let run=null, generation=0;
function tick(token){
  if(!run || run.paused || token!==generation) return;
  const steps=Math.min(100,run.limit-run.epoch);
  trainSteps(run.model,run.data,run.lambda,steps);
  run.epoch+=steps;
  const stats=diagnostics(run.model,run.data,run.lambda);
  const done=run.epoch>=run.limit;
  postMessage({type:'progress',epoch:run.epoch,limit:run.limit,done,...stats});
  if(!done) setTimeout(()=>tick(token),0);
}
onmessage=({data})=>{
  if(data.type==='start'){
    generation++;
    run={model:createModel(data.seed),data:dataset(data.seed),lambda:data.lambda,epoch:0,limit:20000,paused:false};
    tick(generation);
  } else if(data.type==='pause' && run){run.paused=true;}
  else if(data.type==='resume' && run){run.paused=false;generation++;tick(generation);}
  else if(data.type==='reset'){generation++;run=null;}
};
