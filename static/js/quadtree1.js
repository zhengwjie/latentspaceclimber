let data=[],hoveredId,numPoints=50,width=400,height=600;

let quadTree=d3.quadtree()
  .x(d=>d.x)
  .y(d=>d.y)


function updateData(){
    data=[];
    for(let i=0;i<numPoints;++i){
        data.push({
            x:Math.random()*width,
            y:Math.random()*height,
            id:i,
            r:Math.random()*20+1
        })
    }
    quadTree.addAll(data);
}
function handleMousemove(e){
    let pos=d3.pointer(e,this);
    let d=quadTree.find(pos[0],pos[1],20);
    hoveredId=d?d.id:undefined;
    update();
}
function initEvent(){
    d3.select("svg")
    .on("mousemove",handleMousemove);
}
function update(){
    d3.select("#mainsvg")
    .selectAll("circle")
    .data(data)
    .join("circle")
    .attr('cx',d=>d.x)
    .attr('cy',d=>d.y)
    .attr("r",d=>d.r)
    .style("fill",d=>d.id==hoveredId?'red':"orange")
}
updateData();
update();
initEvent();

