let data = [], width = 600, height = 400, numPoints = 100;

let hoveredId;

function updateData() {
    data=[];
    for(let i=0;i<numPoints;++i){
        data.push({
            x:Math.random()*width,
            y:Math.random()*height,
            r:1+Math.random()*30,
            id:i
        })
    }
}

function update() {
    d3.select("#mainsvg")
    .selectAll("circle")
    .data(data)
    .join("circle")
    .attr("cx",function(d){return d.x})
    .attr("cy",function(d){return d.y})
    .attr("r",function(d){return d.r;})
    .style("fill",function(d) {
        return d.id==hoveredId? "red": "orange";
    })
    .on("mouseover",function(d){hoveredId=d.id; update();})
    .on("mouseout",function(d){hoveredId=null; update();})
}

updateData();
update();

