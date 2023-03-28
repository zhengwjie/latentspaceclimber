// 绘制条形图
// 绘制双条形图
var total_width=240;
// var scale = d3.scaleOrdinal(d3.schemeTableau10);
var scale=d3.interpolateBlues;
var scalarWidth=d3.scaleLinear().domain([0, 1]).range([24, 81]);
function get_len(d){
    str=d3.format(".0%")(d);
    
    if(str.length==2){
        return 95+scalarWidth(d)-15.5;
    }else{
        return 95+scalarWidth(d)-21.5;
    }
}

function draw_bar1(feature_values){
    // scale.domain(feature_values);
    var svg=d3.select("#feature_vector_view");
    svg.selectAll("g").remove();
    svg
    .append("g")
    .attr("id","feature_value_bar")
    .selectAll("rect")
    .data(feature_values)
    .join("rect")
    .attr("index",(d,i)=>i)
    .attr("x", (d,i) => 109)
    .attr("fill", "#CDCDCD")
    .attr("y", (d,i) => i*30)
    .attr("width", (d,i)=>scalarWidth(d))
    .style("opacity",1)
    .attr("height", 20);

    svg.append("g")
    .selectAll("text")
    .data(feature_values)
    .join("text")
    .attr("x",(d)=>get_len(d))
    .attr("y",(d,i) => 15+i*30)
    .text(d=>d3.format(".0%")(d))
    .attr("font-size",12)
    .attr("fill","white");

    var axis=svg.append("g");
    var numbers=[0,1,2,3,4,5,6,7,8,9];
    var ranges=[];
    for(let i=0;i<10;++i){
        ranges.push(i*30);
    }
    var ordinal_scalar=d3.scaleOrdinal()
    .domain(numbers)
    .range(ranges)
    ;
    var ordinal_axis=d3.axisLeft(ordinal_scalar).tickPadding(0);
    axis.attr("transform", "translate(" + 109 + "," + 10 + ")")
    .call(ordinal_axis);
    var axisRight=axis.append("g").attr("id","axisRight");
    var ordinal_axisRight=d3.axisRight(ordinal_scalar).tickPadding(0);
    axisRight.attr("transform", "translate(" + 91 + "," + 10 + ")")
    .call(ordinal_axisRight);
    svg.selectAll("g path").attr("stroke","block");
    svg.selectAll("#axisRight text").attr("fill","none");
}
function drwa_lr_axis(len,mark_number){
    var numbers=[];
    for(let i=0;i<len;++i){
        numbers.push(i);
    }
    var axis_id="left_axis";
    if(mark_number==1){
        axis_id="right_axis";
    }
    var svg=d3.select("#feature_vector_view");
    svg.select("#"+axis_id).remove();
    var axis=svg.append("g").attr("id",axis_id);
    var ranges=[];
    for(let i=0;i<len;++i){
        ranges.push(i*30);
    }
    var ordinal_scalar=d3.scaleOrdinal()
    .domain(numbers)
    .range(ranges)
    ;
    if(mark_number==0){
        var axisRight=d3.axisRight(ordinal_scalar).tickPadding(0);
        axis.attr("transform", "translate(" + 81 + "," + 10 + ")")
        .call(axisRight);
    }else{
        var ordinal_axis=d3.axisLeft(ordinal_scalar).tickPadding(0);
        axis.attr("transform", "translate(" + 99 + "," + 10 + ")")
        .call(ordinal_axis);
    }
    svg.selectAll("g path").attr("stroke","block");
    // svg.selectAll("#axisRight text").attr("fill","none");

}

function draw_axis_directions(max_value){
    console.log(max_value);
    var numbers=[];
    for(let i=0;i<max_value;++i){
        numbers.push(i);
    }
    var svg=d3.select("#feature_vector_view");
    svg.select("#axis").remove();
    svg.select("#axisRight").remove();

    var axis=svg.append("g").attr("id","axis");
    var ranges=[];
    for(let i=0;i<max_value;++i){
        ranges.push(i*30);
    }
    var ordinal_scalar=d3.scaleOrdinal()
    .domain(numbers)
    .range(ranges)
    ;
    var ordinal_axis=d3.axisLeft(ordinal_scalar).tickPadding(0);
    axis.attr("transform", "translate(" + 109 + "," + 10 + ")")
    .call(ordinal_axis);
    var axisRight=svg.append("g").attr("id","axisRight");
    var ordinal_axisRight=d3.axisRight(ordinal_scalar).tickPadding(0);
    axisRight.attr("transform", "translate(" + 91 + "," + 10 + ")")
    .call(ordinal_axisRight);
    svg.selectAll("g path").attr("stroke","block");
    svg.selectAll("#axisRight text").attr("fill","none");
}
var max_value=0;

function darw_accmulate_bar(accumulating_contribute_rate,mark_number){
    // scale.domain(accumulating_contribute_rate);
    var svg=d3.select("#feature_vector_view");
    // if(accumulating_contribute_rate.length>max_value){
    //     max_value=accumulating_contribute_rate.length;
    //     draw_axis_directions(max_value);
    // }
    if(mark_number==0){
        svg.select("#left_accumulate_bar").remove();
        svg.select("#left_values").remove();
        svg.append("g")
        .attr("id","left_accumulate_bar")
        .selectAll("rect")
        .data(accumulating_contribute_rate)
        .join("rect")
        .attr("index",(d,i)=>i)
        .attr("x",(d,i)=>81-scalarWidth(d))
        .attr("y",(d,i) => i*30)
        .attr("width",(d,i)=>scalarWidth(d))
        .attr("fill",d=>"#CDCDCD")
        .attr("height",20)
        .style("opacity",1);

        svg.append("g")
        .attr("id","left_values")
        .selectAll("text")
        .data(accumulating_contribute_rate)
        .join("text")
        .attr("x",(d,i)=>81-scalarWidth(d))
        .attr("y",(d,i) => 15+i*30)
        .text(d=>d3.format(".0%")(d))
        .attr("font-size",12)
        .attr("fill","white");
        drwa_lr_axis(accumulating_contribute_rate.length,0);
    }else{
        svg.select("#right_accumulate_bar").remove();
        svg.select("#right_values").remove();
        svg.append("g")
        .attr("id","right_accumulate_bar")
        .selectAll("rect")
        .data(accumulating_contribute_rate)
        .join("rect")
        .attr("index",(d,i)=>i)
        .attr("x",(d,i)=>99)
        .attr("y",(d,i) => i*30)
        .attr("width",(d,i)=>scalarWidth(d))
        .attr("fill",d=>"#CDCDCD")
        .attr("height",20)
        .style("opacity",1)
        ;
        svg.append("g")
        .attr("id","right_values")
        .selectAll("text")
        .data(accumulating_contribute_rate)
        .join("text")
        .attr("x",(d,i)=>get_len(d))
        .attr("y",(d,i) => 15+i*30)
        .text(d=>d3.format(".0%")(d))
        .attr("font-size",12)
        .attr("fill","white");
        drwa_lr_axis(accumulating_contribute_rate.length,1);
    }
    // https://observablehq.com/@d3/bar-chart-race-explained
    update_axis_numbers();
}
function update_axis_numbers(){
    var left_num=d3.selectAll("#left_values text")._groups[0].length;
    var right_num=d3.selectAll("#right_values text")._groups[0].length;
    if(left_num==0 || right_num==0){
        return;
    }
    if(left_num>right_num){
        d3.selectAll("#right_axis text").remove();
    }else{
        d3.selectAll("#left_axis text").remove();
    }
}
function show_accmulate_value(accumulating_contribute_rate){
    var svg=d3.select("#feature_vector_view");
    svg.append("g")
    .selectAll("text")
    .data(accumulating_contribute_rate)
    .join("text")
    .attr("x",(d,i)=>87)
    .attr("y",(d,i) => 15+i*30)
    .text((d,i)=>d3.format(".0%")(d))
    .attr("font-size",12)
    .attr("fill","black");
}

function draw_bar(weights){
    alphas=weights.slice(1,weights.length).trim().split(/\s+/);
    positive_alphas=Array(10);
    negative_alphas=Array(10);
    for(let i=0;i<alphas.length;++i){
        alpha=parseFloat(alphas[i]);
        if(alpha>0){
            positive_alphas[i]=alpha;
            negative_alphas[i]=0;
        }else{
            negative_alphas[i]=-alpha;
            positive_alphas[i]=0;
        }
    }

    var svg=d3.select("#feature_vector_view");
    svg.append("g")
    .attr("id","weights")
    .selectAll("rect")
    .data(negative_alphas)
    .join("rect")
    .attr("index",(d,i)=>i)
    .attr("x",(d,i)=>87-scalarWidth(d))
    .attr("y",(d,i) => i*30)
    .attr("width",(d,i)=>scalarWidth(d))
    .attr("fill",d=>"#CDCDCD")
    .attr("height",20)
    .style("opacity",1)
    ;
    svg.append("g")
    .attr("id","weights")
    .selectAll("rect")
    .data(positive_alphas)
    .join("rect")
    .attr("index",(d,i)=>i)
    .attr("x",(d,i)=>103)
    .attr("y",(d,i) => i*30)
    .attr("width",(d,i)=>scalarWidth(d))
    .attr("fill",d=>"#CDCDCD")
    .attr("height",20)
    .style("opacity",1)
    ;
}

// 颜色需要变化
// 需要画坐标轴
