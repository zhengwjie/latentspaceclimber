var overview_row_number = 100;
var contour_length=500;
var scale_number = contour_length/overview_row_number;
var overview_linear_x;
var overview_linear_y;
function draw_contour(data) {
//    绘制一个contour
    console.log(data);
    console.log("hhhh");
    alert("hhh");
    var grid = data.novelty_scores;
    var num_x=data.num_x;
    var num_y=data.num_y;

    overview_linear_x=d3.scaleLinear().domain([data.min_x,data.max_x]).range([0,contour_length]);
    overview_linear_y=d3.scaleLinear().domain([data.min_y,data.max_y]).range([0,contour_length]);

    var min_value = d3.min(grid);
    var max_value = d3.max(grid);
    var mainGroup = d3.select("#interpret_images");
    var color = ["#fee5d9", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"];
    var line_color=["black","rgb(247, 26, 26)","rgb(59, 59, 248)","green"]
    // rgb(247, 26, 26) 是红色的

    var fill_color=["#ffffff",
        "#f0f0f0",
        "#d9d9d9","#bdbdbd"]
    thresholds = Array([])
    console.log(min_value);
    console.log(max_value);
    // for (let i = 1; i <= 2; ++i) {
    //     thresholds.push(min_value + (2*i+3) * (max_value - min_value) /10 );
    // }

    thresholds.push(min_value+0.4*(max_value-min_value));
    thresholds.push(min_value+0.65*(max_value-min_value));

    // var scale_number=len_axis/row_interpret_images;
    var contours = d3.contours()
        .size([num_x, num_y])
        .thresholds(thresholds)
        (grid);

    var contour = d3.select("#novelty_contour");
    console.log(contours);
    var path = d3.geoPath();
    contour.selectAll("path")
        .data(contours)
        .enter()
        .append("path")
        .attr("d", path)
        // .attr('fill', (d, i) => color[i])
        .attr("stroke", function(d,i) { return line_color[i]; })
        .attr("fill", (d, i) => fill_color[i])
        .attr("stroke-width",0.2)
        .attr("transform", ` translate(0,0) scale(${scale_number})`);
}
var index;

function draw_image_list(data){
    var image_list=document.getElementById("explored_images_list");
    image_list.innerHTML="";
    for(let i=0;i<data.length;++i){
        img_info=data[i];
        var li=document.createElement("li");
        var image=document.createElement("img");
        var div=document.createElement("div");
        div.setAttribute("width",100);
        div.setAttribute("height",100);
        image.setAttribute("src","data:image/png;base64,"+img_info.image);
        image.setAttribute("width",80);
        image.setAttribute("height",80);
        div.style.paddingBottom="10px";
        div.appendChild(image);
        li.appendChild(div);
        image_list.appendChild(li);
    }
    var overview_svg=d3.select("#novelty_contour");
//    在这个里面绘制几个点
    overview_svg.selectAll("circle").remove();
    overview_svg
        .selectAll("circle")
        .enter()
        .data(data)
        .join("circle")
        .attr("cx",d=>overview_linear_x(+d.point_x))
        .attr("cy",d=>overview_linear_y(+d.point_y))
        .attr("r","2px")
        .attr("fill","black");
      const markerBoxWidth = 6;
      const markerBoxHeight = 6;
      const refX = markerBoxWidth / 2;
      const refY = markerBoxHeight / 2;
      const markerWidth = markerBoxWidth / 2;
      const markerHeight = markerBoxHeight / 2;
      const arrowPoints = [[0, 0], [0, 6], [6, 3]];
      overview_svg
        .append('defs')
        .append('marker')
        .attr('id', 'arrow')
        .attr('viewBox', [0, 0, markerBoxWidth, markerBoxHeight])
        .attr('refX', refX)
        .attr('refY', refY)
        .attr('markerWidth', markerBoxWidth)
        .attr('markerHeight', markerBoxHeight)
        .attr('orient', 'auto-start-reverse')
        .append('path')
        .attr('d', d3.line()(arrowPoints))
        .attr('stroke', 'black');
        var prev_point_x=+data[0].point_x;
        var prev_point_y=+data[0].point_y;
        overview_svg.selectAll("line").remove();
    for(let i=1;i<data.length;++i){
        var current_point_x=+data[i].point_x;
        var current_point_y=+data[i].point_y;
        var line = overview_svg.append("line")
            .attr("x1",overview_linear_x(prev_point_x))
            .attr("y1",overview_linear_y(prev_point_y))
            .attr("x2",overview_linear_x(current_point_x))
            .attr("y2",overview_linear_y(current_point_y))
            .attr("stroke","black")
            .attr("stroke-width",1)
            .attr("marker-end","url(#arrow)");
        prev_point_x=current_point_x;
        prev_point_y=current_point_y;
    }

}
function update_exploration_list(){
    //don't need to post any parameters
    $.post("/look_history_z",{
    },function (data,status){
        draw_image_list(data.history_info);
    })

}
function novelty_overview() {
    update_exploration_list();
    // window.open("/index", "newwindow", "height=100, width=400, toolbar= no, menubar=no, scrollbars=no, resizable=yes, location=no, status=no");
    // layer.msg('hello');
    //在打开novelty_overview的时候，需要提前更新一下历史记录

    index=layer.open({
        title: ["Latent Space Overview","height:30px;line-height:30px;padding-left:10px;border-bottom-color:rgb(176, 169, 169);" +
        "font-family:serif;" +
        "font-weight:bold;" +
        "font-size:20px"],
        type: 1,
        content: $("#novelty_overview") //这里content是一个普通的String
    });
    // layui.use(['layer', 'form'], function(){
    //     var layer = layui.layer;
    //     var form = layui.form;

    //     layer.msg('Hello World');
    //   });
}

function overview_mouseup(e){
    if(e.button==0){
        var overview_svg=d3.select("#novelty_contour");
        overview_svg.selectAll("image").remove();
    }
}

function overview_mousedown(e){
    // alert("mousedown");
    if(e.button==0){
        [x,y]=d3.pointer(e);
        true_x=overview_linear_x.invert(x);
        true_y=overview_linear_y.invert(y);
    //  需要根据这一个坐标，获取一个点
        $.post("/novelty_overview_select",{
            point_x:true_x,
            point_y:true_y
        },function (data,status){
            var overview_svg=d3.select("#novelty_contour");
            overview_svg.append("image")
                  .attr("xlink:href","data:image/png;base64,"+data.image)
                  .attr("width",50)
                  .attr("height",50)
                  .attr("y",y)
                  .attr("x",x);
        })
    }
}
//点击进入下一步
function overview_right_click(e){
    e.preventDefault();
    console.log(e.button);
    [x,y]=d3.pointer(e);

    true_x=overview_linear_x.invert(x);
    true_y=overview_linear_y.invert(y);
    $.post("/novelty_overview_select",{
        point_x:true_x,
        point_y:true_y
    },function (data,status){
        var overview_svg=d3.select("#novelty_contour");
        overview_svg.append("image")
              .attr("xlink:href","data:image/png;base64,"+data.image)
              .attr("width",50)
              .attr("height",50)
              .attr("y",y)
              .attr("x",x);
    })
    $.post("/novelty_overview_start_exploration",{
        "point_x":true_x,
        "point_y":true_y
    },function (data,status){
        update_axis();
        show(data.svd_images);
        draw_bar(data.feature_values);
        darw_accmulate_bar(data.accumulating_contribute_rate);
        show_neighbor_images(data.neighbor_images);
        deactivate_next_step();
        update_exploration_list();
        // draw_overview();
        // layer.close(index);
    })
    var overview_svg=d3.select("#novelty_contour");
    overview_svg.selectAll("image").remove();
    // layer.close(index);
}
