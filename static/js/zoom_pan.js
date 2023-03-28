

function change_domain(scale_axis,g,axis,scale_number){
    domain=scale_axis.domain();
    start=domain[0]*scale_number;
    end=domain[1]*scale_number;
    scale_axis.domain([start,end]);
    axis.scale().domain([start,end]);
    g.call(axis);
    return end;
}

function zoom_right(scale_number){
    // 获取domain_x
    // 和domain_y
    scale_axis=right_scalex;
    g=right_gx;
    axis=right_axisx;
    end=change_domain(scale_axis,g,axis,scale_number);
    scale_axis=right_scaley;
    g=right_gy;
    axis=right_axisy;
    end=change_domain(scale_axis,g,axis,scale_number);

    // end 是新的range 
    // 请求数据，然后更新图片
    $.post("/zoom",{
        domain_range:end,
        scale:scale_number,
        svg_mark:1
    },function(data,status){
        svg=d3.select("#right_svg");
        show(svg,data.svd_images,end);
        if(selected_svg=='right_svg' && selected_index!=null){
            // 需要重新请求数据
            // 更新信息面板
            // 更新另一个界面的数据
            // 其实就是触发选中按钮
            var image=document.querySelectorAll("#right_svg image")[selected_index];
            image.addEventListener('click',image_click,false);
            var event = new Event('click');
            image.dispatchEvent(event);
            image.removeEventListener('click',image_click,false);
            // alert("hhh");
        }
        // 如果有选中的点，并且选中的点在当前界面中则需要修改
    })
    

}

function zoom_left(scale_number){
    // 获取domain_x
    // 和domain_y

    scale_axis=left_scalex;
    g=left_gx;
    axis=left_axisx;
    end=change_domain(scale_axis,g,axis,scale_number);
    scale_axis=left_scaley;
    g=left_gy;
    axis=left_axisy;

    end=change_domain(scale_axis,g,axis,scale_number);
    // end 是修改后的range
    $.post("/zoom",{
        domain_range:end,
        scale:scale_number,
        svg_mark:0
    },function(data,status){
        svg=d3.select("#left_svg");
        show(svg,data.svd_images,end);
        if(selected_svg=='left_svg' && selected_index!=null){
            // 需要重新请求数据
            // 更新信息面板
            // 更新另一个界面的数据
            // 其实就是触发选中按钮
            var image=document.querySelectorAll("#left_svg image")[selected_index];
            image.addEventListener('click',image_click,false);
            var event = new Event('click');
            image.dispatchEvent(event);
            image.removeEventListener('click',image_click,false);
            
        }

    })



    // // 请求数据
    // $.post("/api/svd_images",
    // {domain_x:domain_x,
    //  domain_y:domain_y,
    //  row_number:row_number
    // },
    // function(data,status){
    // //   展示图片，不更新控制面板
    
    //   show(data.svd_images);
    // })
}
//  在这里实现zoom和pan的逻辑
function left_zoomin(){
    zoom_left(0.8);
}

function left_zoomout(){
    zoom_left(1.25); 
}

//  在这里实现zoom和pan的逻辑
function right_zoomin(){
    zoom_right(0.8);
}

function right_zoomout(){
    zoom_right(1.25); 
}

  // prevent scrolling then apply the default filter
function filter(event) {
    event.preventDefault();
    // 禁止滚动
    if(event.type === 'wheel'){
        return false;
    }
    return (!event.ctrlKey || event.type === 'wheel') && !event.button;
  }
var left_original_domain_x;
var left_original_domain_y;
// 开始平移
function left_pan_start(e){

    d3.selectAll("#svd_images image")
    .attr("transform", "translate(" + e.transform.x + "," + e.transform.y + ")");
    d3.selectAll("#svd_images rect")
    .attr("transform", "translate(" + e.transform.x + "," + e.transform.y + ")");
    d3.selectAll("#svd_images #center_logo")
    .attr("transform", "translate(" + e.transform.x + "," + e.transform.y + ")");
    d3.selectAll("#svd_images .click_logo")
    .attr("transform", "translate(" + e.transform.x + "," + e.transform.y + ")");

    left_original_domain_x=left_scalex.domain();
    left_original_domain_y=left_scaley.domain();
    left_gx.call(left_axisx.scale(e.transform.rescaleX(left_scalex)));
    left_gy.call(left_axisy.scale(e.transform.rescaleY(left_scaley)));
}
// 平移结束
function left_pan_end(e){

    x_scale=left_original_domain_x;
    y_scale=left_original_domain_y;
    var split_number=row_number-1;
    console.log(x_scale);
    console.log(y_scale);
    dis_x=(x_scale[1]-x_scale[0])/split_number;
    dis_y=(y_scale[1]-y_scale[0])/split_number;

    domainX=e.transform.rescaleX(left_scalex).domain();
    domainY=e.transform.rescaleY(left_scaley).domain();
    shift_x=domainX[0]-x_scale[0];
    shift_y=domainY[0]-y_scale[0];

    start_domainX=x_scale[0]+Math.floor(shift_x/dis_x)*dis_x;
    start_domainY=y_scale[0]+Math.floor(shift_y/dis_y)*dis_y;
    final_domainX=[start_domainX,start_domainX+dis_x*split_number];
    final_domainY=[start_domainY,start_domainY+dis_y*split_number];
    
    left_scalex.domain(final_domainX);
    left_scaley.domain(final_domainY);
    left_axisx.scale().domain(final_domainX);
    left_axisy.scale().domain(final_domainY);
    left_gx.call(left_axisx);
    left_gy.call(left_axisy);

    $.post("/api/svd_images",
    {domain_x:final_domainX.toString(),
     domain_y:final_domainY.toString(),
     row_number:row_number
    },
    function(data,status){
      show(data.svd_images);
    //   draw_bar(data.feature_values);
    //   darw_accmulate_bar(data.accumulating_contribute_rate);
    })
}

var right_original_domain_x;
var right_original_domain_y;
// 开始平移
function right_pan_start(e){
   
    d3.selectAll("#pure_image image")
    .attr("transform", "translate(" + e.transform.x + "," + e.transform.y + ")");
    d3.selectAll("#mainsvg rect")
    .attr("transform", "translate(" + e.transform.x + "," + e.transform.y + ")");

    d3.selectAll("#mainsvg svg")
    .attr("transform", "translate(" + e.transform.x + "," + e.transform.y + ")");

    d3.selectAll("#mainsvg #center_logo")
    .attr("transform", "translate(" + e.transform.x + "," + e.transform.y + ")");
    d3.selectAll("#mainsvg .click_logo")
    .attr("transform", "translate(" + e.transform.x + "," + e.transform.y + ")");
    
    right_original_domain_x=right_scalex.domain();
    right_original_domain_y=right_scaley.domain();

    right_gx.call(right_axisx.scale(e.transform.rescaleX(right_scalex)));
    right_gy.call(right_axisy.scale(e.transform.rescaleY(right_scaley)));
}
// 平移结束
function right_pan_end(e){

    x_scale=right_original_domain_x;
    y_scale=right_original_domain_y;
    var split_number=row_number-1;
    dis_x=(x_scale[1]-x_scale[0])/split_number;
    dis_y=(y_scale[1]-y_scale[0])/split_number;

    domainX=e.transform.rescaleX(right_scalex).domain();
    domainY=e.transform.rescaleY(right_scaley).domain();
    shift_x=domainX[0]-x_scale[0];
    shift_y=domainY[0]-y_scale[0];

    start_domainX=x_scale[0]+Math.floor(shift_x/dis_x)*dis_x;
    start_domainY=y_scale[0]+Math.floor(shift_y/dis_y)*dis_y;
    final_domainX=[start_domainX,start_domainX+dis_x*split_number];
    final_domainY=[start_domainY,start_domainY+dis_y*split_number];
    
    right_scalex.domain(final_domainX);
    right_scaley.domain(final_domainY);
    right_axisx.scale().domain(final_domainX);
    right_axisy.scale().domain(final_domainY);
    right_gx.call(right_axisx);
    right_gy.call(right_axisy);

    domainX=e.transform.rescaleX(right_scalex).domain();
    domainY=e.transform.rescaleY(right_scaley).domain();
   
    $.post("/zoom_pan_right_images",
    {domain_x:final_domainX.toString(),
     domain_y:final_domainY.toString()
    },
    function(data,status){
        only_show_pure_images(data.images);
    })
}



// 目前左边的放大和缩小都实现了
// 还缺一个平移的功能
var pan_stat=0;
// 0表示未选中
function left_pan(){
    pan_stat=1-pan_stat;
    console.log(pan_stat);
    var left_svg=d3.select("#svd_images");
    var icon=document.getElementsByName("expand-outline")[0];
    // 如果未选中
    if(pan_stat==0){
        icon.style.color="#b3b1b1";
        var zoom=d3.zoom();
        left_svg.call(zoom);
    }else{
        // 选中了之后呢
        icon.style.color="#777676";
        var zoom=d3.zoom()
        .scaleExtent([0.1,5])
        .filter(filter)
        .on("zoom",left_pan_start)
        .on("end",left_pan_end);
        left_svg.call(zoom);
    }
}

// 查看前一步
function prev_step(){
    clear_env();
    $.post("/prev_step",
    function(data,status){
        update_axis();
        show(data.svd_images);
        draw_bar(data.feature_values);
        darw_accmulate_bar(data.accumulating_contribute_rate);
        show_neighbor_images(data.neighbor_images);
    })
}

// 查看后一步
function next_step(){
    clear_env();
    $.post("/look_next_step",
    function(data,status){
        update_axis();
        show(data.svd_images);
        draw_bar(data.feature_values);
        darw_accmulate_bar(data.accumulating_contribute_rate);
        show_neighbor_images(data.neighbor_images);
    })
}
// 刚开始是没有激活的状态
// zoom in 0
// zoom out 1
// pan  2
// next step 3
var btn_status=[0,0,0,0];
var previous_activated_button=0;
var colors=["#575555","#a9a4a4"];
var global_node;
function change_button_color(node,color){
    for(var i=0;i<node.children.length;++i){
        node.children[i].setAttribute("style","fill:"+color);
    }
}
function get_node_index(node){
    var index=node.getAttribute("name");
    return parseInt(index);
}

function switch_colors(node){
    if(previous_activated_button){
        previous_index=get_node_index(previous_activated_button);
        btn_status[previous_index]=0;
        change_button_color(previous_activated_button,colors[0]);
    }
    if(previous_activated_button==node){
        previous_activated_button=0;
        return 0;
    }
    previous_activated_button=node;
    // 激活下一步的事件
    change_button_color(previous_activated_button,colors[1]);
    index=get_node_index(previous_activated_button);
    btn_status[index]=1;
    return 1;
}

function next_setp(){
    if(selected_image_left){
        selected_image_left.setAttribute("opacity",0.5);
    }
    if(selected_image_right){
        selected_image_right.setAttribute("opacity",0.5);
    }
    this.setAttribute("opacity",1);
    clear_env();
    var shift_x=this.getAttribute("shift_x");
    var shift_y=this.getAttribute("shift_y");
    var feature_index_x=this.getAttribute("feature_index_x");
    var feature_index_y=this.getAttribute("feature_index_y");

    // 开始发送请求
    $.post("/next_step",{
        shift_x:shift_x,
        shift_y:shift_y,
        feature_index_x:feature_index_x,
        feature_index_y:feature_index_y,
        row_number:row_number
    },function(data,status){
        // 显示图片
        // 并更新特征值
        // 并且把右边的界面清空
        console.log(data);
        update_axis();
        show(data.svd_images);
        draw_bar(data.feature_values);
        darw_accmulate_bar(data.accumulating_contribute_rate);
    })
}

function activate_image_click(){
    var right_images=document.querySelectorAll("#left_svg image");
    for(let i=0;i<right_images.length;++i){
        var image_node=right_images[i];
        image_node.addEventListener('click',image_click,false);
    }
    var left_images=document.querySelectorAll("#right_svg image");
    for(let i=0;i<left_images.length;++i){
        var image_node=left_images[i];
        image_node.addEventListener('click',image_click,false);
    }
}

function activate_next_step(){
    var right_images=document.querySelectorAll("#left_svg image");
    for(let i=0;i<right_images.length;++i){
        var image_node=right_images[i];
        image_node.removeEventListener('click',image_click,false);
        image_node.addEventListener('click',next_setp,false);
    }
    var left_images=document.querySelectorAll("#right_svg image");
    for(let i=0;i<left_images.length;++i){
        var image_node=left_images[i];
        image_node.removeEventListener('click',image_click,false);
        image_node.addEventListener('click',next_setp,false);
    }
}

function deactivate_image_click_event(){
    var right_images=document.querySelectorAll("#right_svg image");
    for(let i=0;i<right_images.length;++i){
        var image_node=right_images[i];
        image_node.removeEventListener('click',image_click,false);
    }
    var left_images=document.querySelectorAll("#left_svg image");
    for(let i=0;i<left_images.length;++i){
        var image_node=left_images[i];
        image_node.removeEventListener('click',image_click,false);
        
    }
}

var total_status=0;
function change_next_step_status(node){
    var status=switch_colors(node);
    // 如果状态为1，说明已经激活了
    if(status==1){
        // 把image的onclick的事件进行修改
        deactivate_pan();
        deactivate_zoomin();
        deactivate_zoomout();
        activate_next_step();
    }
    else{
        deactivate_next_step();
    }
}

var current_event;
function select_pan(node){
    var status=switch_colors(node);
    var left_svg=d3.select("#svd_images");
    var right_svg=d3.select("#mainsvg");

    if(status==1){
        current_event="pan";
        deactivate_image_click_event();
        deactivate_zoomin();
        deactivate_zoomout();
        var zoom=d3.zoom()
        .scaleExtent([0.1,5])
        .filter(filter)
        .on("zoom",left_pan_start)
        .on("end",left_pan_end);
        left_svg.call(zoom);

        var right_zoom=d3.zoom()
        .scaleExtent([0.1,5])
        .filter(filter)
        .on("zoom",right_pan_start)
        .on("end",right_pan_end);
        right_svg.call(right_zoom);
    }else{
        current_event="";
        deactivate_next_step();
        deactivate_pan();
    }
}

function deactivate_pan(){
    var left_svg=d3.select("#left_svg");
    var right_svg=d3.select("#right_svg");
    var zoom=d3.zoom();
    left_svg.call(zoom);
    right_svg.call(zoom);
}

// zoom 左边和右边激活这个事件
function activate_zoomin(){
    left_svg=document.getElementById("left_svg");
    left_svg.addEventListener('click',left_zoomin,false);
    right_svg=document.getElementById("right_svg");
    right_svg.addEventListener('click',right_zoomin,false);
}
function deactivate_zoomin(){
    left_svg=document.getElementById("left_svg");
    left_svg.removeEventListener('click',left_zoomin,false);
    right_svg=document.getElementById("right_svg");
    right_svg.removeEventListener('click',right_zoomin,false);
}
function activate_zoomout(){
    left_svg=document.getElementById("left_svg");
    left_svg.addEventListener('click',left_zoomout,false);
    right_svg=document.getElementById("right_svg");
    right_svg.addEventListener('click',right_zoomout,false);
}
function deactivate_zoomout(){
    left_svg=document.getElementById("left_svg");
    left_svg.removeEventListener('click',left_zoomout,false);
    right_svg=document.getElementById("right_svg");
    right_svg.removeEventListener('click',right_zoomout,false);
}
// 放大和缩小
function select_zoomin(node){
    var status=switch_colors(node);
    if(status==1){
        global_node=node;
        current_event="zoom";
        deactivate_image_click_event();
        activate_zoomin();
        deactivate_zoomout();
    }else{
        current_event="";
        activate_image_click();
        deactivate_zoomin();
       
    }
}

function select_zoomout(node){
    var status=switch_colors(node);
    if(status==1){
        current_event="zoom";
        global_node=node;
        deactivate_image_click_event();
        activate_zoomout();
        deactivate_zoomin();
    }else{
        current_event="";
        activate_image_click();
        deactivate_zoomout();
    }
}

function recycle(){
    clear_env();
    $.post("/recycle",
    function(data,status){

        clear_env();
        var svg=d3.select("#left_svg");
        show(svg,data.svd_images,data.domain_range);
        // draw_bar(data.feature_values);
        darw_accmulate_bar(data.accumulating_contribute_rate,0);
        // show_accmulate_value(data.accumulating_contribute_rate);

        show_neighbor_images(data.neighbor_images);
        if(current_event!='zoom'){
        activate_image_click();
        }
        mark_left_div();
    })
}
