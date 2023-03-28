document.write('<script src="/static/js/jquery-2.2.4.min.js"></script>');
var row_number=9; 
var img_size=50;
var distance_imgs=15;//两张图片之间的距离是3px
var padding=30; //给坐标轴留的空隙
var len_axis=distance_imgs*(row_number-1)+img_size*row_number;
var total_size=padding*2+len_axis;
var axis_padding=30;
var transform_x=total_size-axis_padding;
var transform_y=2*padding-axis_padding;
var img_interpret_size=40;
var axis_padding_x=axis_padding+(img_size)/2;
var transform_gy=transform_y+(img_size)/2;
var image_size_plus_distance_imgs=img_size+distance_imgs;
var row_interpret_images=19;
var dis_interpret_images=30;
var selected_image_left;
var selected_image_left_shift_x;
var selected_image_left_shift_y;
var selected_image_left_index;
var selected_image_left_relative_loc_x;
var selected_image_left_relative_loc_y;
var selected_image_right;
var selected_image_right_index;
var selected_image_right_shift_x;
var selected_image_right_shift_y;
var selected_image_right_relative_loc_x;
var selected_image_right_relative_loc_y;
var end_point_x;
var end_point_y;
var right_image_end_relative_loc_x=-8;
var right_image_end_relative_loc_y=-8;

// 记录了当前的探索到了第几步
var global_step=0;
var global_images;
var min_probability;
var max_probability;
var color=d3.interpolateReds;
var color=d3.scaleLinear().domain([0.2,1]).range([color(0),"#cb181d"]);
// var color=d3.interpolateRgb("red", "blue");
// var scalar_color=d3.scaleLinear().domain([0.3, 0.7]).range([0, 1]);
var img_size_scalar=d3.scaleLinear().domain([0, 1]).range([img_size-20, img_size]);
var init_domain=10;
var right_gx;
var right_gy;
var right_scalex;
var right_scaley;
var right_axisx;
var right_axisy;
var selected_svg=null;
var selected_index=null;
var distance_image_border;
// 放大，缩小，平移都不更新控制面板
// 但是需要更改图片大小
function update_global_values(){
  border_distance=580/(13*row_number-1);
  border_size=12*border_distance;
  img_size=5*border_size/6;
  image_size_plus_distance_imgs=border_distance+border_size;
  img_size_scalar=d3.scaleLinear().domain([0, 1]).range([img_size-20, img_size]);
  distance_image_border=(border_size-img_size)/2;
}
function clear_svg(svg){
  svg.select("#center_logo").remove();
  svg.select(".click_logo").remove();
  svg.selectAll(".border").remove();
  svg.selectAll("image").remove();
  svg.selectAll(".axis").remove();
}
// 在一个svg中显示图片
function show(svg,images,domain_range){
  clear_svg(svg);
  drwa_axis(svg,domain_range);
    global_images=images;
    var mainGroup=svg;
    var row_number=Math.floor(Math.sqrt(images.length));

    var container_name=svg.attr("id");

    update_global_values();

    mainGroup.select("#outer_border")
    .selectAll("rect")
    .data(images)
    .join("rect")
    .attr("class","border")
    .attr("y",(d,i)=>-15+(Math.floor(i/row_number)*image_size_plus_distance_imgs))
    .attr("x",((d,i)=>5+((i%row_number)*image_size_plus_distance_imgs)))
    .attr("width",d=>(mark(d)+2*distance_image_border))
    .attr("height",d=>(mark(d)+2*distance_image_border))
    .attr('transform',d=>(`translate(${axis_padding+(img_size-mark(d))/2}, ${transform_y+(img_size-mark(d))/2})`))
    .attr("novelty_score",d=>(+d.novelty_score))
    .attr("fill",d=>color(+d.novelty_score))
    .attr("stroke","grey")
    .attr("container",container_name)
    ;

    mainGroup
    .selectAll("image")
    .data(images)
    .join("image")
    .attr("xlink:href",d=>"data:image/png;base64,"+d.image)
    .attr("width",d=>mark(d))
    .attr("height",d=>mark(d))
    .attr("y",(d,i)=>distance_image_border-15+(Math.floor(i/row_number)*image_size_plus_distance_imgs))
    .attr("x",((d,i)=>distance_image_border+5+((i%row_number)*image_size_plus_distance_imgs)))
    .attr('transform',d=>(`translate(${axis_padding+(img_size-mark(d))/2}, ${transform_y+(img_size-mark(d))/2})`))
    .attr("index",(d,i)=>i)
    .attr("probability",(d)=>d.probability)
    .attr("opacity",1)
    .attr("container",container_name)
    .attr("weights",d=>d.weights)
    ;
    mark_images(svg);
    // mark_central_image();
    // deactivate_next_step();
}

function mark_images(svg){
  var rects=svg.selectAll(".border");
  svg.selectAll("#center_logo").remove();
  svg.selectAll(".click_logo").remove();
  var middle_index=(row_number*row_number-1)/2;
  console.log(middle_index);
  var logo_size=24*9*(1/row_number)
  rects.each(
    (d,i)=>{
      var rect_x=axis_padding+4;
      var rect_y=transform_y+4;

      if(i==middle_index){
        var svg_code=`<svg t="1666411352971" class="icon1" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4387" width="${logo_size}" height="${logo_size}"><path d="M284.458667 941.397333c-36.437333 15.637333-68.48-7.68-64.896-47.168l22.613333-248.917333-164.394667-188.053333c-26.069333-29.824-13.653333-67.562667 24.789334-76.309334l243.370666-55.381333 127.786667-214.677333c20.288-34.090667 59.946667-34.069333 80.213333 0l127.786667 214.677333 243.370667 55.381333c38.656 8.789333 50.858667 46.485333 24.789333 76.309334l-164.394667 188.053333 22.741334 249.002667c3.605333 39.509333-28.458667 62.805333-64.896 47.146666l-229.504-98.517333-229.376 98.453333z" fill="#d94801" stroke="black" stroke-width="100" p-id="4388"></path></svg>`;
        svg.append("g")
        .attr("id","center_logo")
        .html(svg_code)
        .attr("transform",'translate(' + rect_x + ',' + rect_y+ ')')
        ;
        svg.select("#center_logo svg")
        .attr("x",+(rects.nodes()[i].getAttribute("x"))+(img_size-mark(d))/2+mark(d)-10)
        .attr("y",+(rects.nodes()[i].getAttribute("y"))+(img_size-mark(d))/2+mark(d)-10);
        if(selected_index==null){
          var image_node=document.createElement("img");
          image_node.setAttribute("width","100");
          image_node.setAttribute("height","100");
          image_node.setAttribute("src","data:image/png;base64,"+d.image);
          var image_div=document.querySelector("#center_image");
          image_div.innerHTML="";
          image_div.appendChild(image_node);
        }
      }else if(i==selected_index && svg.attr("id")==selected_svg){
        var svg_code=`<svg t="1666413378157" class="icon1" viewBox="-100 -100 1154 1154" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="5525" width="${logo_size}" height="${logo_size}"><path d="M512 1024C229.248 1024 0 794.752 0 512S229.248 0 512 0s512 229.248 512 512-229.248 512-512 512z m-114.176-310.954667a53.333333 53.333333 0 0 0 75.434667 0l323.328-323.328a53.333333 53.333333 0 1 0-75.434667-75.434666l-287.914667 283.306666-128.853333-128.853333a53.333333 53.333333 0 1 0-75.434667 75.434667l168.874667 168.874666z" p-id="5526" fill="#d94801" stroke="black" stroke-width="100"></path></svg>`;
        // var svg_code=`<svg t="1672140628528" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2841" width="200" height="200"><path d="M512 0C230.4 0 0 230.4 0 512s230.4 512 512 512 512-230.4 512-512S793.6 0 512 0zM844.8 364.8l-428.8 428.8L179.2 556.8c-25.6-25.6-25.6-64 0-89.6s64-25.6 89.6 0l147.2 147.2 339.2-339.2c25.6-25.6 64-25.6 89.6 0S870.4 339.2 844.8 364.8z" fill="#272636" p-id="2842"></path></svg>`;
        
        svg.append("g")
          .attr("class","click_logo")
          .html(svg_code)
          .attr("transform",'translate(' + rect_x + ',' + rect_y+ ')');
        svg.select(".click_logo svg")
          .attr("x",+(rects.nodes()[i].getAttribute("x"))+(img_size-mark(d))/2+mark(d)-10)
          .attr("y",+(rects.nodes()[i].getAttribute("y"))+(img_size-mark(d))/2+mark(d)-10);
      }
    }

  );
}
// 左边和右边都只有一个是需要记录选中的图片
// 显示图片
// 标记是否需要选中
// 记录主视图

function mark_central_image(){
  var svg=d3.select("#svd_images");
  var rects=svg.selectAll("rect");
  svg.select("#center_logo").remove();
  svg.select(".click_logo").remove();

  rects.each(
    (d,i)=>{
      var rect_x=axis_padding;
      var rect_y=transform_y;
      if(d.relative_loc_x==0&&d.relative_loc_y==0){
        var svg_code=`<svg t="1666411352971" class="icon1" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4387" width="24" height="24"><path d="M284.458667 941.397333c-36.437333 15.637333-68.48-7.68-64.896-47.168l22.613333-248.917333-164.394667-188.053333c-26.069333-29.824-13.653333-67.562667 24.789334-76.309334l243.370666-55.381333 127.786667-214.677333c20.288-34.090667 59.946667-34.069333 80.213333 0l127.786667 214.677333 243.370667 55.381333c38.656 8.789333 50.858667 46.485333 24.789333 76.309334l-164.394667 188.053333 22.741334 249.002667c3.605333 39.509333-28.458667 62.805333-64.896 47.146666l-229.504-98.517333-229.376 98.453333z" fill="#d94801" stroke="black" stroke-width="100" p-id="4388"></path></svg>`;
        svg.append("g")
        .attr("id","center_logo")
        .html(svg_code)
        .attr("transform",'translate(' + rect_x + ',' + rect_y+ ')')
        ;
        svg.select("#center_logo svg")
        .attr("x",+(rects.nodes()[i].getAttribute("x"))+(50-mark(d))/2+mark(d)-10)
        .attr("y",+(rects.nodes()[i].getAttribute("y"))+(50-mark(d))/2+mark(d)-10);
        if(selected_image_left_relative_loc_x==null && selected_image_right_relative_loc_x==null){
          var image_node=document.createElement("img");
          image_node.setAttribute("width","100");
          image_node.setAttribute("height","100");
          image_node.setAttribute("src","data:image/png;base64,"+d.image);
          var image_div=document.querySelector("#center_image");
          image_div.innerHTML="";
          image_div.appendChild(image_node);
        }
     }
     else if(d.relative_loc_x==selected_image_left_relative_loc_x&&d.relative_loc_y==selected_image_left_relative_loc_y){
      var logo_size=24*9*(1/row_number);
      var svg_code=`<svg t="1666413378157" class="icon1" viewBox="-100 -100 1154 1154" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="5525" width="${logo_size}" height="${logo_size}"><path d="M512 1024C229.248 1024 0 794.752 0 512S229.248 0 512 0s512 229.248 512 512-229.248 512-512 512z m-114.176-310.954667a53.333333 53.333333 0 0 0 75.434667 0l323.328-323.328a53.333333 53.333333 0 1 0-75.434667-75.434666l-287.914667 283.306666-128.853333-128.853333a53.333333 53.333333 0 1 0-75.434667 75.434667l168.874667 168.874666z" p-id="5526" fill="#d94801" stroke="black" stroke-width="100"></path></svg>`;
      svg.append("g")
        .attr("class","click_logo")
        .html(svg_code)
        .attr("transform",'translate(' + rect_x + ',' + rect_y+ ')');
      svg.select(".click_logo svg")
        .attr("x",+(rects.nodes()[i].getAttribute("x"))+(50-mark(d))/2+mark(d)-10)
        .attr("y",+(rects.nodes()[i].getAttribute("y"))+(50-mark(d))/2+mark(d)-10);

      var left_image_node=document.querySelectorAll("#svd_images image")[i];
      left_image_request_data(left_image_node);
     }
    }
  );
}

// 左边的click事件
// 右边的click事件
// 只有一个选定的image
// 
function right_image_click(){
  selected_image_right_relative_loc_x=this.getAttribute("relative_loc_x");
  selected_image_right_relative_loc_y=this.getAttribute("relative_loc_y");
  selected_image_left_relative_loc_x=null;
  selected_image_left_relative_loc_y=null;
  var svg=d3.select("#mainsvg");
  d3.selectAll(".click_logo").remove();
  var width=+(this.getAttribute("width"))-10;
  var rect_x=+(this.getAttribute("x"))+(img_size-width)/2+width-10;
  var rect_y=+(this.getAttribute("y"))+(img_size-width)/2+width-10;
  var logo_size=24*9*(1/row_number);
  var svg_code=`<svg t="1666413378157" class="icon1" viewBox="-100 -100 1154 1154" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="5525" width="${logo_size}" height="${logo_size}"><path d="M512 1024C229.248 1024 0 794.752 0 512S229.248 0 512 0s512 229.248 512 512-229.248 512-512 512z m-114.176-310.954667a53.333333 53.333333 0 0 0 75.434667 0l323.328-323.328a53.333333 53.333333 0 1 0-75.434667-75.434666l-287.914667 283.306666-128.853333-128.853333a53.333333 53.333333 0 1 0-75.434667 75.434667l168.874667 168.874666z" p-id="5526" fill="#d94801" stroke="black" stroke-width="100"></path></svg>`;
  svg.append("g")
  .attr("class","click_logo")
  .html(svg_code)
  .attr("transform",'translate(' + axis_padding + ',' + transform_y+ ')')
  svg.select(".click_logo svg")
  .attr("x",rect_x)
  .attr("y",rect_y)
  
  select_image(this);
}

function change_target_sample(img_node){
  src=img_node.href.baseVal;
  var img=document.createElement("img");
  img.setAttribute("src",src);
  img.setAttribute("width",100);
  img.setAttribute("height",100);
  var center_image=document.getElementById("center_image");
  center_image.innerHTML="";
  center_image.appendChild(img);
}
function change_info_pannel_image(src,data){
  var img=document.createElement("img");
  img.setAttribute("src",src);
  img.setAttribute("width",100);
  img.setAttribute("height",100);
  var center_image=document.getElementById("center_image");
  center_image.innerHTML="";
  center_image.appendChild(img);
  // draw_bar(data.feature_value_rate);
  // darw_accmulate_bar(data.accumulating_contribute_rate);
  show_neighbor_images(data.neighbor_images);
}
function show_neighbor_images(images){
  var svg=d3.select("#neighbors");
  svg.selectAll("image")
  .data(images)
  .join("image")
  .attr("xlink:href",d=>"data:image/png;base64,"+d)
  .attr("width",50)
  .attr("height",50)
  .attr("y",0)
  .attr("x",(d,i)=>7.5+i*57.5);

}
function highlight_barchart(img){

  var feature_bar=document.querySelectorAll("#feature_value_bar rect");
  var accumulate_bar=document.querySelectorAll("#accumulate_bar rect");
  for(let i=0;i<10;++i){
    feature_bar[i].style.opacity=1;
    accumulate_bar[i].style.opacity=1;
    feature_bar[i].setAttribute("fill","#CDCDCD");
    accumulate_bar[i].setAttribute("fill","#CDCDCD");
  }
  var idx1=parseInt(img.getAttribute("feature_index_x"));
  var idx2=parseInt(img.getAttribute("feature_index_y"));
  feature_bar[idx1].style.opacity=1;
  feature_bar[idx2].style.opacity=1;
  accumulate_bar[idx1].style.opacity=1;
  accumulate_bar[idx2].style.opacity=1;
  feature_bar[idx1].setAttribute("fill","#5b5d6b");
  feature_bar[idx2].setAttribute("fill","#5b5d6b");
  accumulate_bar[idx1].setAttribute("fill","#5b5d6b");
  accumulate_bar[idx2].setAttribute("fill","#5b5d6b");

}
function select_image(selected_image){
  // 要请求一些数据
  // 请求特征值的数据
  // 请求图片的mask
  // 
  highlight_barchart(selected_image);
  $.post(
    "/info_pannel",
    { feature_index_x:selected_image.getAttribute('feature_index_x'),
      feature_index_y:selected_image.getAttribute('feature_index_y'),
      shift_x:selected_image.getAttribute('shift_x'),
      shift_y:selected_image.getAttribute('shift_y')
    },function(data,status){
      change_info_pannel_image(selected_image.href.baseVal,data);
    }
  )
}

// 显示左边的图像的逻辑：
// 放大，平移，缩放，点击，或者随机选择一张图片
// 在显示图片的时候，不仅要考虑图片的大小
// 还要考虑两个logo的标记问题

// 左边的信息面板的更新：如果没有选中的图片，那么就显示中心图片

// 如果是放大和缩小的操作，就是显示更新之后的图片


// 左边的信息面板是点击之后再进行更新
// 或者是 左边的图片被选中之后进行 放大和缩小之后 
// 
// 选中左边，右边就是null
// 选中右边，左边就是null
// 左边的放大和缩小 ：如果有选中的更新信息面板，更新右边的界面
// 右边的放大和缩小 ：只需要更新信息面板

function image_click() {
  // 把选择的图片填充到信息面板中
  // 要把选择的图片高亮显示
  // 标记一下这张图片被点击了
  var svg=this.getAttribute("container");
  if(svg=="left_svg"){
    // d3.select("#left_border")
    // .attr("stroke-width","5px");
    // d3.select("#right_border")
    // .attr("stroke-width","0px");
    d3.select("#left_border")
    .attr("fill","#DDDDDD");
    d3.select("#right_border")
    .attr("fill","white");

    d3.select("#left_values")
    .selectAll("text")
    .attr("fill","black");
    d3.select("#right_values")
    .selectAll("text")
    .attr("fill","white");

  }else{
    // d3.select("#right_border")
    // .attr("stroke-width","5px");
    // d3.select("#left_border")
    // .attr("stroke-width","0px");
    d3.select("#right_border")
    .attr("fill","#DDDDDD");
    d3.select("#left_border")
    .attr("fill","white");

    d3.select("#right_values")
    .selectAll("text")
    .attr("fill","black");
    d3.select("#left_values")
    .selectAll("text")
    .attr("fill","white");

  }

  var svg=d3.select("#"+svg);
  svg.selectAll(".click_logo").remove();
  var width=+(this.getAttribute("width"))-10;
  var rect_x=+(this.getAttribute("x"))+(50-width)/2+width-10;
  var rect_y=+(this.getAttribute("y"))+(50-width)/2+width-10;
  var logo_size=24*9*(1/row_number);
  var svg_code=`<svg t="1666413378157" class="icon1" viewBox="-100 -100 1154 1154" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="5525" width="${logo_size}" height="${logo_size}"><path d="M512 1024C229.248 1024 0 794.752 0 512S229.248 0 512 0s512 229.248 512 512-229.248 512-512 512z m-114.176-310.954667a53.333333 53.333333 0 0 0 75.434667 0l323.328-323.328a53.333333 53.333333 0 1 0-75.434667-75.434666l-287.914667 283.306666-128.853333-128.853333a53.333333 53.333333 0 1 0-75.434667 75.434667l168.874667 168.874666z" p-id="5526" fill="#d94801" stroke="black" stroke-width="100"></path></svg>`;
  
  // var svg_code=`<svg t="1672140628528" class="icon1" viewBox="-100 -100 1124 1124" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2841" width="22" height="22"><path d="M512 0C230.4 0 0 230.4 0 512s230.4 512 512 512 512-230.4 512-512S793.6 0 512 0zM844.8 364.8l-428.8 428.8L179.2 556.8c-25.6-25.6-25.6-64 0-89.6s64-25.6 89.6 0l147.2 147.2 339.2-339.2c25.6-25.6 64-25.6 89.6 0S870.4 339.2 844.8 364.8z" fill="#d94801" stroke="black" stroke-width="100" p-id="2842"></path></svg>`;

  svg.append("g")
  .attr("class","click_logo")
  .html(svg_code)
  .attr("transform",'translate(' + axis_padding + ',' + transform_y+ ')');
  svg.select(".click_logo svg")
  .attr("x",rect_x)
  .attr("y",rect_y);
  request_data(this);
}


function request_data(image_node){
  var svg=image_node.getAttribute("container");
  var show_svg="#right_svg";
  selected_index=image_node.getAttribute("index");
  selected_svg=svg;
  mark_svg=0;
  if(svg=='right_svg'){
    mark_svg=1;
    show_svg="#left_svg";
  }
  var d3_svg=d3.select(show_svg);
  
  $.post("/select_image",
    {child_index:image_node.getAttribute('index'),
    mark:mark_svg
  },
    function(data,status){
      //渲染图片
      clear_svg(d3_svg);
      show(d3_svg,data.svd_images,data.domain_range);
      show_neighbor_images(data.neighbor_images);
      
      // draw_bar(data.feature_values);
      darw_accmulate_bar(data.accumulating_contribute_rate,1-mark_svg);
      
      change_target_sample(image_node);
      if(current_event!='zoom'){
        activate_image_click();
      }

      update_image_list_ul(data.image_list);

    })
  // 还需要更新一下别的数据
  // 更新邻居，更新柱状图，更新显示的图片

  // select_image(image_node);
}

function update_image_list_ul(image_list){
  var ul=document.getElementById("image_list_ul");
  // TO DO:
  // 把图片以列表的形式展示出来
  // 创建列表
  // 创建图片元素
  ul.innerHTML="";
  console.log(image_list);
  for(var i=0;i<image_list.length;++i){
    li=construct_li_image("data:image/png;base64,"+image_list[i],i);
    ul.appendChild(li);
  }

}

function construct_li_image(image_src,i){
  var li=document.createElement("li");
  var image=document.createElement("img");

  var div=document.createElement("div");
  div.setAttribute("width",50);
  div.setAttribute("height",50);
  image.setAttribute("src",image_src);
  image.setAttribute("width",50);
  image.setAttribute("height",50);
  image.setAttribute("step",i);
  image.onclick=function(){
    $.post("/return_history_step",{
      step_index:this.getAttribute("step")
    },function(data,status){
      var svg=d3.select("#right_svg");
      clear_svg(svg);
  
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
  
      update_image_list_ul(data.image_list);
    })

  };
  div.style.paddingTop="7.75px";
  div.appendChild(image);
  li.appendChild(div);
  return li;
}

function only_show_pure_images(images){
  var mainGroup=d3.select("#mainsvg");

  // var color=d3.interpolateHslLong("blue", "red");
  var loc_x;
  var loc_y;
  if(selected_image_left_relative_loc_x==0){
      loc_x=8;
  }else{
      loc_x=8*selected_image_left_relative_loc_x/Math.abs(selected_image_left_relative_loc_x);
  }
  if(selected_image_left_relative_loc_y==0){
      loc_y=8;
  }else{
      loc_y=8*selected_image_left_relative_loc_y/Math.abs(selected_image_left_relative_loc_y);
  }
  if(selected_image_right_relative_loc_y==null){
    selected_image_right_relative_loc_y=loc_y;
    selected_image_right_relative_loc_x=loc_x;
  }

  mainGroup.select("#right_outer_border")
  .selectAll("rect")
  .data(images)
  .join("rect")
  .attr("class","border")
  .attr("y",(d,i)=>-10+(Math.floor(i/row_number)*image_size_plus_distance_imgs-5))
  .attr("x",((d,i)=>10+((i%row_number)*image_size_plus_distance_imgs-5)))
  .attr("width",d=>mark(d)+10)
  .attr("height",d=>mark(d)+10)
  .attr('transform',d=> `translate(${axis_padding+(50-mark(d))/2}, ${transform_y+(50-mark(d))/2})`)
  .attr("probability",d=>(+d.probability))
  .attr("fill",d=>color(+d.novelty_score))
  .attr("stroke","grey")
  ;

  mainGroup.select("#pure_image")
  .selectAll("image")
  .data(images)
  .join("image")
  .attr("xlink:href",(d)=>"data:image/png;base64,"+d.image)
  .attr("width",d=>mark(d))
  .attr("height",d=>mark(d))
  .attr("y",(d,i)=>-10+(Math.floor(i/row_number)*(img_size+distance_imgs)))
  .attr("x",((d,i)=>10+((i%row_number)*(img_size+distance_imgs))))
  .attr('transform', d=>`translate(${axis_padding+(50-mark(d))/2}, ${transform_y+(50-mark(d))/2})`)
  .attr("index",(d,i)=>i)
  .attr("shift_x",(d)=>d.shift_x)
  .attr("shift_y",(d)=>d.shift_y)
  .attr("feature_index_x",(d)=>d.index_x)
  .attr("feature_index_y",(d)=>d.index_y)
  .attr("relative_loc_x",(d)=>d.relative_loc_x)
  .attr("relative_loc_y",(d)=>d.relative_loc_y)
  .attr("probability",(d)=>d.probability)
  .attr("opacity",1)
  .attr("class","pure_image");
  right_mark_central_image();
}

function right_mark_central_image(){

  var svg=d3.select("#mainsvg");
  var rects=svg.selectAll("rect");

  svg.select("#center_logo").remove();
  svg.selectAll(".click_logo").remove();
  rects.each(
    (d,i)=>{
      var rect_x=+(rects.nodes()[i].getAttribute("x"))+(50-mark(d))/2+mark(d)-10;
      var rect_y=+(rects.nodes()[i].getAttribute("y"))+(50-mark(d))/2+mark(d)-10;
      if(d.relative_loc_x==0&&d.relative_loc_y==0){
        var svg_code=`<svg t="1666411352971" class="icon1" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="4387" width="24" height="24"><path d="M284.458667 941.397333c-36.437333 15.637333-68.48-7.68-64.896-47.168l22.613333-248.917333-164.394667-188.053333c-26.069333-29.824-13.653333-67.562667 24.789334-76.309334l243.370666-55.381333 127.786667-214.677333c20.288-34.090667 59.946667-34.069333 80.213333 0l127.786667 214.677333 243.370667 55.381333c38.656 8.789333 50.858667 46.485333 24.789333 76.309334l-164.394667 188.053333 22.741334 249.002667c3.605333 39.509333-28.458667 62.805333-64.896 47.146666l-229.504-98.517333-229.376 98.453333z" fill="#d94801" stroke="black" stroke-width="100" p-id="4388"></path></svg>`;
        svg.append("g")
        .attr("id","center_logo")
        .html(svg_code)
        .attr("transform",'translate(' + axis_padding + ',' + transform_y+ ')')
        ;
        svg.select("#center_logo svg")
        .attr("x",rect_x)
        .attr("y",rect_y);
     }else if(selected_image_right_relative_loc_x==(+d.relative_loc_x)&&
     selected_image_right_relative_loc_y==(+d.relative_loc_y)){
      var logo_size=24*9*(1/row_number);
      var svg_code=`<svg t="1666413378157" class="icon1" viewBox="-100 -100 1154 1154" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="5525" width="${logo_size}" height="${logo_size}"><path d="M512 1024C229.248 1024 0 794.752 0 512S229.248 0 512 0s512 229.248 512 512-229.248 512-512 512z m-114.176-310.954667a53.333333 53.333333 0 0 0 75.434667 0l323.328-323.328a53.333333 53.333333 0 1 0-75.434667-75.434666l-287.914667 283.306666-128.853333-128.853333a53.333333 53.333333 0 1 0-75.434667 75.434667l168.874667 168.874666z" p-id="5526" fill="#d94801" stroke="black" stroke-width="100"></path></svg>`;
      svg.append("g")
      .attr("class","click_logo")
      .html(svg_code)
      .attr("transform",'translate(' + axis_padding + ',' + transform_y+ ')');
      svg.select(".click_logo svg")
      .attr("x",rect_x)
      .attr("y",rect_y);
      var right_image_node=document.querySelectorAll("#pure_image image")[i];
      select_image(right_image_node);
     }
    }
  );
}
function show_pure_images(data){

  images=data.images;

  var domain_x=data.domain_x;
  var domain_y=data.domain_y;
  console.log(domain_x);
  console.log(domain_y);
  domain_x=domain_x.split(",");
  domain_y=domain_y.split(",");
  var start_x=parseFloat(domain_x[0]);
  var end_x=parseFloat(domain_x[1]);
  var start_y=parseFloat(domain_y[0]);
  var end_y=parseFloat(domain_y[1]);

  only_show_pure_images(images);
  var right_images=document.querySelectorAll("#pure_image image");
  for(let i=0;i<right_images.length;++i){
    var image_node=right_images[i];
    image_node.addEventListener('click',right_image_click,false);
  }
  var mainGroup=d3.select("#mainsvg");
  mainGroup=mainGroup.select("#pure_image");
  // 下面是画坐标轴的代码
  mainGroup.selectAll("g").remove();
  right_gx=mainGroup.append("g");
  right_gy=mainGroup.append("g");
  right_scalex = d3.scaleLinear().domain([start_x, end_x]).range([0,len_axis-img_size]);
  right_scaley = d3.scaleLinear().domain([start_y, end_y]).range([0, len_axis-img_size]);
  right_axisx = d3.axisBottom(right_scalex);
  right_axisy=d3.axisLeft(right_scaley);

  right_gx
  .attr("transform", `translate(${axis_padding_x+10},${transform_x-4})`)
  .attr("id","right_gx")
  .call(right_axisx);
  right_gy
  .attr("transform", `translate(${axis_padding+4},${transform_gy-10})`)
  .attr("id","right_gy")
  .call(right_axisy);
  //把中心和点击的图片都进行标记
}

function drwa_axis(svg,domain_range){
  svg.selectAll(".axis").remove();
  gx=svg.append("g")
  .attr("class","axis");
  gy=svg.append("g")
  .attr("class","axis");
  scalex = d3.scaleLinear().domain([-domain_range, domain_range]).range([0,len_axis-img_size]);
  scaley = d3.scaleLinear().domain([-domain_range, domain_range]).range([0, len_axis-img_size]);
  axisx = d3.axisBottom(scalex);
  axisy=d3.axisLeft(scaley);

  gx
  .attr("transform", `translate(${axis_padding_x+10},${transform_x-4})`)
  .attr("id","right_gx")
  .call(axisx);
  gy
  .attr("transform", `translate(${axis_padding+4},${transform_gy-10})`)
  .attr("id","right_gy")
  .call(axisy);
  var svg_type=svg.attr("id");
  if(svg_type=='left_svg'){
    left_gx=gx;left_gy=gy;
    left_scalex=scalex;left_scaley=scaley;
    left_axisx=axisx;left_axisy=axisy;
  }else{
    right_gx=gx;right_gy=gy;
    right_scalex=scalex;right_scaley=scaley;
    right_axisx=axisx;right_axisy=axisy;
  }
}

function show_interpret_images(images){
  var mainGroup=d3.select("#mainsvg");
  var row_number=1+(row_interpret_images-1)/3;
  var sample_images=Array();
  console.log(images.length);
  for(let i=0;i<row_interpret_images;i+=3){
    for(let j=0;j<row_interpret_images;j+=3){
      sample_images.push(images[i*row_interpret_images+j]); 
    }
  }
  mainGroup
  .selectAll("image")
  .data(sample_images)
  .join("image")
  .attr("xlink:href",(d)=>"data:image/png;base64,"+d.image)
  .attr("width",img_interpret_size)
  .attr("height",img_interpret_size)
  .attr("y",(d,i)=>(Math.floor(i/row_number)*(img_interpret_size+dis_interpret_images)))
  .attr("x",((d,i)=>((i%row_number)*(img_interpret_size+dis_interpret_images))))
  .attr('transform', `translate(${axis_padding}, ${transform_y})`)
  .attr("index",(d,i)=>i)
  .attr("shift_x",(d)=>d.shift_x)
  .attr("shift_y",(d)=>d.shift_y)
  .attr("class","interpret_image");
  images=document.querySelectorAll("#mainsvg .interpret_image");
  for(let i=0;i<images.length;++i){
    images[i].style.display="none";
  }
}
// 概率密度大的地方，颜色越深
function draw_prob_dense(data){
  images=data.interpret_images;

  var mainGroup=d3.select("#interpret_images");
  var color = ["#fee5d9", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"];
  thresholds = Array([])
  for(let i=0;i<5;++i){
    thresholds.push(min_value+i*(max_value-min_value)/5);
  }

  const grid = new Array(row_interpret_images*row_interpret_images);
  for(let i=0;i<row_interpret_images*row_interpret_images;++i){
    grid[i]=parseFloat(images[i].probability);
  }
  var scale_number=len_axis/row_interpret_images;
  var contours = d3.contours()
  .size([row_interpret_images, row_interpret_images])
  .thresholds(thresholds)
  (grid);

  var contour=mainGroup;
  console.log(contours);
  var path = d3.geoPath();
  contour.
  selectAll("path")
  .data(contours)
  .enter()
  .append("path")
  .attr("d", path)
  .attr('fill', (d,i)=>color[i])
  .attr("transform",` translate(${axis_padding},${transform_y}) scale(${scale_number})`)
  paths=document.querySelectorAll("#mainsvg path");
  // for(let i=0;i<images.length;++i){
  //   paths[i].style.display="none";
  // }
}
function draw(robust_value){
    let scalex = d3.scaleLinear().domain([-8, 8]).range([0, 700]);
    let scaley = d3.scaleLinear().domain([-8, 8]).range([0, 700]);
    let axisx = d3.axisBottom(scalex);
    let axisy=d3.axisRight(scaley);

    n=robust_value.length;
    m=robust_value.length;
    const grid = new Array(n * m);
    for(let i=0;i<m;++i){
        for(let j=0;j<n;++j){
            grid[i*n+j]=robust_value[i][j];
        }
    }
    grid.n = n;
    grid.m = m;
    var maxValue=Math.max.apply(null,grid);
    var minValue=Math.min.apply(null,grid);
    thresholds = Array.from([0,0.01,0.03,0.05,0.07,0.09])

    //Compute the contour polygons at log-spaced
    //intervals; returns an array of MultiPolygon.

    var contours = d3.contours()
      .size([n, m])
      .thresholds(thresholds)
      (grid);

    var path = d3.geoPath();

    //color = d3.scaleSequential([-1, 1], d3.interpolateSpectral)
    var color = ["#fee5d9", "#fcbba1", "#fc9272", "#fb6a4a", "#de2d26", "#a50f15"];
    var mainGroup = d3.select("#mainsvg");
    var contour=mainGroup.append("g").attr("id","contour");
    function zoomHandler(e){
      var svg=d3.select("#mainsvg #contour");
      svg.attr("transform",e.transform);
      gx.call(axisx.scale(e.transform.rescaleX(scalex)));
      gy.call(axisy.scale(e.transform.rescaleY(scaley)));

    }
    function zoomEndHandler(e){
      domainX=e.transform.rescaleX(scalex).domain();
      domainY=e.transform.rescaleY(scaley).domain();
      console.log(domainX);
      console.log(domainY);
      // 重新请求数据
      // 重新绘制页面
      
    }
    let zoom=d3.zoom()
    .scaleExtent([0.1,20])
    .on("zoom",zoomHandler)
    .on("end",zoomEndHandler);

    contour.
      selectAll("path")
      .data(contours)
      .enter()
      .append("path")
      .attr("d", path)
      .attr('fill', (d,i)=>color[i])
      .attr("transform","scale(16)")
    var svg=d3.select("#mainsvg");
    // 绘制坐标轴
    
    const gx = svg.append("g").call(axisx);
    const gy = svg.append("g").call(axisy);
    gx.select(".domain").attr("display","none");
    gy.select(".domain").attr("display","none");
      // set up the ancillary zooms and an accessor for their transforms
    svg.call(zoom);
}
function drawaxes(axes_points){
        var colors=["#DC143C","#C71585","#FF00FF","#0000CD","#00FA9A"]
        var line=d3.select("#line");
        let scale = d3.scaleLinear().domain([0,1]).range([0,492]);
        var valueline = d3.line()
        .x(function(d,i) { console.log(d);return scale(d[0]); })
        .y(function(d,i) { console.log(d);return scale(d[1]); })
        .curve(d3.curveBasis);
        for(var i=0;i<axes_points.length;++i){
          line
          .append("path")
          .attr("d",valueline(axes_points[i]))
          .attr('stroke', colors[i])
            .attr('stroke-width', 2)
          .attr("fill","none");
        }
      }
      
function drawlines(){
        // 先向后台发送请求，请求点的数据
        // 接下来绘制线
        var width=400;
        var height=400;
        var points = d3.range(1, 5).map(i => [i * width / 5, 50 + Math.random() * (height - 100)]);
        console.log(points);
        var index=0;
        var index1=1;
        var valueline = d3.line()
        .x(function(d,i) { console.log(d);return d[0]; })
        .y(function(d,i) { console.log(d);return d[1]; })
        .curve(d3.curveBasis);
  
        var mainGroup=d3.select("#line");
        mainGroup
        .append("path")
        .attr("d",valueline(points))
        .attr('stroke', 'black')
          .attr('stroke-width', 2)
        .attr("fill","none");
        mainGroup.append("line")
              .attr("x1", 20)
              .attr("y2", 20)
              .attr("x2", 300)
              .attr("y2", 100)
              .attr("stroke", "black")
              .attr("stroke-width", "2px")
              .style('zIndex',"1");
      }
function add_icon(){
  var heartIcon = d3.icon({type: 'heart'});
  var svgContainer=d3.select("#svd_images");
  svgContainer.append('path')
          .attr("transform", "translate(20,20)")
          .attr('d', heartIcon)
          .attr("stroke", "red")
          .attr("stroke-width", 2)
          .attr("fill", "none");

}
function test_function(){
  alert("This is for test")
}

function mark(d){
  var measures=document.getElementById("measure_select").options;
  measure_index=document.getElementById("measure_select").selectedIndex;
  var select_measure=measures[measure_index].innerHTML;

  switch(select_measure){
    case 'Select Measure':
      return img_size_scalar(1);
    case 'Robustness':
      console.log(d.robustness);
      return img_size_scalar(d.robustness);
    case 'Probability':
      return img_size_scalar(d.probability);
    case 'Realness':
      return img_size_scalar(d.realness);
    case 'KOR':
      return img_size_scalar(d.kor);
    case 'QED':
      return img_size_scalar(d.qed);
    case 'logP':
      return img_size_scalar(d.logP);
    case 'SAScore':
      return img_size_scalar(d.sascore);
    // TO DO: housegan的指标 
    // variance_area,overlap,compatability

    case 'Variance_area':
      return img_size_scalar(d.variance_area);
    case 'Overlap':
      return img_size_scalar(d.overlap);
    case 'Compatability':
      return img_size_scalar(d.compatability);
    // moluclar 的指标'kor',"qed","logP","sascore"

  }
  


}
// //   var heatmapInstance = h337.create({
//     // only container is required, the rest will be defaults
//     //只需要一个container，也就是最终要绘制图形的dom节点，其他都默认
//     container: document.querySelector('#center_image'),
//     maxOpacity: .6,
//     // minimum opacity. any value > 0 will produce 
//     // no transparent gradient transition 
//     minOpacity: .0,
//     gradient: {
//       // enter n keys between 0 and 1 here
//       // for gradient color customization
//       // '.75': '#fee0d2',
//       '.85': '#fc9272',
//       '.995': "#de2d26"
//       },
//   });

//   // now generate some random data
//   // var max = d3.max(global_points,d=>+d.value)
//   // for(let i=0;i<points.length;++i){
//   //   points[i].x=+points[i].x;
//   //   points[i].y=+points[i].y;
//   //   points[i].value=+points[i].value;
//   // }
//   // // heatmap data format
//   // var data = {
//   //   max: max,//所有数据中的最大值
//   //   data: points//最终要展示的数据
//   // };
//   // // if you have a set of datapoints always use setData instead of addData
//   // // for data initialization
//   // heatmapInstance.setData(data);

    // mainGroup.select("#left_inner")
    // .selectAll("rect")
    // .data(images)
    // .join("rect")
    // .attr("class","inner")
    // .attr("y",(d,i)=>-10+(Math.floor(i/row_number)*image_size_plus_distance_imgs))
    // .attr("x",((d,i)=>10+((i%row_number)*image_size_plus_distance_imgs)))
    // .attr("width",d=>img_size_scalar(d.probability))
    // .attr("height",d=>img_size_scalar(d.probability))
    // .attr('transform',d=>(`translate(${axis_padding+(50-img_size_scalar(d.probability))/2}, ${transform_y+(50-img_size_scalar(d.probability))/2})`))
    // .attr("fill","black")

      // if(selected_image_left){
      //   if( d.shift_x==selected_image_left_shift_x && d.shift_y==selected_image_left_shift_y){
      //     return 1;
      //   }
      //   if(current_event=="zoom" && i==selected_image_left_index){
      //     return 1;
      //   }
      // }