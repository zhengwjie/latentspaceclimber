// 展示起始点
function start_point_show() {

    index=layer.open({
        title: ["Starting Point Selector","height:30px;line-height:30px;padding-left:10px;border-bottom-color:#CDCDCD;" +
        "font-family:serif;" +
        "font-weight:bold;" +
        "font-size:20px;overflow:visible;"
    ],
        type: 1,
        content: $("#start_point_view") //这里content是一个普通的String
    });
    
    var title_div=document.getElementsByClassName("layui-layer")[0];
    var div=document.createElement("span");
    // div.style.float="right";
    // div.style.paddingRight="45px";
    // div.style.paddingTop="10px";
    div.style.width="30px";
    div.style.height="30px";
    div.style.position="absolute";
    div.style.top="6px";
    div.style.left="405px";

    div.setAttribute("id","btn_div");
    const svg_code=`<svg 
    t="1666240092306" class="icon" viewBox="0 0 1024 1024" version="1.1" xmlns="http://www.w3.org/2000/svg" p-id="2318" width="16" height="16"><path d="M684.032 403.456q-17.408-8.192-15.872-22.016t11.776-22.016q3.072-2.048 19.968-15.872t41.472-33.28q-43.008-49.152-102.4-77.312t-129.024-28.16q-64.512 0-120.832 24.064t-98.304 66.048-66.048 98.304-24.064 120.832q0 63.488 24.064 119.808t66.048 98.304 98.304 66.048 120.832 24.064q53.248 0 100.864-16.896t87.04-47.616 67.584-72.192 41.472-90.624q7.168-23.552 26.624-38.912t46.08-15.36q31.744 0 53.76 22.528t22.016 53.248q0 14.336-5.12 27.648-21.504 71.68-63.488 132.096t-99.84 103.936-128.512 68.096-148.48 24.576q-95.232 0-179.2-35.84t-145.92-98.304-98.304-145.92-36.352-178.688 36.352-179.2 98.304-145.92 145.92-98.304 179.2-36.352q105.472 0 195.584 43.52t153.6 118.272q23.552-17.408 39.424-30.208t19.968-15.872q6.144-5.12 13.312-7.68t13.312 0 10.752 10.752 6.656 24.576q1.024 9.216 2.048 31.232t2.048 51.2 1.024 60.416-1.024 58.88q-1.024 34.816-16.384 50.176-8.192 8.192-24.576 9.216t-34.816-3.072q-27.648-6.144-60.928-13.312t-63.488-14.848-53.248-14.336-29.184-9.728z" p-id="2319" fill="#8a8a8a"></path></svg>`;
    let doc = new DOMParser().parseFromString(svg_code, 'text/html');
    let btn = doc.querySelector('.icon');
    div.append(btn);
    title_div.appendChild(div);
    btn=title_div.querySelector("#btn_div");
    btn.addEventListener('click',request_samples,false);
    var node=document.querySelector(".layui-layer-shade");
    node.remove();
    var pop_out=document.querySelector(".layui-layer");
    pop_out.style.boxShadow = "0px 0px 7px black";
    pop_out.style.border="1px solid #cdcdcd";
    var title_div=document.querySelector(".layui-layer-title");
    title_div.style.backgroundColor="#dfdcdc";

    // var shadow_div=document.getElementById("layui-layer1");
    // shadow_div.setAttribute("box-shadow","")


}

function request_samples(){
    $.post("/samples",{

    },function(data,status){
        var x_min=parseFloat(data.min_x);
        var x_max=parseFloat(data.max_x);
        var y_min=parseFloat(data.min_y);
        var y_max=parseFloat(data.max_y);
        var img_info=data.img_info;
        var mainsvg=d3.select("#sample_svg");
        var x_scalar=d3.scaleLinear().domain([x_min, x_max]).range([2, 419]);
        var y_scalar=d3.scaleLinear().domain([y_min, y_max]).range([2, 219]);
        mainsvg.selectAll("g").remove();
        mainsvg.append("g")
        .attr("id","border")
        .selectAll("rect")
        .data(img_info)
        .join("rect")
        .attr("width",34)
        .attr("height",34)
        .attr("y",d=>y_scalar(+d.point_y)-2)
        .attr("x",d=>x_scalar(+d.point_x)-2)
        .attr("style","fill:white;")
        .attr("index",(d,i)=>i)
        .attr("id",(d,i)=>"border"+i)
        ;

        mainsvg.append("g")
        .selectAll("image")
        .data(img_info)
        .join("image")
        .attr("xlink:href",d=>"data:image/png;base64,"+d.img)
        .attr("width",30)
        .attr("height",30)
        .attr("y",d=>y_scalar(+d.point_y))
        .attr("x",d=>x_scalar(+d.point_x))
        .attr("index",(d,i)=>i)
        .attr("show",0)
        .attr("id",(d,i)=>"image"+String(i))
        .on("click",start_exploration);
    })
}
// 开启探索
function start_exploration(){
    d3.select("#border").selectAll("rect").attr("style","fill:white;");
    var idx=d3.select(this).attr("index");
    var mainsvg=d3.select("#sample_svg");
    console.log(mainsvg.select("#border").selectAll("rect"));

    var rect=mainsvg.select("#border"+idx);
    rect.attr("style","fill:red;");

    $.post("/select_sample",{
        selected_inx:idx
    },function(data,status){
        
        clear_env();
        // update_axis();
        var svg=d3.select("#left_svg");
        show(svg,data.svd_images,data.domain_range);
        
        show_neighbor_images(data.neighbor_images);
        if(current_event!='zoom'){
        activate_image_click();
        }
        mark_left_div();
        update_image_list_ul(data.image_list);
        darw_accmulate_bar(data.accumulating_contribute_rate,0);

    })
}

function show_select_box(){
    var samples_div=document.getElementById("input_selection");
    var idx=d3.select(this).attr("index");
    if(d3.select(this).attr("show")==0){
        d3.select(this).attr("show",1);
        var input=document.createElement("input");
        input.setAttribute("type","checkbox");
        input.setAttribute("id","input"+String(idx));
        input.setAttribute("index",idx);
        if(selected_inx.includes(idx)){
            input.setAttribute("checked","true");
        }
        input.style.position="fixed";
        input.style.marginTop=String(d3.select(this).attr("y"))+"px";
        input.style.marginLeft=String(d3.select(this).attr("x"))+"px";
        input.addEventListener('change',change_select_state,false);
        samples_div.appendChild(input);

    }else{
        d3.select(this).attr("show",0);
        var input=document.getElementById("input"+String(idx));
        samples_div.removeChild(input);
    }
}
var selected_inx=[];
function change_select_state(){
    // this.checked
    var input_index=this.getAttribute("index");
    if(this.checked==true){
        // 选中
        selected_inx.push(input_index);
    }else{
        selected_inx.map((val,i)=>{
            if(val==input_index){
                selected_inx.splice(i,1);
            }
        })
    }

    // update_selected_imgs();

    // 改变状态的逻辑
    // 选中就添加到右边的界面中
    // 取消选择就从右边的界面删除选择的按钮
    // 使用一个全局数组记录一下选中的数据
    // 在左边的界面显示选中的图片
}
// 根据selected_inx更新右边的界面
// 右边的界面设计
// 找找论文再写

function update_selected_imgs(){
    var selected_image_list=document.getElementById("selected_images_list");
    selected_image_list.innerHTML="";
    for(let i=0;i<selected_inx.length;++i){
        var idx=selected_inx[i];
        var img=document.getElementById("image"+String(idx));
        var img_src=img.getAttribute("href");
        var weight=1/selected_inx.length;
        var img_li=create_li_image(img_src,weight);
        selected_image_list.appendChild(img_li);
    }

}
// 在这里面创建一个进度条

function create_li_image(image_src,weight){
    var li=document.createElement("li");
    var image=document.createElement("img");
    var slider=document.createElement("input");
    slider.setAttribute("type",'range');
    slider.setAttribute("min",0);
    slider.setAttribute("max",1);
    slider.setAttribute("value",0.33);
    slider.setAttribute("step","0.01");
    var div=document.createElement("div");
    div.setAttribute("width",100);
    div.setAttribute("height",100);
    image.setAttribute("src",image_src);
    image.setAttribute("width",80);
    image.setAttribute("height",80);
    div.style.paddingBottom="10px";
    div.appendChild(image);
    div.appendChild(slider);
    li.appendChild(div);
    return li;
}

// 需要有两个按钮，一个是合成，一个是
// var image_list=document.getElementById("explored_images_list");
// image_list.innerHTML="";
// for(let i=0;i<data.length;++i){
//     img_info=data[i];
//     var li=document.createElement("li");
//     var image=document.createElement("img");
//     var div=document.createElement("div");
//     div.setAttribute("width",100);
//     div.setAttribute("height",100);
//     image.setAttribute("src","data:image/png;base64,"+img_info.image);
//     image.setAttribute("width",80);
//     image.setAttribute("height",80);
//     div.style.paddingBottom="10px";
//     div.appendChild(image);
//     li.appendChild(div);
//     image_list.appendChild(li);
// }