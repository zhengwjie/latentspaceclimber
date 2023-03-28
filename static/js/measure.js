
var model_measures={
    "Select Model":["Select Measure"],
    "SNGAN(MNIST)":["Select Measure","Robustness","Probability","Realness"],
    "StyleGAN2-ADA(CIFAR10)":["Select Measure","Robustness","Probability","Realness"],
    "drug_generator":["Select Measure",'QED','KOR','logP','SAScore'],          
    "housegan":["Select Measure","Variance_area","Overlap","Compatability"]
}


function update_measure(model_name){
    var measure_ele=document.getElementById("measure_select");
    document.querySelectorAll('#measure_select option').forEach(option => option.remove());
    measures=model_measures[model_name];
    for(let i=0;i<measures.length;++i){
        var option = document.createElement("option");
        option.text = measures[i];
        measure_ele.add(option);
    }
}

// 这里根据选择的属性修改映射的图像和边框的大小
function update_image_size(){
    // 获取选择的model_name
    // 获取选择指标
    // var models=document.getElementById("model_select").options;
    // model_index=document.getElementById("model_select").selectedIndex;
    // var select_model=models[model_index].innerHTML;

    // var measures=document.getElementById("measure_select").options;
    // measure_index=document.getElementById("measure_select").selectedIndex;
    // var select_measure=measures[measure_index].innerHTML;
    // 更新左边的界面
    d3.selectAll("#left_svg .border")
    .attr("width",d=>(mark(d)+2*distance_image_border))
    .attr("height",d=>(mark(d)+2*distance_image_border))
    .attr('transform',d=>(`translate(${axis_padding+(img_size-mark(d))/2}, ${transform_y+(img_size-mark(d))/2})`));

    d3.selectAll("#left_svg image")
    .attr("width",d=>(mark(d)))
    .attr("height",d=>(mark(d)))
    .attr('transform',d=>(`translate(${axis_padding+(img_size-mark(d))/2}, ${transform_y+(img_size-mark(d))/2})`));
    svg=d3.select("#left_svg");
    mark_images(svg);
    

    // 更新右边的界面
    d3.selectAll("#right_svg .border")
    .attr("width",d=>(mark(d)+2*distance_image_border))
    .attr("height",d=>(mark(d)+2*distance_image_border))
    .attr('transform',d=>(`translate(${axis_padding+(img_size-mark(d))/2}, ${transform_y+(img_size-mark(d))/2})`));

    d3.selectAll("#right_svg image")
    .attr("width",d=>(mark(d)))
    .attr("height",d=>(mark(d)))
    .attr('transform',d=>(`translate(${axis_padding+(img_size-mark(d))/2}, ${transform_y+(img_size-mark(d))/2})`));
    // right_mark_central_image();
    svg=d3.select("#right_svg");
    mark_images(svg);
}

