// 需要记录一下当前走到了多少步
var current_step=0;


function clear_env(){
    global_step=0;
    max_value=0;
    clear_env_next_step();
    var svg=d3.select("#left_svg");
    clear_svg(svg);
    var svg=d3.select("#right_svg");
    clear_svg(svg);
}
function clear_env_next_step(){
    selected_image_left=null;
    selected_image_left_shift_x=null;
    selected_image_left_shift_y=null;
    selected_image_right=null;
    selected_image_left_index=null;
    selected_image_right_index=null;
    selected_image_right_shift_x=null;
    selected_image_right_shift_y=null;
    selected_image_left_relative_loc_x=null;
    selected_image_left_relative_loc_y=null;
    selected_image_right_relative_loc_x=null;
    selected_image_right_relative_loc_y=null;
    end_point_x=null;
    end_point_y=null;
    selected_index=null;
    selected_svg="left_svg";
}
function left_next_step(){
    clear_env();
    var shift_x=selected_image_left.getAttribute("shift_x");
    var shift_y=selected_image_left.getAttribute("shift_y");
    var feature_index_x=selected_image_left.getAttribute("feature_index_x");
    var feature_index_y=selected_image_left.getAttribute("feature_index_y");
    current_step=current_step+1;
    document.getElementById("progress").setAttribute("value",current_step);

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
        d3.selectAll("#left_outer_border rect").remove();
        d3.selectAll("#left_inner rect").remove();
        show(data.svd_images);
        draw_bar(data.feature_values);
        darw_accmulate_bar(data.accumulating_contribute_rate);
    })
}

function right_next_step(){
    clear_env();
    var shift_x=selected_image_right.getAttribute("shift_x");
    var shift_y=selected_image_right.getAttribute("shift_y");
    var feature_index_x=selected_image_right.getAttribute("feature_index_x");
    var feature_index_y=selected_image_right.getAttribute("feature_index_y");
    current_step=current_step+1;
    document.getElementById("progress").setAttribute("value",current_step);
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
        update_axis();
        d3.selectAll("#left_outer_border rect").remove();
        d3.selectAll("#left_inner rect").remove();
        console.log(data);
        show(data.svd_images);
        draw_bar(data.feature_values);
        darw_accmulate_bar(data.accumulating_contribute_rate);
    })
}