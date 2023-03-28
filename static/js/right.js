
function hide_image(e){
    images=document.querySelectorAll("#mainsvg image");
    var show_type;
    if(images.length>0){
        if(images[0].style.display=="none"){
            show_type='block';
        }else{
            show_type='none';
        }
    }
    for(let i=0;i<images.length;++i){
        images[i].style.display=show_type;
    }
}
// 其实三种视图都已经展示出来了
// 但是
function changet_view(){
    

    
}