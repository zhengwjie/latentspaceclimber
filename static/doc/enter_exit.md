之前创建的HTML和SVG元素都是进入元素，要被移除的就是exiting元素。
在一些实例中区别对待entering和exiting元素是有用的。尤其是当处理transitions的时候。

你可以通过给.join函数传函数区别对待entering和exiting函数。
.join(
    function(enter){
        return enter
        .append('circle')
        .style('opacity',0);

    },
    function(update){
        return update.style('opacity',1);
    },
    function(exit){
        return exit.remove();
    }
)
Note that the enter, update and exit functions must return the selection


