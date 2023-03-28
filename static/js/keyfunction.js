// var letters='ABCDEFGHIJKLMNOPQRSTUVWXYZ';
// var i=25;

// function doInsert(){
//     if(i<0){
//         return;
//     }
//     var myData=letters.slice(i).split("");
//     i--;
//     update(myData);
// }

// function update(data){
//     d3.select("#content")
//     .selectAll("div")
//     .data(data,function(d){
//         return d;
//     })
//     .join('div')
//     .transition()
//     .style('left',function(d,i){
//         return i*32+'px';
//     })
//     .text(function(d){
//         return d;
//     });
// }

// doInsert();

var letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ';
var i = 25;

function doInsert() {
    if(i < 0)
        return;

    var myData = letters.slice(i).split('');
    i--;
    update(myData);
}

function update(data) {
    d3.select('#content')
        .selectAll('div')
        .data(data, function(d) {
            return d;
        }
        )
        .join('div')
        .transition()
        .style('left', function(d, i) {
            return i * 32 + 'px';
        })
        .text(function(d) {
            return d;
        });
}

doInsert();

