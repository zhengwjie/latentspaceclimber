addPoints(){
    that.points[parseInt(that.active)].is_leaves = false;
    if(that.active > 0){
      that.points[parseInt(that.active)].legendList.is_leaves = false;
    }
    if(that.active != 0)
      color_list.push(palette(that.points[that.active].state));

    var count;
    var res = [];
    var sum = [];
    // eslint-disable-next-line no-unused-vars
    var sum_total = 0;
    var res_kind = [];
    var feature_length = 0;
    // var color_kind;
    let scale = d3.scaleLinear().domain([0,30000]).range([0,800]);
    if (that.active < 0) return;
    var legendlist = {};
    var res_kind_list = [];

    if(that.dataset_num == 4){
      if(that.selectFeature === "year"){
        legendlist["class"] = "Birth Year";
        // color_kind = color(0);
        feature_length = that.classificationYear.length;
        count = new Set(that.classificationYear).size;
        // eslint-disable-next-line no-unused-vars
        let data1 = [[], [], [], [], [], [], [], [], [], []];
        if(that.select_feature[0]){
          for(let i = 0; i < that.currentPoint.length; i++)
          {
            data1[that.data_behind[that.currentPoint[i]][0]].push(that.currentPoint[i])
          }
        }
        let setdata = new Set(that.classificationYear);
        setdata.forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var res_kind_tmp = [];
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          for(let i = 0; i < that.classificationYear.length; i++) {
            if (t === that.classificationYear[i]){
              res_tmp.push(1);
              tmp = tmp.concat(data1[i]);
              sum_tmp += that.data_year[i].count;
              res_kind_tmp.push(i)
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      } else if(that.selectFeature === "education") {
        legendlist["class"] = "Education Level";
        // color_kind = color(0);
        feature_length = that.classificationEducation.length;
        count = new Set(that.classificationEducation).size;
        // eslint-disable-next-line no-unused-vars
        let data1 = [[], [], [], [], []];
        if(that.select_feature[1]){
          for(let i = 0; i < that.currentPoint.length; i++)
          {
            data1[that.data_behind[that.currentPoint[i]][1]].push(that.currentPoint[i])
          }
        }
        let setdata = new Set(that.classificationEducation);
        setdata.forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var res_kind_tmp = [];
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          for(let i = 0; i < that.classificationEducation.length; i++) {
            if (t === that.classificationEducation[i]){
              res_tmp.push(1);
              tmp = tmp.concat(data1[i]);
              sum_tmp += that.data_education[i].count;
              res_kind_tmp.push(i)
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      } else if(that.selectFeature === "maritalStatus") {
        legendlist["class"] = "Marital Status";
        // color_kind = color(0);
        feature_length = that.classificationMaritalStatus.length;
        count = new Set(that.classificationMaritalStatus).size;
        // eslint-disable-next-line no-unused-vars
        let data1 = [[], [], [], [], [], [], [], []];
        if(that.select_feature[2]){
          for(let i = 0; i < that.currentPoint.length; i++)
          {
            data1[that.data_behind[that.currentPoint[i]][2]].push(that.currentPoint[i])
          }
        }
        let setdata = new Set(that.classificationMaritalStatus);
        setdata.forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var res_kind_tmp = [];
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          for(let i = 0; i < that.classificationMaritalStatus.length; i++) {
            if (t === that.classificationMaritalStatus[i]){
              res_tmp.push(1);
              tmp = tmp.concat(data1[i]);
              sum_tmp += that.data_maritalStatus[i].count;
              res_kind_tmp.push(i)
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      }else if(that.selectFeature === "income") {
        legendlist["class"] = "Income";
        // color_kind = color(0);
        feature_length = that.classificationIncome.length;
        count = new Set(that.classificationIncome).size;
        // eslint-disable-next-line no-unused-vars
        let data1 = [[], [], [], [], [], [], [], [], [], [], []];
        if(that.select_feature[3]){
          for(let i = 0; i < that.currentPoint.length; i++)
          {
            data1[that.data_behind[that.currentPoint[i]][3]].push(that.currentPoint[i])
          }
        }
        let setdata = new Set(that.classificationIncome);
        setdata.forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var res_kind_tmp = [];
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          for(let i = 0; i < that.classificationIncome.length; i++) {
            if (t === that.classificationIncome[i]){
              res_tmp.push(1);
              tmp = tmp.concat(data1[i]);
              sum_tmp += that.data_income[i].count;
              res_kind_tmp.push(i)
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      }else if(that.selectFeature === "kidhome") {
        legendlist["class"] = "Kidhome";
        // color_kind = color(0);
        feature_length = that.classificationKidhome.length;
        count = new Set(that.classificationKidhome).size;
        // eslint-disable-next-line no-unused-vars
        let data1 = [[], [], []];
        if(that.select_feature[4]){
          for(let i = 0; i < that.currentPoint.length; i++)
          {
            data1[that.data_behind[that.currentPoint[i]][4]].push(that.currentPoint[i])
          }
        }
        let setdata = new Set(that.classificationKidhome);
        setdata.forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var res_kind_tmp = [];
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          for(let i = 0; i < that.classificationKidhome.length; i++) {
            if (t === that.classificationKidhome[i]){
              res_tmp.push(1);
              tmp = tmp.concat(data1[i]);
              sum_tmp += that.data_kidhome[i].count;
              res_kind_tmp.push(i)
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      }else if(that.selectFeature === "teenhome") {
        legendlist["class"] = "teenhome";
        // color_kind = color(0);
        feature_length = that.classificationTeenhome.length;
        count = new Set(that.classificationTeenhome).size;
        // eslint-disable-next-line no-unused-vars
        let data1 = [[], [], []];
        if(that.select_feature[5]){
          for(let i = 0; i < that.currentPoint.length; i++)
          {
            data1[that.data_behind[that.currentPoint[i]][5]].push(that.currentPoint[i])
          }
        }
        let setdata = new Set(that.classificationTeenhome);
        setdata.forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var res_kind_tmp = [];
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          for(let i = 0; i < that.classificationTeenhome.length; i++) {
            if (t === that.classificationTeenhome[i]){
              res_tmp.push(1);
              tmp = tmp.concat(data1[i]);
              sum_tmp += that.data_teenhome[i].count;
              res_kind_tmp.push(i)
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      }else if(that.selectFeature === "recency") {
        legendlist["class"] = "recency";
        // color_kind = color(0);
        feature_length = that.classificationRecency.length;
        count = new Set(that.classificationRecency).size;
        // eslint-disable-next-line no-unused-vars
        let data1 = [[], [], [], [], [], [], [], [], [], []];
        if(that.select_feature[6]){
          for(let i = 0; i < that.currentPoint.length; i++)
          {
            data1[that.data_behind[that.currentPoint[i]][6]].push(that.currentPoint[i])
          }
        }
        let setdata = new Set(that.classificationRecency);
        setdata.forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var res_kind_tmp = [];
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          for(let i = 0; i < that.classificationRecency.length; i++) {
            if (t === that.classificationRecency[i]){
              res_tmp.push(1);
              tmp = tmp.concat(data1[i]);
              sum_tmp += that.data_recency[i].count;
              res_kind_tmp.push(i)
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      }else if(that.selectFeature === "complain") {
        legendlist["class"] = "complain";
        // color_kind = color(0);
        feature_length = that.classificationComplain.length;
        count = new Set(that.classificationComplain).size;
        // eslint-disable-next-line no-unused-vars
        let data1 = [[], []];
        if(that.select_feature[7]){
          for(let i = 0; i < that.currentPoint.length; i++)
          {
            data1[that.data_behind[that.currentPoint[i]][7]].push(that.currentPoint[i])
          }
        }
        let setdata = new Set(that.classificationComplain);
        setdata.forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var res_kind_tmp = [];
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          for(let i = 0; i < that.classificationComplain.length; i++) {
            if (t === that.classificationComplain[i]){
              res_tmp.push(1);
              tmp = tmp.concat(data1[i]);
              sum_tmp += that.data_complain[i].count;
              res_kind_tmp.push(i)
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      }
    } else if(that.dataset_num == 5) {
      if(that.selectFeature === "continent"){
        legendlist["class"] = "Continent";
        // color_kind = color(0);
        feature_length = that.classificationContinen.length;
        count = new Set(that.classificationContinen).size;
        // eslint-disable-next-line no-unused-vars
        let data1 = [[], [], [], [], [], []];
        if(that.select_feature[0]){
          for(let i = 0; i < that.currentPoint.length; i++)
          {
            data1[that.data_behind[that.currentPoint[i]][0]].push(that.currentPoint[i])
          }
        }
        let setdata = new Set(that.classificationContinen);
        setdata.forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var res_kind_tmp = [];
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          for(let i = 0; i < that.classificationContinen.length; i++) {
            if (t === that.classificationContinen[i]){
              res_tmp.push(1);
              tmp = tmp.concat(data1[i]);
              sum_tmp += that.data_continent[i].count;
              res_kind_tmp.push(i)
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      } else if(that.selectFeature === "population") {
        legendlist["class"] = "Population";
        // color_kind = color(0);
        feature_length = that.classificationPopulation.length;
        count = new Set(that.classificationPopulation).size;
        // eslint-disable-next-line no-unused-vars
        let data1 = [[], [], [], [], [], []];
        if(that.select_feature[1]){
          for(let i = 0; i < that.currentPoint.length; i++)
          {
            data1[that.data_behind[that.currentPoint[i]][1]].push(that.currentPoint[i])
          }
        }
        let setdata = new Set(that.classificationPopulation);
        setdata.forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var res_kind_tmp = [];
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          for(let i = 0; i < that.classificationPopulation.length; i++) {
            if (t === that.classificationPopulation[i]){
              res_tmp.push(1);
              tmp = tmp.concat(data1[i]);
              sum_tmp += that.data_population[i].count;
              res_kind_tmp.push(i)
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      } else if(that.selectFeature === "population_density") {
        legendlist["class"] = "population Density";
        // color_kind = color(0);
        feature_length = that.classificationPopulationDensity.length;
        count = new Set(that.classificationPopulationDensity).size;
        // eslint-disable-next-line no-unused-vars
        let data1 = [[], [], [], []];
        if(that.select_feature[2]){
          for(let i = 0; i < that.currentPoint.length; i++)
          {
            data1[that.data_behind[that.currentPoint[i]][2]].push(that.currentPoint[i])
          }
        }
        let setdata = new Set(that.classificationPopulationDensity);
        setdata.forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var res_kind_tmp = [];
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          for(let i = 0; i < that.classificationPopulationDensity.length; i++) {
            if (t === that.classificationPopulationDensity[i]){
              res_tmp.push(1);
              tmp = tmp.concat(data1[i]);
              sum_tmp += that.data_population_density[i].count;
              res_kind_tmp.push(i)
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      }else if(that.selectFeature === "median_age") {
        legendlist["class"] = "Median Age";
        // color_kind = color(0);
        feature_length = that.classificationMedianAge.length;
        count = new Set(that.classificationMedianAge).size;
        // eslint-disable-next-line no-unused-vars
        let data1 = [[], [], [], []];
        if(that.select_feature[3]){
          for(let i = 0; i < that.currentPoint.length; i++)
          {
            data1[that.data_behind[that.currentPoint[i]][3]].push(that.currentPoint[i])
          }
        }
        let setdata = new Set(that.classificationMedianAge);
        setdata.forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var res_kind_tmp = [];
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          for(let i = 0; i < that.classificationMedianAge.length; i++) {
            if (t === that.classificationMedianAge[i]){
              res_tmp.push(1);
              tmp = tmp.concat(data1[i]);
              sum_tmp += that.data_median_age[i].count;
              res_kind_tmp.push(i)
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      }else if(that.selectFeature === "aged_65_older") {
        legendlist["class"] = "Age >65";
        // color_kind = color(0);
        feature_length = that.classificationAged65Older.length;
        count = new Set(that.classificationAged65Older).size;
        // eslint-disable-next-line no-unused-vars
        let data1 = [[], [], [], [], []];
        if(that.select_feature[4]){
          for(let i = 0; i < that.currentPoint.length; i++)
          {
            data1[that.data_behind[that.currentPoint[i]][4]].push(that.currentPoint[i])
          }
        }
        let setdata = new Set(that.classificationAged65Older);
        setdata.forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var res_kind_tmp = [];
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          for(let i = 0; i < that.classificationAged65Older.length; i++) {
            if (t === that.classificationAged65Older[i]){
              res_tmp.push(1);
              tmp = tmp.concat(data1[i]);
              sum_tmp += that.data_aged_65_older[i].count;
              res_kind_tmp.push(i)
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      }else if(that.selectFeature === "gdp_per_capita") {
        legendlist["class"] = "Capita GDP";
        // color_kind = color(0);
        feature_length = that.classificationCapitaGDP.length;
        count = new Set(that.classificationCapitaGDP).size;
        // eslint-disable-next-line no-unused-vars
        let data1 = [[], [], [], [], [], []];
        if(that.select_feature[5]){
          for(let i = 0; i < that.currentPoint.length; i++)
          {
            data1[that.data_behind[that.currentPoint[i]][5]].push(that.currentPoint[i])
          }
        }
        let setdata = new Set(that.classificationCapitaGDP);
        setdata.forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var res_kind_tmp = [];
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          for(let i = 0; i < that.classificationCapitaGDP.length; i++) {
            if (t === that.classificationCapitaGDP[i]){
              res_tmp.push(1);
              tmp = tmp.concat(data1[i]);
              sum_tmp += that.data_gdp_per_capita[i].count;
              res_kind_tmp.push(i)
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      }else if(that.selectFeature === "human_development_index") {
        legendlist["class"] = "Human Development Index";
        // color_kind = color(0);
        feature_length = that.classificationHumanDevelopmentIndex.length;
        count = new Set(that.classificationHumanDevelopmentIndex).size;
        // eslint-disable-next-line no-unused-vars
        let data1 = [[], [], [], [], [], []];
        if(that.select_feature[6]){
          for(let i = 0; i < that.currentPoint.length; i++)
          {
            data1[that.data_behind[that.currentPoint[i]][6]].push(that.currentPoint[i])
          }
        }
        let setdata = new Set(that.classificationHumanDevelopmentIndex);
        setdata.forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var res_kind_tmp = [];
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          for(let i = 0; i < that.classificationHumanDevelopmentIndex.length; i++) {
            if (t === that.classificationHumanDevelopmentIndex[i]){
              res_tmp.push(1);
              tmp = tmp.concat(data1[i]);
              sum_tmp += that.data_human_development_index[i].count;
              res_kind_tmp.push(i)
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      }
    } else {
      if(that.selectFeature === "week"){
        legendlist["class"] = "Week";
        // color_kind = color(0);
        feature_length = that.classificationWeek.length;
        count = new Set(that.classificationWeek).size;
        // eslint-disable-next-line no-unused-vars
        var data1 = [[], [], [], [], [], [], []];
        if(that.select_feature[0]){
          for(let i = 0; i < that.currentPoint.length; i++)
          {
            data1[that.data_behind[that.currentPoint[i]][0] - 1].push(that.currentPoint[i])
          }
        }
        var setdata = new Set(that.classificationWeek);
        setdata.forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var res_kind_tmp = [];
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          for(let i = 0; i < that.classificationWeek.length; i++) {
            if (t === that.classificationWeek[i]){
              res_tmp.push(1);
              tmp = tmp.concat(data1[i]);
              sum_tmp += that.data_week[i].count;
              res_kind_tmp.push(i)
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      } else if(that.selectFeature === "hour") {
        legendlist["class"] = "Hour";
        // color_kind = color(1);
        feature_length = that.classificationHour.length;
        count = new Set(that.classificationHour).size;
        // eslint-disable-next-line no-unused-vars
        var data2 = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []];
        if(that.select_feature[1]) {
          for(let i = 0; i < that.currentPoint.length; i++) {
            data2[that.data_behind[that.currentPoint[i]][that.select_feature[1] + that.select_feature[0] - 1]].push(that.currentPoint[i])
          }
        }
        new Set(that.classificationHour).forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          var res_kind_tmp = [];
          for(let i = 0; i < that.classificationHour.length; i++)
          {
            if (t === that.classificationHour[i]){
              res_tmp.push(1);
              res_kind_tmp.push(i)
              tmp = tmp.concat(data2[i])
              sum_tmp += that.data_hour[i].count;
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      }else if(that.selectFeature === "type"){
        legendlist["class"] = "Criminal Type";
        // color_kind = color(2);
        feature_length = that.classificationType.length;
        count = new Set(that.classificationType).size;
        // eslint-disable-next-line no-unused-vars
        var data3 = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []];
        if(that.select_feature[2]){
          for(let i = 0; i < that.currentPoint.length; i++)
          {
            data3[that.data_behind[that.currentPoint[i]][that.select_feature[2] + that.select_feature[1] + that.select_feature[0] - 1]].push(that.currentPoint[i])
          }
        }
        new Set(that.classificationType).forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          var res_kind_tmp = [];
          for(let i = 0; i < that.classificationType.length; i++)
          {
            if (t === that.classificationType[i]){
              res_tmp.push(1);
              tmp = tmp.concat(data3[i])
              sum_tmp += that.data_type[i].count;
              res_kind_tmp.push(i)
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      }else if(that.selectFeature === "location"){
        legendlist["class"] = "Location Type";
        // color_kind = color(3);
        feature_length = that.classificationLocation.length;
        count = new Set(that.classificationLocation).size;
        // eslint-disable-next-line no-unused-vars
        var data4 = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], []];
        if(that.select_feature[3]){
          for(let i = 0; i < that.currentPoint.length; i++)
          {
            data4[that.data_behind[that.currentPoint[i]][that.select_feature[3] + that.select_feature[2] + that.select_feature[1] + that.select_feature[0]- 1]].push(that.currentPoint[i])
          }
        }
        new Set(that.classificationLocation).forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          var res_kind_tmp = [];
          for(let i = 0; i < that.classificationLocation.length; i++)
          {
            if (t === that.classificationLocation[i]){
              res_tmp.push(1);
              tmp = tmp.concat(data4[i])
              sum_tmp += that.data_location[i].count;
              res_kind_tmp.push(i)
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      }else if(that.selectFeature === "position"){
        legendlist["class"] = "Location";
        // color_kind = color(4);
        feature_length = that.classificationPosition.length;
        count = new Set(that.classificationPosition).size;
        // eslint-disable-next-line no-unused-vars
        var data5 = [[], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []];
        if(that.select_feature[4]){
          for(let i = 0; i < that.currentPoint.length; i++)
          {
            var tmpc = that.select_feature[4] + that.select_feature[3] + that.select_feature[2] + that.select_feature[1] + that.select_feature[0] - 1;
            data5[that.data_behind[that.currentPoint[i]][tmpc]].push(that.currentPoint[i])
          }
        }
        new Set(that.classificationPosition).forEach(t=>{
          if(t === 1){
            count--;
            return;
          }
          var tmp = [];
          var sum_tmp = 0;
          var res_tmp = [];
          var res_kind_tmp = [];
          for(let i = 0; i < that.classificationPosition.length; i++)
          {
            if (t === that.classificationPosition[i]){
              res_tmp.push(1);
              tmp = tmp.concat(data5[i])
              sum_tmp += that.data_position[i].count;
              res_kind_tmp.push(i)
            }
            else
              res_tmp.push(0);
          }
          sum_total += sum_tmp;
          sum.push(sum_tmp);
          res.push(tmp);
          res_kind.push(res_tmp);
          res_kind_list.push(res_kind_tmp);
        })
      }
    }

    if(!that.points[that.active].is_leaves && that.active >= 0){
      d3.selectAll(".kind" + that.embeddingCount + "_" + that.points[that.active].state).classed("delete", false);
      for (let i = 0; i < that.points.length; i++){
        let tmp = i;
        while(tmp >= 0){
          if(that.points[tmp].father === that.points[that.active].state){
            that.points[i].show = false;
            color_list.push(palette(that.points[i].state));
            break;
          }
          tmp = that.points[tmp].father;
        }
      }

      palette = d3.scaleOrdinal()
          .domain(paletteList)
          .range(color_list)
          .unknown("#ccc");

      that.gen_legend();
      that.updateLines(1);
      that.updatePoints();
    }

    for (let i = 0; i < count; i++) {
      that.treeNodeCount ++;
      var ages = [];
      for(let j = 0; j < feature_length; j++) {
        ages.push({
          age: j + '',
          population: scale(sum[i]) / feature_length,
          opacity: res_kind[i][j] * 0.5 + 0.5,
          color: palette(that.treeNodeCount)
        });
      }
      var point = {
        state: that.treeNodeCount,
        sum: 200,
        show: true,
        father: that.active,
        weight: ((sum[i] / that.sum_total)).toFixed(2),
        data_set: res[i],
        is_leaves: true,
        feature_length: feature_length,
        x: Math.random() * 373,
        y: Math.random() * 676,
        ages: ages,
        legendList: {
          className :  legendlist["class"],
          list : res_kind_list[i],
          is_leaves : true,
          fatherNode: parseInt(that.active) - 1,
          color: palette(that.treeNodeCount)
        }
      };
      links.push([parseInt(that.active), that.treeNodeCount]);
      curveTypes.push({
        name: 'curveLinear',
        curve: d3.curveLinear,
        active: true,
        lineString: '',
        clear: false,
        info: ''
      });
      that.points.push(point);
      pointsVisible.push(true);
    }

    that.points_gen_color();
    that.active = -1;
    that.gen_legend();
    that.update();
  }