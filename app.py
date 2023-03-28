
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import torch
torch.cuda.empty_cache()
import flask
from flask_cors import CORS
from flask import Flask, request,render_template,make_response

#from main_hw import app_hw
from DirectionDiscovery import draw_graph,novelty
from torch_tools.visualization import to_image
from utils1 import make_noise
from DirectionDiscovery import prepare_novelty_overview
from housegan import housegan_utils 
from common_utils import get_child_vector
import json
app = Flask(__name__)
#app.register_blueprint(app_hw)
app.config['TEMPLATES_AUTO_RELOAD'] = True
CORS(app)

import base64
from io import BytesIO
import numpy as np
global_generator=None
global_discriminator=None
global_classifier=None
global_umap=None
global_umap_neural=None
global_rob_predictor=None
global_z=None
global_z_image=None
global_feature_value_vector=None
global_svd_images_alphas=None

global_candidate=None
global_candidate_image=None

global_candidate_feature_value_vector=None
global_candidate_svd_images_alphas=None
# 记录现在的主视图
global_centre=0
# global_feature_index 应该是一个字典
# 这个应该不需要更新
global_feature_index_dict={}

global_knn_model=None
global_knn_dataset=None
global_feature_extractor=None
global_model=None
# 默认是9
global_row_number=9

global_umap_model=None
global_overview_dataset=None

global_given_y=None
global_given_w=None
global_samples=None
global_domain_range=None
history_z=[]
history_image=[]
global_step=0
# 点击了之后怎么办
# 列表是不是也要更新
# 在列表后面补充
# 不修改列表
def refresh_global_values():
    global global_generator,global_discriminator
    global global_classifier,global_z,global_centre
    global global_model,global_row_number,history_z,global_step,history_image
    global_generator=None
    global_discriminator=None
    global_classifier=None
    global_z=None
    global_model=None
    # 记录现在的主视图
    global_centre=0
    # 默认是9
    global_row_number=9
    history_z=[]
    history_image=[]
    global_step=0

domainrange_dict={
    "mnist":2,
    "cifar10":1.5,
    "molecular_drug":0.3,
    "house_design":2
}

def image_to_base64(img):
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    binary_data = output_buffer.getvalue()
    base64_data = str(base64.b64encode(binary_data), encoding="utf-8")
    return base64_data
def get_domain(shift):
    if shift<0:
        domain=str(shift)+",0"
    elif shift==0:
        domain="0,0.0001"
    else:
        domain="0,"+str(shift)
    return domain

plane_images = []


# @app.route('/')
# def index():
#     return flask.send_from_directory('static', 'index.html')
@app.route('/', methods=['GET', 'POST'])
@app.route('/index', methods=['GET', 'POST'])
def index(name=None):
    return render_template('index.html', name = name)


@app.route('/contour')
def contour():
    return flask.send_from_directory('static', 'contour.html')

@app.route("/demo")
def demo():
    return flask.send_from_directory("static","demo.html")


# 清空所有的变量

@app.route('/robustness_contour')
def robustness_contour():
    refresh_global_values()
    
    return flask.send_from_directory('static', 'robustness_contour.html')

@app.route('/all_contour')
def all_contour():
    return flask.send_from_directory('static', 'all_contour.html')

@app.route('/defs')
def defs():
    return flask.send_from_directory('static', 'defs.html')


@app.route('/datajoin')
def datajoin():
    return flask.send_from_directory('static', 'datajoin.html')


@app.route('/keyfunction')
def keyfunction():
    return flask.send_from_directory('static', 'keyfunction.html')


@app.route("/quadtree")
def quadtree():
    return flask.send_from_directory("static", "quadtree.html")



import torch
device = "cuda" if torch.cuda.is_available() else "cpu"


# 要发送的数据是x轴起始的位置，终止的位置
# y轴起始的终止的位置
@app.route("/api/svd_images", methods=['GET',"POST"])
def get_svd_images():
    # 加上选择模型的过程
    domain_range=request.form.get("domain_range",type=float)
    return get_svd_plane(domain_range)

# 计算在一个点处的svd_images以及alphas
def get_svd_images_and_alphas(z,feature_value_vector,domain_range,
given_alphas=None,scale=None):
    global global_generator
    global global_discriminator
    global global_z
    global global_feature_value_vector

    global global_feature_extractor
    global global_knn_dataset
    global global_knn_model
    global global_model
    global global_row_number
    global global_given_w
    global global_given_y

    svd_images,alphas=draw_graph.generate_svd_plane(
    global_generator,
    z,
    given_y=global_given_y,
    given_w=global_given_w,
    domain_range=domain_range,
    feature_value_vector=feature_value_vector,
    given_alphas=given_alphas,
    discriminator=global_discriminator,
    feature_extractor=global_feature_extractor,
    model_name=global_model,
    knn_dataset=global_knn_dataset,
    knn_model=global_knn_model,
    scale=scale,
    shifts_count=global_row_number)
    return svd_images,alphas

def get_images_base64(z=None,domain_range=None,feature_value_vector=None,got_svd_images_and_alphas=None):
    global global_generator
    global global_discriminator
    global global_z
    global global_feature_value_vector

    global global_feature_extractor
    global global_knn_dataset
    global global_knn_model
    global global_model
    global global_row_number
    global global_given_w
    global global_given_y
    global history_z
    global history_image

    if z is None:
        z=global_z
    if feature_value_vector is None:
        feature_value_vector=global_feature_value_vector
    if domain_range is None:
        domain_range=domainrange_dict.get(global_model)

    if got_svd_images_and_alphas is None:
        got_svd_images_and_alphas=get_svd_images_and_alphas(z,feature_value_vector,domain_range)
    
    svd_images,alphas=got_svd_images_and_alphas
    
    images_base64=[]
    # feature_vector_indexs_one_dimension=[]
    for x in range(len(svd_images)):
        image_dict={
        "image":image_to_base64(svd_images[x][0]),
        'probability':svd_images[x][1],
        "realness":svd_images[x][2],
        "novelty_score":svd_images[x][3],
        "robustness":svd_images[x][4],
        "weights":str(alphas[x])
        }
        if global_model=='molecular_drug':
            # 'kor',"qed","logP","sascore"
            image_dict['kor']=str(svd_images[x][5])
            image_dict["qed"]=str(svd_images[x][6])
            image_dict["logP"]=str(svd_images[x][7])
            image_dict["sascore"]=str(svd_images[x][8])
        elif global_model=='house_design':
            # house gan的指标
            # variance_area,overlap,compatability
            image_dict['variance_area']=str(svd_images[x][5])
            image_dict["overlap"]=str(svd_images[x][6])
            image_dict["compatability"]=str(svd_images[x][7])

        images_base64.append(image_dict)
    return images_base64

# 确定global_z的时候，需要更新global_feature_value_vector以及alphas

# 在这里只进行计算
# 不要更新全局变量
def get_svd_plane(z=None,domain_range=None,feature_value_vector=None,got_svd_images_and_alphas=None):
    # 把global_feature_value_vector和global_feature_index更新一下
    global global_generator
    global global_discriminator
    global global_z
    global global_feature_value_vector

    global global_feature_extractor
    global global_knn_dataset
    global global_knn_model
    global global_model
    global global_row_number
    global global_given_w
    global global_given_y
    global history_z
    global history_image


    images_base64=get_images_base64(z,domain_range,feature_value_vector,got_svd_images_and_alphas)

    feature_value_rate,accumulating_contribute_rates=draw_graph.get_image_info(
        z,
        global_generator,
        global_model,
        feature_value_vector=feature_value_vector,
        given_y=global_given_y,
        given_w=global_given_w,
        rates=0.80
    )

    neighbor_images=draw_graph.get_neighbors(z,
                                             global_generator,
                                             global_model,
                                             global_knn_dataset,
                                             global_knn_model,
                                             feature_extractor=global_feature_extractor,
                                             feature_value_vector=feature_value_vector,
                                             given_y=global_given_y,
                                             given_w=global_given_w
                                             )
    for i in range(len(neighbor_images)):
        neighbor_images[i]=image_to_base64(neighbor_images[i])

    image_list=[]

    for i in range(len(history_image)):
        image_list.append(image_to_base64(history_image[i]))
    if global_candidate_image:
        image_list.append(image_to_base64(global_candidate_image))

    return flask.jsonify({
        "svd_images": images_base64,
        "feature_values":feature_value_rate,
        "accumulating_contribute_rate":accumulating_contribute_rates,
        "neighbor_images":neighbor_images,
        "domain_range":domain_range,
        "image_list":image_list
    })

@app.route("/info_pannel", methods=['GET','POST'])
def update_info_pannel():
    global global_feature_value_vector
    global global_model
    global global_z
    global global_generator
    global global_model
    global global_knn_dataset
    global global_knn_model
    global global_feature_extractor
    global global_given_w
    global global_given_y

    feature_index_x=int(request.form.get("feature_index_x"))
    feature_index_y=int(request.form.get("feature_index_y"))
    shift_x=float(request.form.get("shift_x"))
    shift_y=float(request.form.get("shift_y"))
    s,vh=global_feature_value_vector

    shift=shift_x*vh[feature_index_x]+shift_y*vh[feature_index_y]
    # 需要返回特征值和mask图片

    # feature_value_rate,accumulating_contribute_rates=draw_graph.get_image_info(global_z,
    # global_generator,
    # global_model,
    # shift=shift,
    # given_y=global_given_y,
    # given_w=global_given_w
    # )

    neighbor_images=draw_graph.get_neighbors(global_z,
                                             global_generator,
                                             global_model,
                                             global_knn_dataset,
                                             global_knn_model,
                                             feature_extractor=global_feature_extractor,
                                             feature_value_vector=global_feature_value_vector,
                                             shift=shift,
                                             given_y=global_given_y,
                                             given_w=global_given_w)
    for i in range(len(neighbor_images)):
        neighbor_images[i]=image_to_base64(neighbor_images[i])
    
    return flask.jsonify({
        "neighbor_images": neighbor_images
    })

@app.route("/zoom",methods=['GET','POST'])
def zoom():
    # 获取请求的数据
    # 然后请求图片数据
    # 还需要判断数据global_z还是global_candidate
    global global_centre

    global global_z
    global global_feature_value_vector
    global global_svd_images_alphas

    global global_candidate
    global global_candidate_svd_images_alphas
    global global_candidate_feature_value_vector
    global global_domain_range

    scale=request.form.get("scale",type=float)
    domain_range=request.form.get("domain_range",type=float)
    svg_mark=request.form.get("svg_mark",type=int)
    global_domain_range=domain_range

    if svg_mark==global_centre:
        # 更新的是global_z
        _,alphas=global_svd_images_alphas
        global_svd_images_alphas=get_svd_images_and_alphas(
        global_z,global_feature_value_vector,domain_range,
        given_alphas=alphas,scale=scale)
        return get_svd_plane(z=global_z,domain_range=domain_range,
        feature_value_vector=global_feature_value_vector,
        got_svd_images_and_alphas=global_svd_images_alphas)
    else:
        # 更新的是global_candidate
        _,alphas=global_candidate_svd_images_alphas
        global_candidate_svd_images_alphas=get_svd_images_and_alphas(
        global_candidate,global_candidate_feature_value_vector,domain_range,
        given_alphas=alphas,scale=scale)
        return get_svd_plane(z=global_candidate,domain_range=domain_range,
        feature_value_vector=global_candidate_feature_value_vector,
        got_svd_images_and_alphas=global_candidate_svd_images_alphas)
        
@app.route("/neighboors",methods=['GET','POST'])
def get_neighboors():
    feature_index_x=int(request.form.get("feature_index_x"))
    feature_index_y=int(request.form.get("feature_index_y"))
    shift_x=float(request.form.get("shift_x"))
    shift_y=float(request.form.get("shift_y"))

    global global_feature_value_vector
    global global_model
    global global_z
    global global_generator
    global global_model
    global global_knn_dataset
    global global_knn_model
    global global_feature_extractor
    global global_given_w
    global global_given_y

    s,vh=global_feature_value_vector

    shift=shift_x*vh[feature_index_x]+shift_y*vh[feature_index_y]
    
    # z,generator,model_name,dataset,nbrs,feature_extractor,feature_value_vector=None,shift=None
    neighboors=draw_graph.get_neighbors(
    global_z,
    global_generator,
    global_model,
    global_knn_dataset,
    global_knn_model,
    global_feature_extractor,
    shift=shift,
    given_w=global_given_w,
    given_y=global_given_y)
    for i in range(len(neighboors)):
        neighboors[i]=image_to_base64(neighboors[i])
    return flask.jsonify({
        "neighbors":neighboors
    })

@app.route("/select_model", methods=['GET','POST'])
def select_model():
    # 选中模型之后进行响应
    global global_generator
    global global_discriminator
    global global_centre
    global global_z
    global global_z_image
    
    global global_feature_value_vector
    global global_svd_images_alphas

    global global_feature_extractor
    global global_knn_dataset
    global global_knn_model
    global global_model
    global global_row_number
    global history_z
    global global_umap_model
    global global_overview_dataset
    global global_step
    global global_given_w
    global global_given_y
    global global_samples
    global history_image
    global global_domain_range


    history_z=[]
    history_image=[]
    # 默认中心在左边
    global_centre=0
    global_domain_range=None
    

    model_name=request.form.get("model_name")
    row_number=request.form.get("row_number",type=int)
    if(model_name=="Select Model"):
        global_generator=None

        global_feature_value_vector=None
        return flask.jsonify({
        "svd_images": "",
        "feature_vector_indexs":""})
    if model_name=="SNGAN(MNIST)":
        global_model="mnist"
    elif model_name=="StyleGAN2-ADA(CIFAR10)":
        global_model="cifar10"
    elif model_name=="drug_generator":
        global_model="molecular_drug"
    elif model_name=='housegan':
        global_model="house_design"
    else:
        print(model_name)

    global_generator, _ = draw_graph.load_generator_descriminator(
        global_model
    )

    global_feature_extractor=novelty.load_model()
    # global_feature_extractor 这个在drug数据集中暂时用不上

    global_knn_dataset=novelty.load_image_dataset(global_model)
    global_knn_model=novelty.load_knn_model(global_model)

    if global_model=="house_design":
        batch=housegan_utils.get_graph(global_knn_dataset,idx=0)
        _, nds, eds, _, _=batch
        global_given_y = nds.clone().detach().to(device)
        # 边的情况
        global_given_w = eds.clone().detach().to(device)
        real_nodes = np.where(nds.detach().cpu()==1)[-1]
        housegan_utils.draw_graph_function([real_nodes, eds.detach().cpu().numpy()])
    # 确定了global_z就需要更新另外两个值
    global_z=draw_graph.generate_init_noise(global_model,global_generator,
    global_given_y)
    global_feature_value_vector=draw_graph.get_feature_value_vector(
        global_generator,global_z,model_name=global_model,given_y=global_given_y,given_w=global_given_w)
    domain_range=domainrange_dict.get(global_model)
    global_domain_range=domain_range
    # print(global_feature_value_vector.shape)
    global_svd_images_alphas=get_svd_images_and_alphas(
        global_z,global_feature_value_vector,domain_range
    )
    svd_images,_=global_svd_images_alphas
    global_z_image=svd_images[len(svd_images)//2][0]
    
    global_row_number=row_number
    global_step=0
    global_umap_model=prepare_novelty_overview.get_umap(global_model)
    global_overview_dataset=prepare_novelty_overview.load_increase_dim_model(model_name)

    global_samples=None

    history_z.append(global_z)
    history_image.append(global_z_image)

    return get_svd_plane(z=global_z,domain_range=domain_range,
    feature_value_vector=global_feature_value_vector,
    got_svd_images_and_alphas=global_svd_images_alphas)


@app.route("/update_row_number",methods=['GET','POST'])
def update_row_number():
    global global_row_number

    global global_z
    global global_feature_value_vector
    global global_svd_images_alphas

    global global_candidate
    global global_candidate_image

    global global_candidate_svd_images_alphas
    global global_candidate_feature_value_vector

    global history_z
    global global_step
    global global_domain_range


    received_row_number=request.form.get("row_number",type=int)
    print(received_row_number)

    if global_row_number!=received_row_number:
        global_row_number=received_row_number
        # 重新请求左右的数据，并更新
        # 请求数据需要很多信息
        
        global_svd_images_alphas=get_svd_images_and_alphas(
        global_z,global_feature_value_vector,global_domain_range)
        z_image_base64=get_images_base64(global_z,global_domain_range,global_feature_value_vector)

        candidate_image_base64=[]
        if global_candidate is not None:
            global_candidate_svd_images_alphas=get_svd_images_and_alphas(
            global_candidate,global_candidate_feature_value_vector,global_domain_range)
            candidate_image_base64=get_images_base64(global_candidate,global_domain_range,global_candidate_feature_value_vector)
        if global_centre==0:

            return flask.jsonify({
                "left_svd_images":z_image_base64,
                "right_svd_images":candidate_image_base64,
                "domain_range":global_domain_range
            })
        else:
            return flask.jsonify({
                "left_svd_images":candidate_image_base64,
                "right_svd_images":z_image_base64,
                "domain_range":global_domain_range
            })
        # 这里需要一个function to send back the message
    return flask.jsonify({})
@app.route("/return_history_step",methods=['GET','POST'])
def return_history_step():
    global global_z
    global global_step
    global history_image
    global history_z
    global global_z_image
    global global_candidate
    global global_candidate_image
    global global_domain_range
    global global_feature_value_vector
    global global_svd_images_alphas
    global global_centre
    global_centre=0


    received_history_step=request.form.get("step_index",type=int)
    global_candidate_image=None
    global_candidate=None
    global_z=history_z[received_history_step]
    global_z_image=history_image[received_history_step]
    history_z=history_z[:received_history_step+1]
    history_image=history_image[:received_history_step+1]

    global_step=received_history_step+1

    global_feature_value_vector=draw_graph.get_feature_value_vector(
        global_generator,global_z,model_name=global_model,given_y=global_given_y,given_w=global_given_w)
    domain_range=domainrange_dict.get(global_model)
    global_domain_range=domain_range
    # print(global_feature_value_vector.shape)
    global_svd_images_alphas=get_svd_images_and_alphas(
        global_z,global_feature_value_vector,domain_range
    )

    return get_svd_plane(z=global_z,domain_range=domain_range,
    feature_value_vector=global_feature_value_vector,
    got_svd_images_and_alphas=global_svd_images_alphas)




@app.route("/recycle",methods=['GET','POST'])
def recycle():
    global history_z
    global history_image
    global global_z
    global global_feature_value_vector
    global global_svd_images_alphas

    global global_candidate
    global global_candidate_image
    global global_candidate_svd_images_alphas
    global global_candidate_feature_value_vector
    global global_generator
    global global_given_y
    global global_model
    global global_given_w


    global_z=draw_graph.generate_init_noise(global_model,global_generator,
    global_given_y)
    history_z=history_z[:1]
    history_image=history_image[:1]

    global_feature_value_vector=draw_graph.get_feature_value_vector(
        global_generator,global_z,model_name=global_model,given_y=global_given_y,given_w=global_given_w)
    domain_range=domainrange_dict.get(global_model)
    global_svd_images_alphas=get_svd_images_and_alphas(
        global_z,global_feature_value_vector,domain_range
    )
    global_candidate=None
    global_candidate_image=None
    global_candidate_svd_images_alphas=None
    global_candidate_feature_value_vector=None

    return get_svd_plane(z=global_z,domain_range=domain_range,
    feature_value_vector=global_feature_value_vector,
    got_svd_images_and_alphas=global_svd_images_alphas)

# 这里默认是已经有candidate
def step_forward(child_index):
    global global_z
    global global_z_image
    global global_feature_value_vector
    global global_svd_images_alphas

    global global_candidate
    global global_candidate_image

    global global_candidate_svd_images_alphas
    global global_candidate_feature_value_vector
    global global_step
    global history_z
    global history_image
    global_step=global_step+1
    global_z=global_candidate
    global_z_image=global_candidate_image

    global_feature_value_vector=global_candidate_feature_value_vector
    global_svd_images_alphas=global_candidate_svd_images_alphas
    
    history_z.append(global_z)
    history_image.append(global_z_image)
    
    pass

@app.route("/select_image", methods=['GET','POST'])
def select_image():
    # 选中左边，展示右边
    # 选中右边，更新左边
    # 需要知道当前选中的图片的来源
    # 如果它现在显示主图在左边，选中的图片来自右边，那么就需要更新 global_z
    # 如果它global_z在右边，选中的图片来自左边，那么就也需要更新global_z
    # 更新global_z 需要用到特征向量以及其步长
    global global_centre
    global global_model
    global global_z
    global global_feature_value_vector
    global global_svd_images_alphas

    global global_candidate
    global global_candidate_image

    global global_candidate_svd_images_alphas
    global global_candidate_feature_value_vector

    global history_z
    global global_step
    global global_domain_range

    # mark==0 表示选择的数据来自左边
    # mark==1 表示选择的数据来自右边
    mark=request.form.get("mark",type=int)
    child_index=request.form.get("child_index",type=int)
    # 计算next_candidate
    if mark!=global_centre:
        step_forward(child_index)
        global_centre=1-global_centre
    

    # set up candidate
    # 这个时候，从global_z所在的主视图中选择图片产生新的树的节点
    # 计算candidate
    # 计算特征值和特征向量
    # 计算svd_images以及alphas
    svd_images,alphas=global_svd_images_alphas
    alpha=alphas[child_index]


    global_candidate=get_child_vector(global_z,global_feature_value_vector,alpha,global_model,global_domain_range)

    global_candidate_feature_value_vector=draw_graph.get_feature_value_vector(
    global_generator,global_candidate,model_name=global_model,given_y=global_given_y,given_w=global_given_w)
    domain_range=domainrange_dict[global_model]
    global_candidate_svd_images_alphas=get_svd_images_and_alphas(
        global_candidate,global_candidate_feature_value_vector,domain_range
    )
    candidate_svd_images,candidate_alphas=global_candidate_svd_images_alphas
    global_candidate_image=candidate_svd_images[len(candidate_svd_images)//2][0]
    
    candidate_svd_images[len(candidate_svd_images)//2]=svd_images[child_index]
    global_candidate_svd_images_alphas=candidate_svd_images,candidate_alphas

    return get_svd_plane(z=global_candidate,domain_range=domain_range,
    feature_value_vector=global_candidate_feature_value_vector,
    got_svd_images_and_alphas=global_candidate_svd_images_alphas)



# @app.route("_pan_right_images", methods=['GET','POST'])
# def zoom_pan_right_images():
#     domain_x=request.form.get("domain_x")
#     domain_y=request.form.get("domain_y")
#     print(domain_x)
#     print(domain_y)

#     # return show_continuous_features(domain_x,domain_y)
#     return None

# next_step这里的逻辑还没有改过

@app.route("/next_step", methods=['GET','POST'])
def next_step():

    global global_z
    global global_generator
    # 全局的特征值
    global global_feature_value_vector
    global global_step
    global global_feature_index_dict
    global history_z
    global global_model
    feature_index_x=int(request.form.get("feature_index_x"))
    feature_index_y=int(request.form.get("feature_index_y"))
    shift_x=float(request.form.get("shift_x"))
    shift_y=float(request.form.get("shift_y"))
    # 保存历史记录


    s,vh=global_feature_value_vector
    feature_vector_x=vh[feature_index_x]
    feature_vector_y=vh[feature_index_y]
    shift_z=feature_vector_x*shift_x+feature_vector_y*shift_y
    next_z=draw_graph.get_nextz(global_z,shift_z,global_model)
    global_z=next_z
    global_feature_value_vector=None
    global_feature_index_dict={}
    history_z=history_z[:global_step+1]
    history_z.append(global_z)
    global_step=global_step+1

    return get_svd_plane()


@app.route("/prev_step", methods=['GET','POST'])
def prev_step():
    # 如果接收到了这个请求，就说明有前一步
    # 我们选择在前端判断判断当前的步数是否是0
    global global_z
    global global_step
    global global_feature_value_vector
    global global_feature_index_dict
    if global_step>=1:
        # 更新一下
        global_step=global_step-1
        global_z=history_z[global_step].to(device=device)
        global_feature_value_vector=None
        global_feature_index_dict={}
    return get_svd_plane()


@app.route("/samples",methods=['GET','POST'])
def get_samples():
    global global_generator
    # 获取数据
    # 降维
    global global_model
    global global_given_y
    global global_given_w
    global global_feature_extractor
    global global_samples
    global history_z
    # 产生30个样本
    imgs,zs,points=draw_graph.generate_samples(global_model,global_generator,global_given_y,global_given_w,global_feature_extractor)
    global_samples=zs
    img_infos=[]
    min_x=np.min(points[:,0])
    max_x=np.max(points[:,0])
    min_y=np.min(points[:,1])
    max_y=np.max(points[:,1])

    for i in range(len(imgs)):
        img_infos.append({
            "img":image_to_base64(imgs[i]),
            "point_x":str(points[i][0]),
            "point_y":str(points[i][1])
        })
    return flask.jsonify({
        "img_info":img_infos,
        "min_x":str(min_x),
        "min_y":str(min_y),
        "max_x":str(max_x),
        "max_y":str(max_y),
    })
@app.route("/select_sample",methods=['GET','POST'])
def select_sample():
    idx=request.form.get("selected_inx",type=int)
    global global_samples
    global global_step
    global global_feature_value_vector
    global global_z
    global global_svd_images_alphas
    global history_z
    global global_generator
    global global_model
    global global_given_y
    global global_given_w
    global history_image

    history_z=[]
    history_image=[]

    global_step=1
    if isinstance(global_samples,list):
        global_z=global_samples[idx].to(device=device)
    else:
        global_z=global_samples[idx:idx+1].to(device=device)
    print(global_z.shape)
    
    global_feature_value_vector=draw_graph.get_feature_value_vector(
        global_generator,global_z,model_name=global_model,given_y=global_given_y,given_w=global_given_w)
    domain_range=domainrange_dict.get(global_model)
    global_svd_images_alphas=get_svd_images_and_alphas(
        global_z,global_feature_value_vector,domain_range
    )
    svd_images,_=global_svd_images_alphas
    global_z_image=svd_images[len(svd_images)//2][0]
    
    history_z.append(global_z)
    history_image.append(global_z_image)
    return get_svd_plane(z=global_z,domain_range=domain_range,
    feature_value_vector=global_feature_value_vector,
    got_svd_images_and_alphas=global_svd_images_alphas)

@app.route("/novelty_overview",methods=['GET','POST'])
def view_novelty():
    global global_overview_dataset

    returned_novelty_scores,num_x,num_y=prepare_novelty_overview.load_novelty(
        model_name=global_model
    )

    print(len(returned_novelty_scores))
    print(returned_novelty_scores)
    print(num_x)
    print(num_y)
    import math
    for i in range(len(returned_novelty_scores)):
        if math.isnan(returned_novelty_scores[i]):
            returned_novelty_scores[i]=0
    print(returned_novelty_scores)
    return flask.jsonify({
        "novelty_scores":returned_novelty_scores,
        "num_x":num_x,
        "num_y":num_y,
        "min_x":str(global_overview_dataset[2]),
        "max_x":str(global_overview_dataset[3]),
        "min_y":str(global_overview_dataset[4]),
        "max_y":str(global_overview_dataset[5])
    })
# 选择一个点,查看图片
#
@app.route("/novelty_overview_select",methods=['GET','POST'])
def novelty_overview_select():
    global global_overview_dataset
    point_x=float(request.form.get("point_x"))
    point_y=float(request.form.get("point_y"))

    _,img=prepare_novelty_overview.increase_dim(point_x,point_y,global_overview_dataset,global_generator)
    img=image_to_base64(img)
    return flask.jsonify({
        "image":img
    })
# 在overview中选择了一个点去开启探索
@app.route("/novelty_overview_start_exploration",methods=['GET','POST'])
def novelty_overview_start_exploration():
    global global_overview_dataset
    global global_z
    global global_feature_value_vector
    global global_feature_index_dict
    global global_step
    point_x=float(request.form.get("point_x"))
    point_y=float(request.form.get("point_y"))

    z,_=prepare_novelty_overview.increase_dim(point_x,point_y,global_overview_dataset,global_generator)
    global_z=z
    history_z.append(global_z)
    global_feature_value_vector = None
    global_feature_index_dict = {}
    global_step=global_step+1

    return get_svd_plane()

#只需要查看历史的点
@app.route("/look_history_z",methods=['GET','POST'])
def look_history_z():
    global history_z
    global global_generator
    global global_feature_extractor
    global global_umap_model
    global global_model
    zs=torch.cat(history_z,dim=0)
    coordinate_in_2D,images=prepare_novelty_overview.get_coordinate_in_two_dim(
        zs,
        global_generator,
        global_feature_extractor,
        global_umap_model,
        global_model
    )
#     一方面需要坐标，一方面，需要请求图片
    history_info=[]
    for i in range(len(images)):
        history_info.append({
            "image":image_to_base64(images[i]),
            "point_x":str(coordinate_in_2D[i][0]),
            "point_y":str(coordinate_in_2D[i][1])
        })

    return flask.jsonify({
        "history_info":history_info
    })

@app.route("/x_y_zoom", methods=['GET'])
def get_x_y_zoom():
    return flask.send_from_directory("static", "x_y_zoom.html")

@app.route("/api/axes_points", methods=['GET'])
def get_axes_points():
    shifts_range=8
    shifts_count=20
    all_axis_points=[]
    for i in range(5):
        axis_points=draw_graph.get_axis_in_plane(i,shifts_range=shifts_range,
        shifts_count=shifts_count)
        all_axis_points.append(axis_points)
    return flask.jsonify({
        'axes_points': all_axis_points
    })
    


from flask import Flask, request, make_response
from datetime import datetime


@app.route('/display/img/<string:filename>', methods=['GET'])
def display_img(filename):
    request_begin_time = datetime.today()
    print("request_begin_time", request_begin_time)
    if request.method == 'GET':
        if filename is None:
            pass
        else:
            image_data = open(filename, "rb").read()
            response = make_response(image_data)
            print(response)
            response.headers['Content-Type'] = 'image/jpg'
            return response
    else:
        pass


app.run(host='0.0.0.0', port=11667, debug=True)