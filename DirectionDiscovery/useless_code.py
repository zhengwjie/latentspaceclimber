# z.requires_grad_()
# z.retain_grad()
# print(z)
# image=G(z)
# print(image.shape)
# image.sum().backward()
# # print(z.grad)
# print(z.grad.sum())
# print(z)

# 这种计算梯度的结果是有问题的
# 
img=G(z).squeeze()
z.requires_grad_()
z.retain_grad()

# for i in range(32):
#     for j in range(32):
#         z.grad.zero_()
# #         print(image[0][0][i][j])
#         torch.autograd.grad(outputs=img[0][0][i][j],create_graph=True,inputs=z,allow_unused=True)
#         a=jacobian_G[i][j]
#         b=z.grad
#         print(torch.max(b-a))
        
# if __name__=="__main__":
#     # prepare the file path to load the models
#     with torch.no_grad():
#         start_time=time.time()
#         rob_predictor_path = "./weights/rob_predictor/rob_predictor.pt"
#         classifier_path = "./weights/mnist/mnist_lenet5.pt"
#         deformator_dir="./models/pretrained/deformators/SN_MNIST/"
#         G_weights='./models/pretrained/generators/SN_MNIST/'
#         # load all the model
#         G,deformator=load_generator(deformator_dir,G_weights)
        
#         inspect_all_directions(G,deformator,"./mnists/dir_all",num_z=3)
        
        # classifier,rob_predictor_model=load_rob_predictor(classifier_path,rob_predictor_path)

        # # generate the plane
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # z=make_noise(1, G.dim_z).cuda()
        # directions=[8,2]
        # shifts_count=4
        # plane_images=generate_plane(G,deformator,directions,z,shifts_count=shifts_count)
        # original_img=plane_images[int(shifts_count/2)][int(shifts_count/2)]
        # new_plane_images=[]
        # for row in plane_images:
        #     row_images=[]
        #     for column in row:
        #         row_images.append(column-original_img)
        #     new_plane_images.append(row_images)
        # plane_to_image(plane_images,"./graphs.png")
        # plane_to_image(new_plane_images,"./new_graphs.png")
                
        # print(plane_images)
        # for row in plane_images:
        #     for img in row:

        #         print(img)

        # get the robustness value
        # normalize=Normalize(-1,2)
        # resize=Resize((28,28))

        # robustness_values=get_robustness_plane_values(classifier,rob_predictor_model,plane_images,normalize,resize)
        # print(robustness_values)
        # end_time=time.time()
        # print(end_time-start_time)
# what do you need？
# a generator ===> image
# rob_predictor
# robuness_value
# 网格数据
# 根据网格数据绘制contour图片

def feature_vector_images(
        generator,
        z,
        feature_vector_x,
        feature_vector_y,
        domain_x="0,5",domain_y="0,5",
        row_number=11,
        discriminator=None,
        feature_extractor=None,
        model_name=None,
        knn_model=None,
        knn_dataset=None,
        given_y=None,
        given_w=None):
    if model_name=="molecular_drug":
        return drug_generator.feature_vector_moleculars(
        generator,
        z,
        feature_vector_x,
        feature_vector_y,
        domain_x=domain_x,domain_y=domain_y,
        row_number=row_number,
        discriminator=discriminator,
        feature_extractor=feature_extractor,
        model_name=model_name,
        knn_model=knn_model,
        knn_dataset=knn_dataset)
    if model_name=="house_design":
        return housegan_utils.feature_vector_housedesigns(
        generator,
        z,
        given_y,given_w,
        feature_vector_x,
        feature_vector_y,
        domain_x=domain_x,domain_y=domain_y,
        row_number=row_number,
        discriminator=discriminator,
        feature_extractor=feature_extractor,
        model_name=model_name,
        knn_model=knn_model,
        knn_dataset=knn_dataset,
        )

    shifted_imgs=[]
    domain_x_vector=domain_x.split(",")
    domain_y_vector=domain_y.split(",")
    shift_x_l=float(domain_x_vector[0])
    shift_x_r=float(domain_x_vector[1])
    shift_y_l=float(domain_y_vector[0])
    shift_y_r=float(domain_y_vector[1])
    mean=torch.zeros(z.shape[1])
    cov=torch.ones(z.shape[1])
    final_z=torch.randn(0,z.shape[1]).to(device=device)
    original_point_x=int((0-shift_x_l)/((shift_x_r-shift_x_l) / (row_number-1)))
    original_point_y=int((0-shift_y_l)/((shift_y_r-shift_y_l) / (row_number-1)))
    count=0
    for i in np.arange(shift_y_l, shift_y_r+1e-9, (shift_y_r-shift_y_l) / (row_number-1)):
        for j in np.arange(shift_x_l, shift_x_r+1e-9, (shift_x_r-shift_x_l) / (row_number-1)):
            current_point_x=count%row_number
            current_point_y=count//row_number
            relative_loc_x=current_point_x-original_point_x
            relative_loc_y=current_point_y-original_point_y
            count=count+1
            shift_vector=j*feature_vector_x+i*feature_vector_y
            shift_vector=torch.tensor(shift_vector).to(device=device)
            z_plus_shift=shift_vector+z
            z_plus_shift=clamp_z(z_plus_shift,model_name)
            probability=stats.multivariate_normal.pdf(
                mean=mean,
                cov=cov,
                x=z_plus_shift.cpu()
            )
            probability=str(normalize_prob(probability,model_name=model_name))
            # img=generator(z+shift_vector)
            final_z=torch.cat((final_z,z_plus_shift),dim=0)
            realness=0
            # if discriminator:
            #     realness=discriminator(img).item()
            # img=to_image(img,True)
            shifted_imgs.append([None,probability,realness,
            str(j),str(i),
            feature_vector_x,feature_vector_y,
            None,None,
            relative_loc_x,relative_loc_y])
    final_imgs=generator(final_z)
    if discriminator:
        realnesses=discriminator(img)
    from DirectionDiscovery import novelty
    score=novelty.novelty_score(final_imgs,feature_extractor,model_name,knn_model)
    # 计算鲁棒性的值
    robust_values=get_robust(final_imgs,model_name)
    if model_name=="mnist":
        robust_values=robust_values/0.32
    else:
        robust_values=torch.log(robust_values)/(math.log(0.005)-math.log(0.001))
        print(robust_values)
    for i in range(len(final_imgs)):
        img=to_image(final_imgs[i],True)
        shifted_imgs[i][0]=img
        if discriminator:
            shifted_imgs[i][2]=realnesses[i].item()
        shifted_imgs[i][7]=score[i]
        shifted_imgs[i][8]=robust_values[i].item()
    return shifted_imgs

