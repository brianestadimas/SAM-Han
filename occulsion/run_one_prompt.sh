strr[0]="single-test"
strr[1]="multi-test"
#gpu show
nn=0
total=${#strr[@]}
while [ $nn -lt $total ]
do

echo "$nn: ${strr[$nn]##*,}"
let nn++

done

read -p "please choose, cmd, exit is [e] or [E]:" strinput

if [ "$strinput" == "e" ] || [ "$strinput" == "E" ];then
echo "exit!!!"
return
fi

if [ $strinput -lt $total ];then
    strcmd=${strr[$strinput]}
    strcmd=${strcmd%%,*}
else
    strcmd=$strinput;
fi

if [ "$strcmd" == "single-test" ];then
    # image_choice_list=('sa_227195.jpg' 'sa_228197.jpg' 'sa_232600.jpg' 'sa_233167.jpg' 'sa_234809.jpg')
    image_choice=('sa_227195.jpg')
    drop_ratio_list=(0.1)
    #drop_ratio_list=(0.1 0.2 0.4 0.6 0.8)
    model_list=('vit_b' 'vit_l' 'vit_h')
    model_path=('../pretrain_model/sam_vit_b_01ec64.pth' \
            '../pretrain_model/sam_vit_l_0b3195.pth' '../pretrain_model/sam_vit_h_4b8939.pth')
    #image_choice=('sa_234809.jpg')
    model_num=${#model_list[@]}
    for ((i=0;i<model_num;i++));
    do
        for drop_ratio in ${drop_ratio_list[@]};
        do
            CUDA_VISIBLE_DEVICES=0 python ./occulsion/sam_han_train.py \
                    --optional "$image_choice" \
                    --drop_ratio $drop_ratio \
                    --sam_checkpoint ${model_path[i]} \
                    --model_type ${model_list[i]} \
                    --seed 1234 \
                    --point_coords_x 512 \
                    --point_coords_y 512
        done
    done
fi

if [ "$strcmd" == "multi-test" ];then
    #--optional "${image_choice%.*}" \
    # ('sa_227195.jpg' 'sa_228197.jpg' 'sa_232600.jpg' 'sa_233167.jpg' 'sa_234809.jpg') 
    occulsion_image_list=$(ls /home/qiaoyu/SAM_Robustness/sam_data100)
    drop_ratio_list=(0.1 0.2 0.4 0.6 0.8)
    model_list=('vit_b' 'vit_l' 'vit_h')
    model_path=('../pretrain_model/sam_vit_b_01ec64.pth' \
            '../pretrain_model/sam_vit_l_0b3195.pth' '../pretrain_model/sam_vit_h_4b8939.pth')
    model_num=${#model_list[@]}
    # /home/qiaoyu/SAM_Robustness/pretrain_model/sam_vit_b_01ec64.pth
    # /home/qiaoyu/SAM_Robustness/pretrain_model/sam_vit_l_0b3195.pth
    for ((i=0;i<model_num;i++));
    do
        for drop_ratio in ${drop_ratio_list[@]};
        do
            for image_choice in ${occulsion_image_list[@]};
            do 
                CUDA_VISIBLE_DEVICES=0 python sam_han_train.py  \
                    --optional "$image_choice" \
                    --sam_checkpoint ${model_path[i]} \
                    --model_type ${model_list[i]} \
                    --drop_ratio $drop_ratio \
                    --seed 1234
            done
        done
    done
fi