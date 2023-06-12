


python eval_base_prompt.py \
        --exp-name test \
        --save-path occulsion/results \
        --model-type vit_b \
        --save-path /home/qiaoyu/SAM_Robustness/occulsion/results \
        --sam-checkpoint /home/qiaoyu/SAM_Robustness/pretrain_model/sam_vit_b_01ec64.pth \
        --path-img-list /home/qiaoyu/SAM_Robustness/select_100_new.txt \
        --path-clean-img-dir /home/qiaoyu/SAM_Robustness/occulsion/sam_sampling_100/sam_original \
        --path-perturbed-img-dir /home/qiaoyu/SAM_Robustness/occulsion/sam_sampling_100/sam_occulsion/drop_ratio_0.1
