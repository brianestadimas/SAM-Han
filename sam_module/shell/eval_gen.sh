


python eval_gen.py --exp-name test \
                   --save-path /home/ansible/ktg/sam_module \
                   --model-type vit_b \
                   --sam-checkpoint /home/ansible/ktg/SAM_NeurIPS2023/checkpoints/sam_vit_b_01ec64.pth \
                   --path-img-list /home/ansible/ktg/SAM/sam_image_list_100.txt \
                   --path-clean-img-dir /home/ansible/ktg/SAM/sam_data \
                   --path-perturbed-img-dir /data/dataset/sam_sampling_100/sam_fog/severity_1 \
                   --points-per-side 1 \
                   --multi-mask false \
                   --get-highest false