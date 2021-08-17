python run_oe_clevrer.py \
    --gt_flag 0 \
    --use_event_ann 0 \
    --program_path /home/zfchen/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v11_2/open_end_questions.json \
    --question_path /home/zfchen/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v11_2/open_end_questions.json \
    --raw_motion_prediction_dir /home/zfchen/code/output/render_output_disk2/prediction_v11_v2 \
    --ann_dir /home/zfchen/code/output/render_output/causal_sim_v11_4 \
    --gt_flag 1 \
    --num_sim 5000 \
    --invalid_video_fn /home/zfchen/code/clevrer_dataset_generation_v2/models/CLEVRER/executor_clevrer/data/invalid_video_v11.txt \
    #--ann_dir /home/zfchen/code/output/ns-vqa_output/v11_prp_refine/config \
