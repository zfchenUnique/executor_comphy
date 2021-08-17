python run_mc_clevrer.py \
    --gt_flag 1 \
    --use_event_ann 0 \
    --n_progs 5000 \
    --ann_dir /home/zfchen/code/output/render_output/causal_sim_v11_4 \
    --program_path /home/zfchen/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v11/multiple_choice_questions.json \
    --question_path /home/zfchen/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v11/multiple_choice_questions.json \
    --raw_motion_prediction_dir /home/zfchen/code/output/render_output_disk2/prediction_v11_v2 \
    --invalid_video_fn /home/zfchen/code/clevrer_dataset_generation_v2/models/CLEVRER/executor_clevrer/data/invalid_video_v11.txt \