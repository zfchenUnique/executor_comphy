python run_oe_clevrer.py \
    --gt_flag 0 \
    --use_event_ann 0 \
    --program_path /home/zfchen/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v16_3/open_end_questions.json \
    --question_path /home/zfchen/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v16_3/open_end_questions.json \
    --raw_motion_prediction_dir /home/zfchen/code/output/render_output_vislab3/v16/predictions_prp \
    --ann_dir /home/zfchen/code/output/render_output_vislab3/v16/prediction_prp_mass_charge \
    --invalid_video_fn data/invalid_video_v16.txt \
    --start_id 4000 \
    --num_sim 100 \

