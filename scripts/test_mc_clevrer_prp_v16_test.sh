python run_mc_clevrer.py \
    --gt_flag 0 \
    --use_event_ann 0 \
    --n_progs 1000 \
    --raw_motion_prediction_dir /home/zfchen/code/output/render_output_vislab3/v16_test/predictions_motion_prp_no_ref_v2_vis50_new \
    --ann_dir /home/zfchen/code/output/render_output_vislab3/v16_test/prediction_prp_mass_charge \
    --invalid_video_fn 'data/invalid_video_v16_1_test.txt' \
    --start_id 10000 \
    --num_sim 2000 \
    --program_path /home/zfchen/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v16_3_2_test_v2_1/multiple_choice_questions.json \
    --question_path /home/zfchen/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v16_3_2_test_v2_1/multiple_choice_questions.json \
    #--program_path /home/zfchen/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v16_3_1_test/multiple_choice_questions.json \
    #--question_path /home/zfchen/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v16_3_1_test/multiple_choice_questions.json \