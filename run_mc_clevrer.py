"""
Run symbolic reasoning on multiple-choice questions
"""
import os
import json
from tqdm import tqdm
import argparse

from executor import Executor
from simulation import Simulation
import pdb
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--n_progs', required=True)
parser.add_argument('--use_event_ann', default=1, type=int)
parser.add_argument('--use_in', default=0, type=int)  # Use interaction network
parser.add_argument('--program_path', default='/home/zfchen/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v13_2/open_end_questions.json')
parser.add_argument('--question_path', default='/home/zfchen/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v13_2/open_end_questions.json')
parser.add_argument('--gt_flag', default=0, type=int)
parser.add_argument('--ann_dir', default='/home/zfchen/code/output/render_output/causal_v13')
parser.add_argument('--track_dir', default='/home/zfchen/code/output/render_output/causal_v13_coco_ann')
parser.add_argument('--raw_motion_prediction_dir', default='')
parser.add_argument('--frame_diff', default=5, type=int)
parser.add_argument('--mc_flag', default=1, type=int)
parser.add_argument('--invalid_video_fn', default = 'invalid_video_v14.txt')
parser.add_argument('--start_id', default=0, type=int)
parser.add_argument('--num_sim', default=5000, type=int)
args = parser.parse_args()

question_path = args.question_path
program_path = args.program_path 

print(question_path)
print(program_path)

with open(program_path) as f:
    parsed_pgs = json.load(f)
with open(question_path) as f:
    anns = json.load(f)

total, correct = 0, 0
total_per_q, correct_per_q = 0, 0
total_expl, correct_expl = 0, 0
total_expl_per_q, correct_expl_per_q = 0, 0
total_pred, correct_pred = 0, 0
total_pred_per_q, correct_pred_per_q = 0, 0
total_coun, correct_coun = 0, 0
total_coun_per_q, correct_coun_per_q = 0, 0

pred_map = {'yes': 'correct', 'no': 'wrong', 'error': 'error'}
pbar = tqdm(range(args.num_sim))
if os.path.isfile(args.invalid_video_fn):
    print('Excluding invalid videos from %s\n'%(args.invalid_video_fn))
    invalid_vid_mat = np.loadtxt(args.invalid_video_fn).astype(np.int)
    invalid_list = invalid_vid_mat.tolist() 
    print(invalid_list)
else:
    invalid_list = []

for ann_idx in pbar:
    file_idx = ann_idx + args.start_id 
    question_scene = anns[file_idx]
    sim = Simulation(args, file_idx, use_event_ann=(args.use_event_ann != 0))
    if file_idx in invalid_list:
        continue
    exe = Executor(sim)
    valid_q_idx = 0
    for q_idx, q in enumerate(question_scene['questions']):
        question = q['question']
        q_type = q['question_type']
        if q_type == 'descriptive': # skip open-ended questions
            continue
        #print('%d %d\n'%(file_idx, valid_q_idx))
        #print(question)
        q_ann = parsed_pgs[file_idx]['questions'][q_idx]
        correct_question = True
        if 'choices' in q_ann:
            for c in q_ann['choices']:
                full_pg = c['program'] + q_ann['question_program']
                ans = c['answer']
                pred = exe.run(full_pg, debug=False)
                pred = pred_map[pred]
                # print(ans, pred)
                if ans == pred:
                    correct += 1
                else:
                    correct_question = False
                total += 1
                
                if q['question_type'].startswith('predictive'):
                    # print(pred, ans)
                    if ans == pred:
                        correct_pred += 1
                    total_pred += 1

                if q['question_type'].startswith('counterfactual'):
                    if ans == pred:
                        correct_coun += 1
                    total_coun += 1
        else:
            for choice_type in ['correct', 'wrong']:
                for c in q_ann[choice_type]:
                    full_pg = c[1] + q_ann['program']
                    #ans = 'yes' if choice_type=='correct' else 'no'
                    ans = choice_type 
                    pred = exe.run(full_pg, debug=False)
                    pred = pred_map[pred]
                    # print(ans, pred)
                    if ans == pred:
                        correct += 1
                    else:
                        correct_question = False
                        debug_flag=False
                        if debug_flag:
                            pred = exe.run(full_pg, debug=True)
                            print('%d %d\n'%(file_idx, valid_q_idx))
                            print(question)
                            print(c[0])
                            print('pred: %s\n'%(pred))
                            print('ans: %s\n'%(ans))
                            pdb.set_trace()
                    total += 1
                    
                    if q['question_type'].startswith('predictive'):
                        # print(pred, ans)
                        if ans == pred:
                            correct_pred += 1
                        total_pred += 1

                    if q['question_type'].startswith('counterfactual'):
                        if ans == pred:
                            correct_coun += 1
                        total_coun += 1

        if correct_question:
            correct_per_q += 1
        total_per_q += 1

        if q['question_type'].startswith('explanatory'):
            if correct_question:
                correct_expl_per_q += 1
            total_expl_per_q += 1

        if q['question_type'].startswith('predictive'):
            if correct_question:
                correct_pred_per_q += 1
            total_pred_per_q += 1

        if q['question_type'].startswith('counterfactual'):
            if correct_question:
                correct_coun_per_q += 1
            total_coun_per_q += 1
        valid_q_idx += 1
    # print('up to scene %d: %d / %d correct options, accuracy %f %%'
    #       % (ann_idx, correct, total, (float(correct)*100/total)))
    # print('up to scene %d: %d / %d correct questions, accuracy %f %%'
    #       % (ann_idx, correct_per_q, total_per_q, (float(correct_per_q)*100/total_per_q)))
    # print()
    pbar.set_description('per choice {:f}, per questions {:f}'.format(float(correct)*100/total, float(correct_per_q)*100/total_per_q))

print('============ results ============')
print('overall accuracy per option: %f %%' % (float(correct) * 100.0 / total))
print('overall accuracy per question: %f %%' % (float(correct_per_q) * 100.0 / total_per_q))
print('predictive accuracy per option: %f %%' % (float(correct_pred) * 100.0 / total_pred))
print('predictive accuracy per question: %f %%' % (float(correct_pred_per_q) * 100.0 / total_pred_per_q))
print('counterfactual accuracy per option: %f %%' % (float(correct_coun) * 100.0 / total_coun))
print('counterfactual accuracy per question: %f %%' % (float(correct_coun_per_q) * 100.0 / total_coun_per_q))
#print('Number of invalid videos %d\n'%(invalid))
print('============ results ============')

output_ann = {
    'total_options': total,
    'correct_options': correct,
    'total_questions': total_per_q,
    'correct_questions': correct_per_q,
    'total_explanatory_options': total_expl,
    'correct_explanatory_options': correct_expl,
    'total_explanatory_questions': total_expl_per_q,
    'correct_explanatory_questions': correct_expl_per_q,
    'total_predictive_options': total_pred,
    'correct_predictive_options': correct_pred,
    'total_predictive_questions': total_pred_per_q,
    'correct_predictive_questions': correct_pred_per_q,
    'total_counterfactual_options': total_coun,
    'correct_counterfactual_options': correct_coun,
    'total_counterfactual_questions': total_coun_per_q,
    'correct_counterfactual_questions': correct_coun_per_q,
}

output_file = 'result_mc.json'
if args.use_in != 0:
    output_file = 'result_mc_in.json'
with open(output_file, 'w') as fout:
    json.dump(output_ann, fout)
