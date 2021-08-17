"""
Run symbolic reasoning on open-ended questions
"""
import os
import json
from tqdm import tqdm
import argparse

from executor import Executor
from simulation import Simulation
import pdb
from utils.utils import print_monitor
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('--use_event_ann', default=1, type=int)
parser.add_argument('--use_in', default=0, type=int)  # Interaction network
parser.add_argument('--program_path', default='/home/zfchen/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v13_2/open_end_questions.json')
parser.add_argument('--question_path', default='/home/zfchen/code/clevrer_dataset_generation_v2/clevrer_question_generation/output/questions_v13_2/open_end_questions.json')
parser.add_argument('--gt_flag', default=0, type=int)
parser.add_argument('--ann_dir', default='/home/zfchen/code/output/render_output/causal_v13')
parser.add_argument('--track_dir', default='/home/zfchen/code/output/render_output/causal_v13_coco_ann')
parser.add_argument('--frame_diff', default=5, type=int)
parser.add_argument('--mc_flag', default=0, type=int)
parser.add_argument('--raw_motion_prediction_dir', default='')
parser.add_argument('--invalid_video_fn', default = 'invalid_video_v14.txt')
parser.add_argument('--start_id', default=0, type=int)
parser.add_argument('--num_sim', default=5000, type=int)
parser.add_argument('--ann_offset', default=0, type=int)
parser.add_argument('--gt_ann_dir', default='/home/zfchen/code/output/render_output_vislab3/v16/render')
parser.add_argument('--ambi_ques_list_fn', default = './data/ambi_ques_v16.txt')
parser.add_argument('--gen_ambi_flag', default=0, type=int)
args = parser.parse_args()

question_path = args.question_path
program_path = args.program_path 

with open(program_path) as f:
    parsed_pgs = json.load(f)
with open(question_path) as f:
    anns = json.load(f)

if args.gen_ambi_flag:
    fh = open(args.ambi_ques_list_fn, 'w')

total, correct = 0, 0

pbar = tqdm(range(args.num_sim))

acc_monitor = {}
acc_monitor_2 = {}
ans_swap = ''

if os.path.isfile(args.invalid_video_fn):
    print('Excluding invalid videos from %s\n'%(args.invalid_video_fn))
    invalid_vid_mat = np.loadtxt(args.invalid_video_fn).astype(np.int)
    invalid_list = invalid_vid_mat.tolist() 
    print(invalid_list)
else:
    invalid_list = []
    fh_invalid = open(args.invalid_video_fn, 'w')

for ann_idx in pbar:
    file_idx = ann_idx  + args.start_id
    ques_idx = file_idx - args.ann_offset
    question_scene = anns[ques_idx]
    sim = Simulation(args, file_idx, use_event_ann=(args.use_event_ann != 0))
    if len(sim.objs)!=len(sim.get_visible_objs()) and len(invalid_list)==0:
        fh_invalid.write('%d\n'%(file_idx))
        print(file_idx)
        continue
    if file_idx in invalid_list:
        continue
    exe = Executor(sim)
    for q_idx, q in enumerate(question_scene['questions']):
        question = q['question']
        parsed_pg = parsed_pgs[ques_idx]['questions'][q_idx]['program']
        #if 'filter_heavy' not in parsed_pg:
        #if 'filter_light' not in parsed_pg:
        #    continue
        q_type = parsed_pg[-1]
        if q_type+'_acc' not in acc_monitor:
            acc_monitor[q_type+'_acc'] = 0
            acc_monitor[q_type+'_total'] = 1
        else:
            acc_monitor[q_type+'_total'] +=1

        q_fam = q['question_family']
        #if q_fam!='event_query':
        #if q_fam!='object_query':
        #    continue
        if q_fam+'_acc' not in acc_monitor_2:
            acc_monitor_2[q_fam+'_acc'] = 0
            acc_monitor_2[q_fam+'_total'] = 1
        else:
            acc_monitor_2[q_fam+'_total'] +=1

        pred = exe.run(parsed_pg, debug=False)
        ans = q['answer']
        if pred == ans:
            correct += 1
            acc_monitor[q_type+'_acc'] +=1
            acc_monitor_2[q_fam+'_acc'] +=1
        elif 'and' in ans: # for query_both     
            eles = ans.split(' ')
            if len(eles)==3:
                ans_swap  = eles[2] + ' and ' + eles[0]
                if pred == ans_swap:
                    correct +=1
                    acc_monitor[q_type+'_acc'] +=1
                    acc_monitor_2[q_fam+'_acc'] +=1
        
        if pred!=ans and pred!=ans_swap:
            #debug_flag = True
            debug_flag = False
            if debug_flag:
                pred = exe.run(parsed_pg, debug=True)
                print('%d %d\n'%(file_idx, q_idx))
                print(question)
                print(parsed_pg)
                print('pred: %s\n'%(pred))
                print('ans: %s\n'%(ans))
                pdb.set_trace()
            else:
                pass
            if args.gen_ambi_flag:
                fh.write('%d %d %d %d\n'%(ann_idx, file_idx, ques_idx, q_idx))
                print('%d %d %d %d\n'%(ann_idx, file_idx, ques_idx, q_idx))
                fh.flush()
        total += 1

    pbar.set_description('acc: {:f}%%'.format(float(correct)*100/total))
print_monitor(acc_monitor)
print_monitor(acc_monitor_2)
print('overall accuracy per question: %f %%' % (float(correct) * 100.0 / total))
