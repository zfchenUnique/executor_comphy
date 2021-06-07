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
args = parser.parse_args()

question_path = args.question_path
program_path = args.program_path 

with open(program_path) as f:
    parsed_pgs = json.load(f)
with open(question_path) as f:
    anns = json.load(f)

total, correct = 0, 0

pbar = tqdm(range(1000))

acc_monitor = {}
ans_swap = ''

for ann_idx in pbar:
    file_idx = ann_idx  + 3000
    question_scene = anns[file_idx]
    sim = Simulation(args, file_idx, use_event_ann=(args.use_event_ann != 0))
    if len(sim.objs)!=len(sim.get_visible_objs()):
        #print('Invalid annotation, sim %d\n'%(file_idx))
        continue
    exe = Executor(sim)
    for q_idx, q in enumerate(question_scene['questions']):
        question = q['question']
        parsed_pg = parsed_pgs[file_idx]['questions'][q_idx]['program']
        #if 'filter_moving' not in parsed_pg and 'filter_stationary' not in parsed_pg:
        #if 'query_shape' not in parsed_pg:
        #    continue
        q_type = parsed_pg[-1]
        if q_type+'_acc' not in acc_monitor:
            acc_monitor[q_type+'_acc'] = 0
            acc_monitor[q_type+'_total'] = 1
        else:
            acc_monitor[q_type+'_total'] +=1

        pred = exe.run(parsed_pg, debug=False)
        ans = q['answer']
        if pred == ans:
            correct += 1
            acc_monitor[q_type+'_acc'] +=1
        elif 'and' in ans: # for query_both     
            eles = ans.split(' ')
            if len(eles)==3:
                ans_swap  = eles[2] + ' and ' + eles[0]
                if pred == ans_swap:
                    correct +=1
                    acc_monitor[q_type+'_acc'] +=1
        
        if pred!=ans and pred!=ans_swap:
            debug_flag=False
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
        total += 1

    #pbar.set_description('acc: {:f}%%'.format(float(correct)*100/total))
print_monitor(acc_monitor)
print('overall accuracy per question: %f %%' % (float(correct) * 100.0 / total))
