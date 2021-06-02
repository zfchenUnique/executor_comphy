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

for ann_idx in pbar:
    file_idx = ann_idx  + 3000 
    question_scene = anns[file_idx]
    sim = Simulation(args, file_idx, use_event_ann=(args.use_event_ann != 0))
    if len(sim.objs)!=len(sim.get_visible_objs()):
        print('Invalid annotation, sim %d\n'%(file_idx))
        continue
    exe = Executor(sim)
    for q_idx, q in enumerate(question_scene['questions']):
        #print('%d %d\n'%(file_idx, q_idx))
        question = q['question']
        parsed_pg = parsed_pgs[file_idx]['questions'][q_idx]['program']
        pred = exe.run(parsed_pg, debug=False)
        ans = q['answer']
        if pred == ans:
            correct += 1
        elif 'and' in ans: # for query_both     
            eles = ans.split(' ')
            if len(eles)==3:
                ans_swap  = eles[2] + ' and ' + eles[0]
                if pred == ans_swap:
                    correct +=1
        else:
            #print(question)
            #print(parsed_pg)
            #pred = exe.run(parsed_pg, debug=True)
            #pdb.set_trace()
            pass
        total += 1

    pbar.set_description('acc: {:f}%%'.format(float(correct)*100/total))
pdb.set_trace()
print('overall accuracy per question: %f %%' % (float(correct) * 100.0 / total))
