import json
import language_evaluation_2.language_evaluation
import json

evaluator = language_evaluation_2.language_evaluation.CocoEvaluator(verbose=True)

def cal_metric(result_file):
    result_list = json.load(open(result_file, "r"))
    predicts = []
    answers = []
    used_img_id = []
    for each in result_list:
        if each['question_id'] in used_img_id:
            continue
        else:
            used_img_id.append(each['question_id'])
            predicts.append(each["pred_caption"])
            answers.append(each["gold_caption"])
        
    print('len(predicts) in cal_metric: ', len(predicts))
    results, scores = evaluator.run_evaluation(predicts, answers)
    print('='*100)
    print (len(result_list), results, scores)
    print('='*100)
    return results, predicts, answers, used_img_id

def cal_metric_single_data(result_list, All_results, instance_id):
    predicts = []
    answers = []
    predicts.append(result_list[instance_id]["pred_caption"])
    answers.append(result_list[instance_id]["gold_caption"])
    if instance_id <= 3200-1:
        img = './construction_dataset/ACID/trainval/'+result_list[instance_id]['question_id']
    elif instance_id <= 3400-1:
        img = './construction_dataset/SAFE???/trainval/'+result_list[instance_id]['question_id']
    elif instance_id <= 3581-1:
        img = './construction_dataset/SAFE???/trainval/'+result_list[instance_id]['question_id']
#     elif 
    image = cv2.imread(img)
    plt.imshow(image)
    plt.show()
    print('-'*77)
    print('pred_caption: ', result_list[instance_id]["pred_caption"])
    print('-'*77)
    for n, item in enumerate(result_list[instance_id]["gold_caption"]):
        print(f'gold_caption {n}: ', item)
    print('-'*77)
    for key in All_results.keys():
        print(f'{key}: ', All_results[key][instance_id])


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


def get_config():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--source_prediction_train",
                        required=True, 
                        type=str,
                        help="The fine-tuned model's predictions to create HP labels.")
    
    parser.add_argument("--source_prediction_val",
                        required=True, 
                        type=str,
                        help="The fine-tuned model's predictions to create HP labels.")
    
    return parser.parse_args()
    
    
if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3,4,5,6,7'
    args = get_config()
    
    train_result_list = json.load(open(args.source_prediction_train, "r"))
    test_result_list = json.load(open(args.source_prediction_val, "r"))
    
    files = [args.source_prediction_train, args.source_prediction_val]
    for j, file_name in enumerate(files):
        results, predicted_captions, gold_captions, question_ids = cal_metric(file_name)
    
        All_results = {'Bleu_1': [],
                   'Bleu_2': [],
                   'Bleu_3': [],
                   'Bleu_4': [],
                   'ROUGE_L': [],
                   'CIDEr': []
                  }

        for i in range(len(results.imgToEval)):
            All_results['Bleu_1'].append(results.imgToEval[i]['Bleu_1'])
            All_results['Bleu_2'].append(results.imgToEval[i]['Bleu_2'])
            All_results['Bleu_3'].append(results.imgToEval[i]['Bleu_3'])
            All_results['Bleu_4'].append(results.imgToEval[i]['Bleu_4'])
            All_results['ROUGE_L'].append(results.imgToEval[i]['ROUGE_L'])
            All_results['CIDEr'].append(results.imgToEval[i]['CIDEr'])
            
        cider = All_results['CIDEr'].copy()
        
        cider_sorted = [i for i, x in sorted(enumerate(cider), key=lambda x: x[1])]
        
        HP_label_train = []
        for i in range(len(cider_sorted)):
            if i <= len(cider_sorted)/4:
                HP_label_train.append({'img_path': question_ids[cider_sorted[i]], 'HP_label': 0})
            elif i <= 2*len(cider_sorted)/4:
                HP_label_train.append({'img_path': question_ids[cider_sorted[i]], 'HP_label': 1})
            elif i <= 3*len(cider_sorted)/4:
                HP_label_train.append({'img_path': question_ids[cider_sorted[i]], 'HP_label': 2})
            else:
                HP_label_train.append({'img_path': question_ids[cider_sorted[i]], 'HP_label': 3})
                
        if j == 0:
            with open(f'./construction_dataset/HP_label_train.json', 'w') as f:
                json.dump(HP_label_train, f)
        else:
            with open(f'./construction_dataset/HP_label_val.json', 'w') as f:
                json.dump(HP_label_train, f)
