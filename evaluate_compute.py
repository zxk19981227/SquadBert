import torch
from utils import get_nbest,predictions,compute_f1,get_final_text,_compute_softmax,get_raw_scores
import collections
from tqdm import tqdm
class Evaluate():
    def __init__(self):
        self.loss=[]
        self.f1=[]
        self.extract=[]
    def __call__(self,features,start_logits,end_logits,token,f):#这里需要把问题id相同的语句合并起来计算
        problem_features={}
        start_problem_logit={}
        end_problem_logit={}
        for num,(feature,start_logit,end_logit) in enumerate(zip(features,start_logits,end_logits)):
            if feature.id not in problem_features.keys():
                problem_features[feature.id]=[feature]
                start_problem_logit[feature.id]=[start_logit]
                end_problem_logit[feature.id]=[end_logit]
            else:
                problem_features[feature.id].append(feature)
                start_problem_logit[feature.id].append(start_logit)
                end_problem_logit[feature.id].append(end_logit)
        for key in tqdm(problem_features.keys()):
            features=problem_features[key]
            start_logits=start_problem_logit[key]
            end_logits=end_problem_logit[key]
            answer_text=features[0].answer
            context=features[0].context
            score_null=100000
            prelim_predictions=[]#这里根据原文注释，其实是追踪start+end最小值
            min_null_score_index=0#这里用来计算null最小的分段位置
            null_start_logit=0
            null_end_logit=0#这里的作用其实是记录最小的start和end位置
            for index,(feature,start_logit,end_logit) in enumerate(list(zip(features,start_logits,end_logits))):
                start_index=get_nbest(start_logit,20)
                end_index=get_nbest(end_logit,20)
                for start in start_index:
                    for end in end_index:
                        if start>=len(feature.token):
                            continue
                        if end>=len(feature.token):
                            continue
                        if start not in feature.token_to_orgs:#这两个位置讲解的其实是无法从现有位置映射到原有位置
                            continue
                        if end not in feature.token_to_orgs:
                            continue
                        if not feature.is_max_context.get(start,False):#这里是检查搜索最小min，的时候是否又出现。这里源代码真的非常谨慎了
                            continue
                        if start>end:
                            continue
                        prelim_predictions.append([index,start,end,start_logit[start],end_logit[end]])#这里其实是把这个句子中所有信息汇聚起来为接下来计算做准备
                prelim_predictions=sorted(prelim_predictions,key=lambda x:(x[3]+x[4]),reverse=True)
            seen_predictions={}
            n_best=[]
            for each in prelim_predictions:
                if len(n_best)>=20:
                    break
                if each[1]>0:
                    feature=features[each[0]]
                    org_doc_start=feature.token_to_orgs[each[1]]
                    orig_doc_end=feature.token_to_orgs[each[2]]
                    tok_text=feature.token[each[1]:each[2]+1]
                    orig_tokens=context[org_doc_start:orig_doc_end+1]
                    tok_text=token.convert_tokens_to_string(tok_text)
                    orig_text=token.convert_tokens_to_string(orig_tokens)
                    tok_text=" ".join(tok_text.lower().strip().split())
                    orig_text=" ".join(orig_text.lower().strip().split())
                    final_text=get_final_text(tok_text,orig_text,True,False,token)#这个函数我还没来得及看，先用原本的用着
                    if final_text in seen_predictions:
                        continue
                    seen_predictions[final_text]=True#这里我的理解是用来解决重复出现
                else:
                    final_text=""
                    seen_predictions[""]=True
                n_best.append([final_text,each[3],each[4]])
            if not n_best:
                n_best.append(["empty",0.0,0.0])
            total_score=[]
            best_non_null_entry=None
            for entry in n_best:
                total_score.append(entry[1]+entry[2])
                if not best_non_null_entry and entry[0]:
                    best_non_null_entry=entry
            probs=_compute_softmax(total_score)
            nbest_json = []
            predictions=[]
            f1=0
            excat=0
            predictions=n_best[0][0]
            f1,excat=get_raw_scores(answer_text,predictions,token)
            f.write("answer:"+str(answer_text)+"\n")
            f.write("predictions:"+str(predictions)+"\n")
            f.write("f1:"+str(f1)+"\n")


            f.write("extract:"+str(excat)+"\n")

            self.f1.append(f1)
            self.extract.append(excat)







