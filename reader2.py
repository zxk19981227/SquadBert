import json
import torch
import logging
from transformers import BertTokenizer
import collections
import os
from Feature import Feature
class Squad_reader(torch.utils.data.Dataset):
    def is_whitespace(self,c):
        if c == " " or c == "\t" or c == "\r" or c == "\n" or ord(c) == 0x202F:
            return True
        return False
    def check_is_max_context(self,spans,position,cur_index):#这个函数的主要作用我感觉原文也没有讲清楚，实质上是为了保存最大的上下文信息，
        #从而选择左右单词数量最大的span进行预测
        best_score=None
        best_index=None
        for (index,span) in enumerate(spans):
            end=span.start+span.length-1
            if position>end or position<span.start:
                continue#z这里代表这个单词并不存在于这里
            score=min(end-position,position-span.start)+0.01*span.length
            if best_index is None or score > best_score:#这里运用到的一个点就是or的特性
                best_index=index
                best_score=score
        return best_index==cur_index

    def _improve_span(self,all_tokens,tok_start,tok_end,orig_text):
        """
        这里实质上是一个逐个匹配过程来生产最小、匹配答案的位置,实质上就是分段后在匹配机制
        :param all_tokens:
        :param tok_start:
        :param tok_end:
        :param token:
        :param orig_text:
        :return:
        """
        # print(orig_text)
        tokens=" ".join(self.token.tokenize(orig_text))
        for i in range(tok_start,tok_end+1):
            for j in range(tok_end,tok_start-1,-1):
                if " ".join(all_tokens[i:j+1])==tokens:
                    return (i,j)
        return tok_start,tok_end
    def __init__(self,path,max_seq_len=384,max_query_length=64,strids=128):
        """
        这里我在开始context内容的char到word映射部分借鉴了transformers内容中的对应部分
        :param path:
        :param max_seq_len:
        :param max_query_length:
        """
        logger=logging.getLogger(__name__)
        if os.path.exists(path+"/features"):
            logger.info("load data from feature")
            tmp=torch.load(path+"/features")
            self.Features=tmp["Features"]
        else:
            self.Features=[]#这里是我为了方便写下predict结果而使用的一个组织
            self.token=BertTokenizer.from_pretrained("bert-base-uncased")
            with open(path,'r') as f:
                data=json.load(f)
            data=data['data']
            qu=0
            for dic in data:
                for compare in dic['paragraphs']:
                    paragraph_context=compare['context']
                    doc_tokens = []
                    char_to_word_offset = []
                    prev_is_whitespace = True
                    for c in paragraph_context:
                        if self.is_whitespace(c):
                            prev_is_whitespace = True
                        else:
                            if prev_is_whitespace:
                                doc_tokens.append(c)
                            else:
                                doc_tokens[-1] += c
                            prev_is_whitespace = False
                        char_to_word_offset.append(len(doc_tokens) - 1)
                    for question in compare['qas']:
                        question_id=question["id"]
                        question_text=question["question"]
                        org_text=None
                        if question['is_impossible']:
                            continue#这里主要是对于无答案直接忽略
                        else:
                            answer=question["answers"][0]
                            save=question["answers"]
                            org_text=answer['text']
                            start_offset=answer['answer_start']
                            answer_length=len(org_text)
                            start_position=char_to_word_offset[start_offset]
                            end_position=char_to_word_offset[start_offset+answer_length-1]
                            actual_answer=" ".join(doc_tokens[start_position:end_position+1])
                            clean_answer=" ".join((org_text).strip().split())
                            if actual_answer.find(clean_answer)==-1:
                                logging.warning("Could not find answer: '%s' vs. '%s'",
                                                   actual_answer,clean_answer)
                                continue
                        tok_to_orig_index=[]
                        orig_to_tok_index=[]
                        all_tokens=[]
                        query_token=self.token.tokenize(question_text)
                        if len(query_token)>max_query_length:
                            query_token=query_token[:max_query_length]
                        for (index,token) in enumerate(doc_tokens):#这里加入index和token主要目的是为了方便构建新的转换
                            #主要是token.tokenize可以将单词继续微分。在该函数介绍中用的例子unwanted 如果不在单词表中，会被分解为un ##want ##ed三个词
                            orig_to_tok_index.append(len(all_tokens))
                            sub_tokens=self.token.tokenize(token)
                            for sub in sub_tokens:
                                tok_to_orig_index.append(index)
                                all_tokens.append(sub)
                        tok_start=orig_to_tok_index[start_position]
                        #这里主要是tok_start和tok_end是分解完的开始结束，improve span里面也介绍了比如japanese但是答案中是japan如何解决
                        #这里为什么不对token进行合并，因为接下来需要将context进行按照strides大小的分段
                        #源代码一点都没有讲解真的看起来太麻烦了，论文里面也没有对于处理方式说任何东西
                        if end_position<len(doc_tokens)-1:#这里比较迷茫的地方在于为什么这里需要加入一个大于length的判断？因为断电位置不对？
                            tok_end=orig_to_tok_index[end_position+1]-1#这里的处理主要是单词分散开了，得从下一个单词往前推一个才是末位置
                        else:
                            tok_end=len(all_tokens)-1#这里我理解的是因为尾部已经超出范围，那么匹配范围只能扩展到最大
                        tok_star,tok_end=self._improve_span(all_tokens,tok_start,tok_end,org_text)
                        max_context_length=max_seq_len-max_query_length-3
                        Span=collections.namedtuple("span",["start","length"])
                        span_list=[]
                        start=0
                        while start < len(all_tokens):#这里设置的非常巧妙
                            length=len(all_tokens)-start
                            if length >max_context_length:
                                length=max_context_length
                            span_list.append(Span(start=start,length=length))
                            if start+length==len(all_tokens):#这里代表已经搜索到最后了，真的非常巧妙，也就是说剩下的小于max_context_length
                                break
                            start+=min(strids,length)
                        for (index,span) in enumerate(span_list):#这里开始针对于每一个span生成一个list，我觉得真的比较繁琐
                            span_tokens=["CLS"]#用来记录span中token以及token的解析方式
                            token_to_org_map={}#这里是用来完成对于token在源数据中位置的映射
                            is_max_context={}
                            segment_ids=[0]

                            for token in query_token:
                                span_tokens.append(token)
                                segment_ids.append(0)
                            span_tokens.append("[SEP]")
                            segment_ids.append(0)
                            for i in range(span.length):
                                current_index=span.start+i
                                token_to_org_map[len(span_tokens)]=current_index#z这个就是现在在span中位置到原目标中位置的影射
                                is_max_context[len(span_tokens)]=self.check_is_max_context(span_list,current_index,index)
                                span_tokens.append(all_tokens[current_index])
                                segment_ids.append(1)
                            segment_ids.append(1)
                            span_tokens.append("[SEP]")
                            inputs=self.token.convert_tokens_to_ids(span_tokens)
                            input_masks=[1]*len(inputs)
                            while len(inputs)<max_seq_len:
                                input_masks.append(0)
                                segment_ids.append(0)
                                inputs.append(0)
                            assert len(inputs) == max_seq_len
                            assert len(input_masks) == max_seq_len
                            assert len(segment_ids) == max_seq_len
                            span_start=span.start
                            span_end=span.start+span.length-1
                            if tok_start>span_end or tok_end<span_start:
                                start_position=0
                                end_position=0#这里就是预测全为0，也就是为query第一个"[cls]"
                            else:
                                offset=len(query_token)+2#这里我忘记了最开始写的时候，后来才发现犯了这个错误
                                start_position=tok_start-span_start+offset
                                end_position=tok_end-span_start+offset
                            feature=Feature(input=inputs,start=start_position,end=end_position,segment=segment_ids,mask=input_masks,
                                            span_index=qu,tokens=span_tokens,span_id=index,token_to_org=token_to_org_map,is_max_context=is_max_context,unique_id=question_id,context=all_tokens,answer=save,question=question_text)

                            self.Features.append(feature)
                            qu+=1
            # if path=="./data/train_v2.0.json":
            #     logger.info("saving features to cached file %s","./data/train_v2.0/feature")
            #     torch.save({"Features":self.Features},"./data/train-v2.0/feature")
            # else:
            #     logger.info("saving features to cached file %s","./data/dev_v2.0/feature")
            #     torch.save({"Features":self.Features},"./data/dev-v2.0/feature")
    def __len__(self):
        return len(self.Features)
    def __getitem__(self, item):
        return torch.tensor(self.Features[item].span_index),torch.tensor(self.Features[item].input),torch.tensor(self.Features[item].segment),torch.tensor(self.Features[item].mask),torch.tensor(self.Features[item].start),torch.tensor(self.Features[item].end)
    def get_answer(self,item):
        return self.Features[item]
# rader=Squad_reader("./data/test.json")
# print(rader.__len__())