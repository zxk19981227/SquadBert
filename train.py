import torch
from numpy import mean
from transformers import BertForQuestionAnswering
from torchsummary import summary
from evaluate_compute import Evaluate
from model import QA
from utils import get_nbest,predictions,compute_f1
import os
from tqdm import tqdm
from reader2 import Squad_reader
os.environ["CUDA_VISIBLE_DEVICES"] = "2"
# reader=Squad_reader("./data/train-v2.0.json")
test_reader=Squad_reader("./data/dev-v2.0.json")
learning_epoch=4

n_best_size=20
model=QA()
output_dir="./modelbin"

model=model.cuda()
# model.load_state_dict(torch.load("./modelbin/model.bin"))
# optimizer=torch.optim.Adam(model.parameters(),3e-5)
# # # summary(model,[(1,350),(1,350),(1,350)])
# for i in range(learning_epoch):
#     loss_com=[]
#     cor1=0
#     cor2=0
#     to=0
#     data_set=torch.utils.data.DataLoader(dataset=reader,batch_size=16,shuffle=True)
#     model.train()
#     print("training epoch ", i)
#     for num,(index,data,seg,mask,start,end) in tqdm(enumerate(data_set)):
#         data=(data).cuda()
#         data=data.long()
#         seg=(seg).cuda()
#         mask=(mask).cuda()
#         start=(start).cuda()
#         end=(end).cuda()
#         out_put=model(input_ids=data,attention_mask=mask,token_type_ids=seg,start_positions=start,end_positions=end)
#         loss, start_pre, end_pre=out_put[:4]
#        # print(test)
#         start_pre=torch.argmax(start_pre,-1)
#         end_pre=torch.argmax(end_pre,-1)
#         loss.backward()
#         loss_com.append(loss.item())
#         optimizer.step()
#         optimizer.zero_grad()
#         cor1+=(start_pre==start).cpu().sum()
#         cor2+=(end_pre==end).cpu().sum()
#         to+=start_pre.shape[0]
#         # print("correct",cor)
#         # print("total",to)
#     print("start predict accuracy", float(cor1)/float(to))
#     print("end predict accuracy", float(cor2)/float(to))
#     print("train loss ", mean(loss_com))
# torch.save(model.state_dict(), './modelbin/3e2model.bin')
model.load_state_dict(torch.load("./modelbin/3e2model.bin"))
print("load_finished")
print("test_len",test_reader.__len__())
with torch.no_grad():
    with open("./predict.txt",'w') as f:
        cor1=0
        cor2=0
        model.eval()
        loss_com=[]
        f1_score=[]
        eval=Evaluate()
        print("test epoch")
        data_set2=torch.utils.data.DataLoader(dataset=test_reader,batch_size=12,shuffle=False)
        number=None
        features=[]
        start_logits=[]
        end_logits=[]
        indexs=[]
        for num, (index,  data, seg, mask, start, end) in tqdm(enumerate(data_set2)):
            data = data.long()
            data = data.cuda()
            seg = seg.cuda()
            mask = mask.cuda()
            start = start.cuda()
            end = end.cuda()
            out_put = model(input_ids=data, attention_mask=mask, token_type_ids=seg, start_positions=start,
                            end_positions=end)
            loss, start_pre, end_pre = out_put[:4]

            loss_com.append(loss.item())
            start_logits.extend(list(start_pre.cpu().numpy()))
            end_logits.extend(list(end_pre.cpu().numpy()))
            for each in index:
                indexs.append(test_reader.get_answer(each))
        eval(indexs,start_logits,end_logits,test_reader.token,f)
        print("f1 4epoch-5e",mean(eval.f1))
        print("extract 4epoch-5e",mean(eval.extract))


