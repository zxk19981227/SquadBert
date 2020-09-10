
class Feature():
    def __init__(self,input,start,end,segment,mask,tokens,span_index,span_id,token_to_org,is_max_context,unique_id,context,answer,question):
        self.input=input
        self.start=start
        self.end=end
        self.segment=segment
        self.mask=mask
        self.token=tokens
        self.span_index=span_index
        self.sample_index=span_id
        self.token_to_orgs=token_to_org
        self.is_max_context=is_max_context
        self.id=unique_id
        self.context=context
        self.answer=answer
        self.question=question