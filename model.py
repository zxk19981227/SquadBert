from transformers import BertModel,BertPreTrainedModel
import torch
import torch.nn as nn
class QA(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.num_labels = 2
        self.dropout=torch.nn.Dropout(0.1)
        self.bert = BertModel.from_pretrained("./tmp",output_hidden_states=True)
        self.qa_outputs = nn.Linear(768, 2)
        # self.qa_ou=nn.Linear(384,2)

        # self.init_weights()
    # def predict(self,
    #     input_ids=None,
    #     attention_mask=None,
    #     token_type_ids=None,
    #     position_ids=None,
    #     start_positions=None,
    #     end_positions=None):
    #     outputs = self.bert(
    #         input_ids,
    #         attention_mask=attention_mask,
    #         token_type_ids=token_type_ids,
    #         position_ids=position_ids,
    #     )
    #
    #     sequence_output = outputs[0]
    #     batch_size=sequence_output.shape[0]
    #     sequence_output=self.dropout(sequence_output)
    #     logits = self.qa_outputs(sequence_output)
    #     sequence_output=torch.nn.functional.gelu(logits)
    #     logits=self.qa_ou(sequence_output)
    #     start_logits, end_logits = logits.split(1, dim=-1)
    #     start_logits = start_logits.squeeze(-1)
    #     end_logits = end_logits.squeeze(-1)
    #     start_logits=start_logits*attention_mask
    #     end_logits=end_logits*attention_mask
    #     outputs = (start_logits,end_logits)
    #     if start_positions is not None and end_positions is not None:
    #         if len(start_positions.size()) > 1:
    #             start_positions = start_positions.squeeze(-1)
    #         if len(end_positions.size()) > 1:
    #             end_positions = end_positions.squeeze(-1)
    #         ignored_index = start_logits.size(1)
    #         start_positions.clamp_(0, ignored_index)
    #         end_positions.clamp_(0, ignored_index)
    #
    #         loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
    #         start_loss = loss_fct(start_logits, start_positions)
    #         end_loss = loss_fct(end_logits, end_positions)
    #         total_loss = (start_loss + end_loss) / 2
    #         outputs = (total_loss,) + outputs
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        start_positions=None,
        end_positions=None,
    ):
        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
        )

        sequence_output = outputs[0]
        batch_size=sequence_output.shape[0]
        sequence_output=self.dropout(sequence_output)
        logits = self.qa_outputs(sequence_output)
        # sequence_output=torch.nn.functional.gelu(logits)
        # logits=self.qa_ou(sequence_output)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        outputs = (start_logits,end_logits)
        if start_positions is not None and end_positions is not None:
            if len(start_positions.size()) > 1:
                start_positions = start_positions.squeeze(-1)
            if len(end_positions.size()) > 1:
                end_positions = end_positions.squeeze(-1)
            ignored_index = start_logits.size(1)
            start_positions.clamp_(0, ignored_index)
            end_positions.clamp_(0, ignored_index)

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=ignored_index)
            start_loss = loss_fct(start_logits, start_positions)
            end_loss = loss_fct(end_logits, end_positions)
            total_loss = (start_loss + end_loss) / 2
            outputs = (total_loss,) + outputs

        return outputs #,(loss), start_logits, end_logits, (hidden_states), (attentions)
