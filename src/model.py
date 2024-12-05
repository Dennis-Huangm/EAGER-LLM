import torch
import torch.distributed
import torch.nn as nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from typing import List, Optional, Tuple, Union
from transformers.modeling_outputs import CausalLMOutputWithPast, ModelOutput
from transformers.models.llama.modeling_llama import LLAMA_INPUTS_DOCSTRING, _CONFIG_FOR_DOC
from transformers import Cache
from transformers import AutoConfig, AutoModelForCausalLM, LlamaConfig, LlamaModel, LlamaForCausalLM
from info_nce import InfoNCE, info_nce
class LLMBSForCausalLM(LlamaForCausalLM):

    def __init__(self, config):
        super().__init__(config)
        self.fc_proj1 = nn.Linear(4096, 4096)
        self.fc_proj2 = nn.Linear(4096, 1024)
        self.info_nce = InfoNCE(negative_mode='paired')
        
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        semantic_embs: Optional[torch.FloatTensor] = None,
        behavior_embs: Optional[torch.FloatTensor] = None,
        negative_semantic_embs: Optional[torch.FloatTensor] = None,
        negative_behavior_embs: Optional[torch.FloatTensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]

        if self.config.pretraining_tp > 1:
            lm_head_slices = self.lm_head.weight.split(self.vocab_size // self.config.pretraining_tp, dim=0)
            logits = [F.linear(hidden_states, lm_head_slices[i]) for i in range(self.config.pretraining_tp)]
            logits = torch.cat(logits, dim=-1)
        else:
            logits = self.lm_head(hidden_states)
        logits = logits.float()

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss_fct = CrossEntropyLoss()
            shift_logits = shift_logits.view(-1, self.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_labels = shift_labels.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)
        
        
        indices = []

        for i in range(input_ids.size(0)):
            row = input_ids[i]
            nonzero_indices = torch.nonzero(row == 2)
            
            if nonzero_indices.numel() > 0:
                indices.append([i, nonzero_indices[0].item()])
            else:
                indices.append([i, row.size(0) - 1])
        indices = torch.tensor(indices)
        global_feat = hidden_states[indices[:,0],indices[:,1]]
        
        zero_row_indices = torch.nonzero(semantic_embs.abs().sum(dim=1) == 0).squeeze()
        assert(torch.equal(zero_row_indices, torch.nonzero(behavior_embs.abs().sum(dim=1) == 0).squeeze()))
        global_feat = F.normalize(global_feat, dim=-1)
        global_feat_1 = self.fc_proj1(global_feat)
        global_feat_2 = self.fc_proj2(global_feat)

        if zero_row_indices.numel() > 0 and indices.shape[0] == hidden_states.shape[0]:
            if zero_row_indices.numel() == 1:
                zero_row_indices = zero_row_indices.unsqueeze(0)
            semantic_embs[zero_row_indices] = global_feat_1[zero_row_indices]
            behavior_embs[zero_row_indices] = global_feat_2[zero_row_indices]
        
        contra_loss_1 = self.info_nce(global_feat_1, semantic_embs, negative_semantic_embs.view(1, 2048, -1).expand(len(global_feat), -1, -1))
        contra_loss_2 = self.info_nce(global_feat_2, behavior_embs, negative_behavior_embs.view(1, 2048, -1).expand(len(global_feat), -1, -1))

        loss = loss + 0.1 * contra_loss_1 + 0.1 * contra_loss_2
        
        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output
        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )