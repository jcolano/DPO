import torch
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data import RandomSampler

from contextlib import nullcontext
from model import ModelArgs, Transformer
from tokenizer import Tokenizer

import pandas as pd 
from tqdm import tqdm
import numpy as np 

from preprocess import get_tokenizer_model_path
from typing import Dict, List, Union, Tuple, Optional, Literal, Any
import torch.nn.functional as F
import torch.nn as nn

class DPOLoss:
    def __init__(self, beta:float=0.05, label_smoothing:float=0.1, loss_type:str='sigmoid'):
        self.beta = beta
        self.label_smoothing = label_smoothing
        self.loss_type = loss_type
        
      
    def compute_log_probs(self, model, batch, avg_log_prob: bool = False ):
        
    
        # RESOLVE FOR CHOSEN
        # ===================
        # Perform forward pass and get logits
        chosen_targets = torch.roll(batch['concat_chosen'], shifts=-1, dims=1)
        chosen_targets[:, -1] = 0 # Replace the last token of each sequence with the padding token
        _chosen_logits = model.forward(batch['concat_chosen'], chosen_targets)  # Shape: (batch_size, seq_len, vocab_size)        
        
        # Since target is shifted one position to the left we sync the positions of the targets and the logits.
        chosen_targets = chosen_targets[:, 1:].clone()
        chosen_logits = _chosen_logits[:, :-1, :]
        
        # Create a mask to make sure we look only at valid tokens (exclude pad tokens)
        chosen_loss_mask = chosen_targets != 0 # Matrix of True and False

        chosen_per_token_logps = torch.gather(chosen_logits.log_softmax(-1), dim=2, index=chosen_targets.unsqueeze(2)).squeeze(2)

        if avg_log_prob:
            chosen_log_probs = (chosen_per_token_logps * chosen_loss_mask).sum(-1) / chosen_loss_mask.sum(-1)
        else:
            chosen_log_probs = (chosen_per_token_logps * chosen_loss_mask).sum(-1)
        
         
        # RESOLVE FOR REJECTED
        # ====================
        rejected_targets = torch.roll(batch['concat_rejected'], shifts=-1, dims=1)
        rejected_targets[:, -1] = 0 # Replace the last token of each sequence with the padding token
        _rejected_logits = model.forward(batch['concat_rejected'], rejected_targets)
           
        rejected_targets = rejected_targets[:, 1:].clone()
        rejected_logits = _rejected_logits[:, :-1, :]
        rejected_loss_mask = rejected_targets != 0

        # dummy token; we'll ignore the losses on these tokens later
        rejected_targets[rejected_targets == 0] = 0

        rejected_per_token_logps = torch.gather(rejected_logits.log_softmax(-1), dim=2, index=rejected_targets.unsqueeze(2)).squeeze(2)

        if avg_log_prob:
            rejected_log_probs = (rejected_per_token_logps * rejected_loss_mask).sum(-1) / rejected_loss_mask.sum(-1)
        else:
            rejected_log_probs = (rejected_per_token_logps * rejected_loss_mask).sum(-1)

        return chosen_log_probs, rejected_log_probs, _chosen_logits, _rejected_logits

    def dpo_loss(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.
        """
        
        #  Reference: https://arxiv.org/pdf/2305.18290.pdf

        # Compute the log probability ratios
        # Since we already have log probabilities, the log of the ratio of probabilities is equivalente to the difference of log probabilities:
        #           log(a/b) = log(a) - log(b)
        chosen_ratio = policy_chosen_logps - reference_chosen_logps
        rejected_ratio = policy_rejected_logps - reference_rejected_logps
        
        # Apply beta to the log probability ratios
        chosen_ration_scaled = self.beta * chosen_ratio
        rejected_ratio_scaled = self.beta * rejected_ratio 
        
        # Compute the loss for each instance
        losses = F.logsigmoid(chosen_ration_scaled) - F.logsigmoid(rejected_ratio_scaled)

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards

    def dpo_loss_hf_implementation(
        self,
        policy_chosen_logps: torch.FloatTensor,
        policy_rejected_logps: torch.FloatTensor,
        reference_chosen_logps: torch.FloatTensor,
        reference_rejected_logps: torch.FloatTensor,
    ) -> Tuple[torch.FloatTensor, torch.FloatTensor, torch.FloatTensor]:
        """
        Compute the DPO loss for a batch of policy and reference model log probabilities.
        """
        
        # References: Huggingface RTL Library and https://github.com/eric-mitchell/direct-preference-optimization
        pi_logratios = policy_chosen_logps - policy_rejected_logps
        ref_logratios = reference_chosen_logps - reference_rejected_logps

        logits = pi_logratios - ref_logratios
        
        # if ipo:
        #     losses = (logits - 1/(2 * beta)) ** 2  # Eq. 17 of https://arxiv.org/pdf/2310.12036v2.pdf
        # else:
        #     # Eq. 3 https://ericmitchell.ai/cdpo.pdf; label_smoothing=0 gives original DPO (Eq. 7 of https://arxiv.org/pdf/2305.18290.pdf)
        #     losses = -F.logsigmoid(beta * logits) * (1 - label_smoothing) - F.logsigmoid(-beta * logits) * label_smoothing

        if self.loss_type == "sigmoid":
            losses = (
                -F.logsigmoid(self.beta * logits) * (1 - self.label_smoothing)
                - F.logsigmoid(-self.beta * logits) * self.label_smoothing
            )
        elif self.loss_type == "hinge":
            losses = torch.relu(1 - self.beta * logits)
        elif self.loss_type == "ipo":
            losses = (logits - 1 / (2 * self.beta)) ** 2
        elif self.loss_type == "kto_pair":
            chosen_KL = (policy_chosen_logps - reference_chosen_logps).mean().clamp(min=0)
            rejected_KL = (policy_rejected_logps - reference_rejected_logps).mean().clamp(min=0)

            chosen_logratios = policy_chosen_logps - reference_chosen_logps
            rejected_logratios = policy_rejected_logps - reference_rejected_logps

            losses = torch.cat(
                (
                    1 - F.sigmoid(self.beta * (chosen_logratios - rejected_KL)),
                    1 - F.sigmoid(self.beta * (chosen_KL - rejected_logratios)),
                ),
                0,
            )
        else:
            raise ValueError(f"Unknown loss type: {self.loss_type}. Should be one of ['sigmoid', 'hinge', 'ipo', 'kto_pair']")

        chosen_rewards = self.beta * (policy_chosen_logps - reference_chosen_logps).detach()
        rejected_rewards = self.beta * (policy_rejected_logps - reference_rejected_logps).detach()

        return losses, chosen_rewards, rejected_rewards
    
    def compute_loss(
        self,
        model,
        batch: Dict[str, Union[List, torch.LongTensor]],
    ) -> Union[torch.Tensor, Tuple[torch.Tensor, Dict[str, torch.Tensor]]]:

        
        # CALCULATE THE POLICY LOGPS
        # ==========================
        policy_chosen_logps, policy_rejected_logps, _, _ = self.compute_log_probs(model, batch, avg_log_prob=False)
        
        # GET THE REFERENCE LOGPS
        # ==========================
        reference_chosen_logps = batch["reference_chosen_logps"]
        reference_rejected_logps = batch["reference_rejected_logps"]

        losses, chosen_rewards, rejected_rewards = self.dpo_loss(
            policy_chosen_logps,
            policy_rejected_logps,
            reference_chosen_logps,
            reference_rejected_logps,
        )
        
        loss = losses.mean()
        chosen_reward = chosen_rewards.mean()
        rejected_reward = rejected_rewards.mean()

        return loss, chosen_reward, rejected_reward, policy_chosen_logps, policy_rejected_logps