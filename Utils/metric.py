import torch
import numpy as np
from torch import Tensor


def vqa_compute_score_with_logits(logits:Tensor,
                              labels:Tensor) -> Tensor:
    """
    Given logits for each answer in VQA classification, select answer with max logits and returns VQA-score for that answer

    Args:
        logits (Tensor): logits for each answer. Shape: [batch_size, num_answers]
        labels (Tensor): label for each answer in {0, 0.3, 0.6, 1} [batch_size, num_answers]

    Returns:
        Tensor: score of predicted answer. Shape: [batch_size, num_answers]
    """
    
    logits = torch.max(logits,1)[1].data
    one_hots = torch.zeros(*labels.size()).to(logits.device)
    one_hots.scatter_(1,logits.view(-1,1),1)
    scores = (one_hots * labels)
    scores = torch.sum(scores,1)
    acc = torch.mean(scores)
    return acc
    