import torch

def compute_grpo_advantage(rewards, eps=1e-8):
    """
    rewards: [B, G]
    return:  [B, G]
    """
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, keepdim=True, unbiased=False)
    adv = (rewards - mean) / (std + eps)
    return adv

import torch

def grpo_loss(
    logp,
    old_logp,
    ref_logp,
    advantages,
    mask,
    clip_eps=0.2,
    beta=0.01,
):
    """
    logp:        [B, G, T]
    old_logp:    [B, G, T]
    ref_logp:    [B, G, T]
    advantages: [B, G]
    mask:        [B, G, T], 1 for response valid tokens, 0 for prompt/pad
    """

    # [B, G, 1] -> broadcast to token level
    adv = advantages.unsqueeze(-1)

    # PPO ratio
    ratio = torch.exp(logp - old_logp)

    unclipped = ratio * adv
    clipped = torch.clamp(ratio, 1 - clip_eps, 1 + clip_eps) * adv

    # maximize objective => minimize negative objective
    policy_loss = -torch.minimum(unclipped, clipped)

    # common GRPO KL estimator
    # x = log(pi_ref / pi_theta)
    x = ref_logp - logp
    kl = torch.exp(x) - x - 1

    loss = policy_loss + beta * kl

    # only response valid tokens count
    loss = (loss * mask).sum() / mask.sum().clamp_min(1.0)

    return loss



import torch.nn.functional as F

def get_token_logprobs(logits, input_ids):
    """
    logits:    [B, T, V]
    input_ids: [B, T]
    return:    [B, T-1], each position predicts next token
    """
    log_probs = F.log_softmax(logits[:, :-1, :], dim=-1)
    target_ids = input_ids[:, 1:]

    token_logp = log_probs.gather(
        dim=-1,
        index=target_ids.unsqueeze(-1)
    ).squeeze(-1)

    return token_logp
