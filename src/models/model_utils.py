import torch
from functools import reduce

# mask_with_tokens
def special_token_mask_generation(input_ids, special_token_ids):
    init_no_mask = torch.full_like(input_ids, False, dtype=torch.bool)
    mask_bl = reduce(lambda acc, el: acc | (input_ids == el),
                     special_token_ids, init_no_mask)
    return mask_bl.to(torch.long)


def text_part_mask_generation(input_ids, special_token_ids, attention_mask):
    mask_text_part = (1 - special_token_mask_generation(input_ids, special_token_ids)) * attention_mask
    return mask_text_part


def exp_mask(_mask, _val, high_rank=False):
    _exp_mask = (torch.ones_like(_mask) - _mask).to(_val.dtype) * \
                torch.full([1], fill_value=-10000, dtype=_val.dtype, device=_val.device)
    if high_rank:
        _exp_mask = _exp_mask.unsqueeze(-1).expand_as(_val)
    return _exp_mask + _val


def zero_mask(_mask, _val, high_rank=False):
    _zero_mask = _mask.to(_val.dtype)
    if high_rank:
        _zero_mask = _zero_mask.unsqueeze(-1).expand_as(_val)
    return _zero_mask * _val


def masked_pool(rep_input, rep_mask, high_rank=True, method="mean", return_new_mask=False):

    dim_pool = rep_mask.dim() - 1
    new_mask = (rep_mask.sum(dim=dim_pool) > 0).to(rep_mask.dtype)

    if method == "mean":
        masked_input = zero_mask(rep_mask, rep_input, high_rank=high_rank)
        rep_output = masked_input.sum(dim=dim_pool)
        denominator = rep_mask.to(rep_output.dtype).sum(dim=dim_pool)
        # remove zero
        denominator = torch.where(
            denominator > 0.,
            denominator, torch.full_like(denominator, fill_value=1.)
        )
        if high_rank:
            denominator = denominator.unsqueeze(-1).expand_as(rep_output)
        rep_output /= denominator

    elif method == "max":
        masked_input = exp_mask(rep_mask, rep_input, high_rank=high_rank)
        rep_output = torch.max(masked_input, dim=dim_pool)[0]
    else:
        raise NotImplementedError

    rep_output = zero_mask(new_mask, rep_output, high_rank=high_rank)

    if return_new_mask:
        return rep_output, new_mask
    else:
        return rep_output
