import torch
from olmo.model import OLMo

def init_metrics(bsz,seq):
    return {
        'C_i_gij_hist': torch.zeros(100),
        'Σj_Σi_gij': 0,
        'Σj_Σi_abs_gij': 0,
        # C_j_pij = abs(Σj pij) / Σj abs(pij)
        'C_j_pij': torch.zeros(bsz,seq), 
        'Σj_pij': torch.zeros(bsz,seq),
        'Σj_abs_pij': torch.zeros(bsz,seq),
        # C_j_qj = abs(Σj qj) / Σj abs(qj)
        'C_j_qj': 0, 
        'Σj_qj': 0,
        'Σj_abs_qj': 0,
        # Cg = Σj abs(Σi pij) / Σj Σi abs(pij)
        'Cg': 0,
        'Σj_abs_Σi_pij': 0,
        'Σj_Σi_abs_pij': 0,
        # Cug = Σi abs(Σj pij) / Σj Σi abs(pij)
        'Cug': 0,
        'Σi_abs_Σj_pij': 0,
        # CuG = abs(Σj Σi pij)) / Σj abs(Σi pij)
        'CuG': 0,
        'Σj_Σi_pij': 0,
        # C_i_dli = abs(Σi Σj pij)) / Σi abs(Σj pij)
        'C_i_dli': 0,
        # M_i_dli = Σi abs(Σj pij) / N
        'M_i_dli': 0,
        #
        'norm_l2sq_delta': 0,
        'norm_l2sq_G': 0,
        'norm_l2sq_gi': torch.zeros(bsz,seq),
        'norm_l1_delta': 0,
        'norm_l1_G': 0,
        'norm_l1_gi': torch.zeros(bsz,seq),
        # TODO: parallelize for speedup
        # TODO: in-seq vs out-seq
        # TODO: implement for zsl-tmp
    }

@torch.inference_mode()
def reduce_metrics_olmo(model: OLMo, bsz: int, seq: int):
    model_metrics = init_metrics(bsz,seq)
    param_metrics = {}
    hist_bins = torch.linspace(0,1,101)
    N = bsz*seq

    for n,p in model.named_parameters():
        m = init_metrics(bsz,seq)

        m['C_i_gij_hist'] = (p.sum_i_gij.abs().cpu() / p.sum_i_abs_gij.cpu()).histogram(hist_bins)[0]
        m['Σj_Σi_gij'] = p.sum_i_gij.sum().item()
        m['Σj_Σi_abs_gij'] = p.sum_i_abs_gij.sum().item()
        model_metrics['Σj_Σi_gij'] += m['Σj_Σi_gij']
        model_metrics['Σj_Σi_abs_gij'] += m['Σj_Σi_abs_gij']
        model_metrics['C_i_gij_hist'] += m['C_i_gij_hist']

        sum_j_pij = torch.cat([torch.stack(x, dim=-1) for x in p.sum_j_pij], dim=0)
        sum_j_abs_pij = torch.cat([torch.stack(x, dim=-1) for x in p.sum_j_abs_pij], dim=0)
        m['Σj_pij'] += sum_j_pij
        m['Σj_abs_pij'] += sum_j_abs_pij
        m['C_j_pij'] = m['Σj_pij'].abs() / m['Σj_abs_pij']
        model_metrics['Σj_pij'] += m['Σj_pij']
        model_metrics['Σj_abs_pij'] += m['Σj_abs_pij']
        # model_metrics['C_j_pij'] computed below

        qj = p.sum_i_gij.mul(p.delta).div(N)
        m['Σj_qj'] = qj.sum()
        m['Σj_abs_qj'] = qj.abs().sum()
        m['C_j_qj'] = m['Σj_qj'].abs() / m['Σj_abs_qj']
        model_metrics['Σj_qj'] += m['Σj_qj']
        model_metrics['Σj_abs_qj'] += m['Σj_abs_qj']
        # model_metrics['C_j_qj'] computed below

        sum_i_pij = p.sum_i_gij.mul(p.delta)
        m['Σj_Σi_pij'] = sum_i_pij.sum()
        m['Σj_Σi_abs_pij'] = sum_j_abs_pij.sum()
        m['Σj_abs_Σi_pij'] = sum_i_pij.abs().sum()
        m['Σi_abs_Σj_pij'] = sum_j_pij.abs().sum()
        model_metrics['Σj_Σi_pij'] += m['Σj_Σi_pij']
        model_metrics['Σj_Σi_abs_pij'] += m['Σj_Σi_abs_pij']
        model_metrics['Σj_abs_Σi_pij'] += m['Σj_abs_Σi_pij']
        # model_metrics['Σi_abs_Σj_pij'] computed below

        m['Cg'] = m['Σj_abs_Σi_pij'] / m['Σj_Σi_abs_pij']
        m['Cug'] = m['Σi_abs_Σj_pij'] / m['Σj_Σi_abs_pij']
        m['CuG'] = m['Σj_Σi_pij'].abs() / m['Σj_abs_Σi_pij']
        m['C_i_dli'] = m['Σj_Σi_pij'].abs() / m['Σi_abs_Σj_pij']
        m['M_i_dli'] = m['Σi_abs_Σj_pij'].abs() / N
        # model metrics computed below

        m['norm_l2sq_delta'] = p.delta.square().sum().item()
        m['norm_l2sq_G'] = p.sum_i_gij.div(N).square().sum().item()
        m['norm_l2sq_gi'] = torch.cat([torch.stack(x, dim=-1) for x in p.norm_l2sq_gi], dim=0)
        model_metrics['norm_l2sq_delta'] += m['norm_l2sq_delta']
        model_metrics['norm_l2sq_G'] += m['norm_l2sq_G']
        model_metrics['norm_l2sq_gi'] += m['norm_l2sq_gi']

        m['norm_l1_delta'] = p.delta.abs().sum().item()
        m['norm_l1_G'] = p.sum_i_gij.div(N).abs().sum().item()
        m['norm_l1_gi'] = torch.cat([torch.stack(x, dim=-1) for x in p.norm_l1_gi], dim=0)
        model_metrics['norm_l1_delta'] += m['norm_l1_delta']
        model_metrics['norm_l1_G'] += m['norm_l1_G']
        model_metrics['norm_l1_gi'] += m['norm_l1_gi']

        param_metrics[n] = m

    model_metrics['C_j_pij'] = model_metrics['Σj_pij'].abs() / model_metrics['Σj_abs_pij']
    model_metrics['C_j_qj'] = model_metrics['Σj_qj'].abs() / model_metrics['Σj_abs_qj']
    model_metrics['Σi_abs_Σj_pij'] = model_metrics['Σj_pij'].abs().sum()
    model_metrics['Cg'] = model_metrics['Σj_abs_Σi_pij'] / model_metrics['Σj_Σi_abs_pij']
    model_metrics['Cug'] = model_metrics['Σi_abs_Σj_pij'] / model_metrics['Σj_Σi_abs_pij']
    model_metrics['CuG'] = model_metrics['Σj_Σi_pij'].abs() / model_metrics['Σj_abs_Σi_pij']
    model_metrics['C_i_dli'] = model_metrics['Σj_Σi_pij'].abs() / model_metrics['Σi_abs_Σj_pij']
    model_metrics['M_i_dli'] = model_metrics['Σi_abs_Σj_pij'] / N

    return {
        'model': model_metrics,
        'param': param_metrics
    }