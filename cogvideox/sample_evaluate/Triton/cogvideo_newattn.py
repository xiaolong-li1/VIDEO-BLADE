import torch
from .kernels.attn_pooling_kernel import attn_with_pooling
from .utils.gilbert3d import gilbert3d
from torch.nn import functional as F
from .utils.tools import timeit
import torch.nn as nn
from .kernels.block_sparse_attn_kernel_with_backward_9_10 import sparse_attention_factory
import time

sparse_attention_fn=sparse_attention_factory(BLOCK_M=128,BLOCK_N=128)
############parameters###############
# 这里的参数是根据实际情况设置的，可能需要根据具体任务进行调整
mask_ratios = {
            1: (0.0, 0.05),    # Top 5% - 完全注意力
            2: (0.05, 0.15),   # 5%-15% - 2x池化
            4: (0.15, 0.25),   # 15%-55% - 4x池化
            8: (0.25, 0.5),     # 55%-100% - 8x池化
            0: (0.5, 1.0)
}
    
use_rearrange=True
width=45
height=30
depth=13
text_length=226
#####################################

def pad_to_multiple(x, multiple):
    """
    在序列维度（dim=2）上填充 x，使其长度为 multiple 的倍数。
    x: [B, H, L, D]
    """
    L = x.size(2)
    remainder = L % multiple
    if remainder != 0:
        pad_len = multiple - remainder
        # 对序列维度在后面补 pad_len 个零（注意 F.pad 参数顺序：最后两个数字对应 dim=2 的左右补充）
        x = F.pad(x, (0, 0, 0, pad_len),mode='replicate')
    return x
def random_sample_tokens(x, block_size=64, sample_num=8):
    """
    对输入 x (shape: [B, H, L, D]) 每 block_size 个 token 分为一块，
    在每个块中随机采样 sample_num 个 token。
    要求 L 是 block_size 的倍数。
    返回采样后的结果，形状为 [B, H, num_blocks * sample_num, D]
    """
    B, H, L, D = x.size()
    num_blocks = L // block_size
    # 重塑为 [B, H, num_blocks, block_size, D]
    x_blocks = x.view(B, H, num_blocks, block_size, D)
    
    # 为每个块生成随机数，并用 topk 选出 sample_num 个随机索引
    rand_vals = torch.rand(B, H, 1, block_size, device=x.device)
    _, indices = torch.topk(rand_vals, sample_num, dim=3)
    # indices 的 shape: [B, H, num_blocks, sample_num]
    
    # 将 indices 扩展到与 x_blocks 最后一个维度 D 对齐
    indices_expanded = indices.unsqueeze(-1).expand(-1, -1, num_blocks, -1, D)
    # 利用 torch.gather 从每个块中采样 token
    sampled = torch.gather(x_blocks, 3, indices_expanded)
    # 重塑回 [B, H, num_blocks * sample_num, D]
    sampled = sampled.view(B, H, num_blocks * sample_num, D)
    return sampled
def efficient_attn_with_pooling(q, k, v, block_size=128, num_keep=32):
    """
    计算下采样后的注意力池化：
      - 从 q, k (shape: [B, H, seq, D]) 中，从 token 224 开始，每 64 个 token 随机采样 num_keep 个 token。
      - 对采样后的 q, k 计算注意力，并用卷积实现下采样，使得效果等效于对原始 attn 做了 64×64 的求和池化。
      - 对 q 与 k 在序列维度做 padding，保证采样时不会丢失末尾数据。
      
    参数：
      block_size: 每个块的 token 数（固定为 64）
      num_keep: 每个块中保留的 token 数（默认 8，可以自定义）
    """
    # 取从 token 224 开始的部分
    q_ = q[:, :, :, :]
    k_ = k[:, :, :, :]
    
    # 在序列维度上 padding 至 block_size 的倍数
    q_ = pad_to_multiple(q_, block_size)
    k_ = pad_to_multiple(k_, block_size)
    
    # 对 q 和 k 分块并随机采样
    sampled_q = random_sample_tokens(q_, block_size, num_keep)  # [B, H, num_blocks*num_keep, D]
    sampled_k = random_sample_tokens(k_, block_size, num_keep)
    _,pooling= attn_with_pooling(
        sampled_q, sampled_k, v, False, 1.0 / (sampled_q.size(-1) ** 0.5), num_keep
    )
    return pooling
def org_attn_with_pooling(q, k, v):
    sm_scale=1.0 / (q.size(-1) ** 0.5)
    causal=False
    block_size=128
    _, pooling = attn_with_pooling(q, k, v, causal, sm_scale,block_size)
    return pooling
def standard_attn(q,k,v):
    mask=torch.ones([q.size(0),q.size(1),q.size(2)//128+1,k.size(2)//128+1],device=q.device,dtype=torch.bool)
    out,lse= block_sparse_attn(q, k, v,block_mask=mask)
    return out,lse
class GilbertRearranger:
    """基于 Gilbert 曲线的序列重排器，用于视频和文本数据的重新排列。"""
    def __init__(self, width, height, depth, text_length=224):
        self.width = width
        self.height = height
        self.depth = depth
        self.total_elements = width * height * depth
        self.text_length = text_length

        coord_to_index = self._gilbert3d_with_index(width, height, depth)
        original_order2gilbert_order = [0] * self.total_elements
        gilbert_order2original_order = [0] * self.total_elements

        for coord_idx, org_idx in coord_to_index.items():
            original_order2gilbert_order[org_idx] = coord_idx
            gilbert_order2original_order[coord_idx] = org_idx

        self.original_order2gilbert_order = torch.tensor(original_order2gilbert_order, dtype=torch.long, device='cuda')
        self.gilbert_order2original_order = torch.tensor(gilbert_order2original_order, dtype=torch.long, device='cuda')

    def _gilbert3d_with_index(self, width, height, depth):
        """生成 Gilbert 曲线的坐标到索引映射。"""
        coord_to_index = {}
        index = 0
        def coord_to_index_func(x, y, z):
            return x + width * (y + height * z)
        for x, y, z in gilbert3d(width, height, depth):
            coord_index = coord_to_index_func(x, y, z)
            coord_to_index[coord_index] = index
            index += 1
        return coord_to_index
    def rearrange(self, q, k, v):
        """将 q、k、v 张量的视频部分按 Gilbert 曲线顺序重排。"""
        seq_dim = -2
        text_part_q, video_part_q = q[..., :self.text_length, :], q[..., self.text_length:, :]
        text_part_k, video_part_k = k[..., :self.text_length, :], k[..., self.text_length:, :]
        text_part_v, video_part_v = v[..., :self.text_length, :], v[..., self.text_length:, :]

        q_rearranged = video_part_q.index_select(seq_dim, self.original_order2gilbert_order)
        k_rearranged = video_part_k.index_select(seq_dim, self.original_order2gilbert_order)
        v_rearranged = video_part_v.index_select(seq_dim, self.original_order2gilbert_order)

        return (torch.cat((q_rearranged,text_part_q), dim=seq_dim),
                torch.cat((k_rearranged,text_part_k), dim=seq_dim),
                torch.cat((v_rearranged,text_part_v), dim=seq_dim))
    
    def reversed_rearrange(self, out):
        """将输出张量的视频部分从 Gilbert 曲线顺序恢复到原始顺序。"""
        seq_dim = -2
        video_part,text_part= out[..., :-self.text_length, :], out[..., -self.text_length:, :]
        out_reversed = video_part.index_select(seq_dim, self.gilbert_order2original_order)
        return torch.cat((text_part, out_reversed), dim=seq_dim)


def transfer_attn_to_mask(attn, mask_ratios=None):
    """
    将注意力权重转换为多级池化掩码矩阵。
    
    Args:
        attn (torch.Tensor): 注意力权重矩阵，形状为 [batch, head, seq, seq]
        mask_ratios (dict): 掩码值对应的百分比范围，格式为 {mask_value: (start_ratio, end_ratio)}
                           默认为 {1: (0.0, 0.05), 2: (0.05, 0.15), 4: (0.15, 0.55), 8: (0.55, 1.0)}
                           其余位置掩码为0（跳过）
    
    Returns:
        torch.Tensor: 多级掩码矩阵，形状同输入
        - 0: 跳过 (不进行注意力计算)
        - 1: 完全注意力 (默认top 5%)
        - 2: 2x池化 (默认5%-15%)
        - 4: 4x池化 (默认15%-55%)
        - 8: 8x池化 (默认55%-100%)
    """
    if mask_ratios is None:
        mask_ratios = {
            1: (0.0, 0.05),    # Top 5% - 完全注意力
            2: (0.05, 0.15),   # 5%-15% - 2x池化
            4: (0.15, 0.55),   # 15%-55% - 4x池化
            8: (0.55, 1.0)     # 55%-100% - 8x池化
        }
    
    batch, heads, seq, _ = attn.shape
    device = attn.device
    # 初始化为int32类型的掩码，默认为0（跳过）
    mask = torch.zeros_like(attn, dtype=torch.int32)
    
    # 批量处理所有查询位置的注意力权重排序
    # attn shape: [batch, heads, seq, seq]
    sorted_weights, indices = torch.sort(attn, dim=-1, descending=True)
    
    # 为每个掩码值设置对应范围 - 批量处理
    for mask_value, (start_ratio, end_ratio) in mask_ratios.items():
        start_idx = max(0, int(seq * start_ratio))
        end_idx = min(seq, int(seq * end_ratio))
        
        if start_idx < end_idx:
            # 创建位置范围掩码 [seq] -> [1, 1, 1, seq]
            position_range = torch.arange(seq, device=device)
            range_mask = (position_range >= start_idx) & (position_range < end_idx)
            range_mask = range_mask.view(1, 1, 1, seq).expand(batch, heads, seq, -1)
            
            # 批量设置掩码值 - 对所有查询位置同时操作
            mask.scatter_(-1, indices, torch.where(range_mask, mask_value, mask.gather(-1, indices)))
    
    # 确保最后两个位置始终有完全注意力（通常是特殊token如EOS等）
    mask[:, :, :, -2:] = 1
    mask[:, :, -2:, :] = 1
    
    return mask


def adaptive_block_sparse_attn(q, k, v):
    """
    Adaptive block-sparse attention mechanism.
    Creates a block mask automatically (based on q, k) without gradient tracking for mask steps.
    Args:
        q: (batch_size, nheads, seqlen, d)
        k: (batch_size, nheads, seqlen, d)
        v: (batch_size, nheads, seqlen, d)
    Returns:
        out: (batch_size, nheads, seqlen, d)
        sparsity: float (0-1)
    """
    sm_scale = 1.0 / (q.size(-1) ** 0.5)
    block_size = 128
    # Disable gradient tracking for pooling and mask operations
    with torch.no_grad():
        pooling = efficient_attn_with_pooling(q, k, v,block_size=block_size)
        mask = transfer_attn_to_mask(pooling,mask_ratios)
    out=sparse_attention_fn(q.contiguous(), k.contiguous(), v.contiguous(), mask, None)
    # out=torch.nn.functional.scaled_dot_product_attention(q,k,v)
    density=0
    for mask_value, (start_ratio, end_ratio) in mask_ratios.items():
        if(mask_value!=0):
            density+=1/mask_value*(end_ratio-start_ratio)
    return out, 1-density


class AdaptiveBlockSparseAttnTrain(nn.Module):
    def __init__(self):
        super().__init__()
        self.gilbert_rearranger = GilbertRearranger(width, height, depth, text_length)
        self.sparsity_acc = 0.0  # Accumulator for sparsity sum
        self.sparsity_counter = 0  # Counter for number of updates
        self.use_rearrange=use_rearrange
    def forward(self, q, k, v):
        if(self.use_rearrange):
            q_r, k_r, v_r = self.gilbert_rearranger.rearrange(q, k, v)
        else:
            q_r=q
            k_r=k
            v_r=v
        # Compute block-sparse attention and get sparsity
        out_r, sparsity = adaptive_block_sparse_attn(q_r, k_r, v_r)
        # Update sparsity statistics
        self.sparsity_acc += sparsity  # Convert tensor to float
        self.sparsity_counter += 1
        # Print average sparsity every 600 calls
        if self.sparsity_counter % 600 == 0:
            avg_sparsity = self.sparsity_acc / self.sparsity_counter
            print(f"sparsity: {avg_sparsity}")
        # Reverse the arrangement
        if(self.use_rearrange):
            out = self.gilbert_rearranger.reversed_rearrange(out_r)
        else:
            out=out_r
        # print(f"out shape: {out.shape},out.dtype: {out.dtype}")
        return out

