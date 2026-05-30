"""
参数分片工具 — 将模型参数拆分到多个 rank。

核心函数：
    shard_parameters: 将参数列表按 element 数量均匀分片
    all_gather_params: 收集所有分片，拼成完整参数
    reduce_scatter_grads: 将梯度求和后按 rank 拆分
"""

import torch
import torch.distributed as dist


def shard_parameters(
    parameters: list[torch.Tensor],
    world_size: int,
    rank: int,
) -> list[torch.Tensor]:
    """
    将参数列表按 element 数量均匀分片。

    不是按参数个数分，而是按总 element 数分，保证每张卡分到的计算量相近。

    Args:
        parameters: 模型的所有参数（flat 后）
        world_size: GPU 总数
        rank: 当前 GPU 编号

    Returns:
        当前 rank 持有的参数分片（flat tensor）
    """
    # 将所有参数 flatten 成一个大 tensor
    flat = torch.cat([p.detach().reshape(-1) for p in parameters])
    total_elements = flat.numel()

    # 计算每个 rank 的分片范围
    shard_size = total_elements // world_size
    start = rank * shard_size
    # 最后一个 rank 处理余数
    end = total_elements if rank == world_size - 1 else start + shard_size

    return flat[start:end].clone()


def all_gather_params(
    local_shard: torch.Tensor,
    world_size: int,
    rank: int,
    device: torch.device,
) -> torch.Tensor:
    """
    收集所有 rank 的分片，拼成完整参数 tensor。

    Args:
        local_shard: 当前 rank 持有的参数分片
        world_size: GPU 总数
        rank: 当前 GPU 编号
        device: 计算设备

    Returns:
        完整的参数 tensor
    """
    # 确保所有分片大小一致（最后一个 rank 可能有余数，需要 padding）
    shard_size = torch.tensor([local_shard.numel()], device=device)
    max_shard_size = shard_size.clone()
    dist.all_reduce(max_shard_size, op=dist.ReduceOp.MAX)

    # Padding 到统一大小
    if local_shard.numel() < max_shard_size.item():
        padding = torch.zeros(
            max_shard_size.item() - local_shard.numel(),
            device=device, dtype=local_shard.dtype,
        )
        padded_shard = torch.cat([local_shard, padding])
    else:
        padded_shard = local_shard

    # All-gather
    gathered = [
        torch.zeros(max_shard_size.item(), device=device, dtype=local_shard.dtype)
        for _ in range(world_size)
    ]
    dist.all_gather(gathered, padded_shard)

    # 拼接并截取有效长度
    full_tensor = torch.cat(gathered)[:local_shard.numel() * world_size]

    return full_tensor


def reduce_scatter_grads(
    full_grad: torch.Tensor,
    world_size: int,
    rank: int,
    device: torch.device,
) -> torch.Tensor:
    """
    将完整梯度求和后按 rank 拆分。

    Args:
        full_grad: 完整的梯度 tensor
        world_size: GPU 总数
        rank: 当前 GPU 编号
        device: 计算设备

    Returns:
        当前 rank 对应的梯度分片
    """
    total_elements = full_grad.numel()
    shard_size = total_elements // world_size

    # 计算每个 rank 的分片
    shards = []
    for i in range(world_size):
        start = i * shard_size
        end = total_elements if i == world_size - 1 else start + shard_size
        shards.append(full_grad[start:end])

    # Reduce-scatter：每个 rank 得到所有 rank 在该分片上的梯度之和
    output = torch.zeros_like(shards[rank])
    dist.reduce_scatter_tensor(output, torch.cat(shards), op=dist.ReduceOp.SUM)

    return output
