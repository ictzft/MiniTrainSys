"""
mini-FSDP — 不依赖 PyTorch FSDP 的参数分片训练原型。

包含两个模块：
    1. MiniFSDP: 参数分片训练（类似 FSDP）
    2. ColumnParallelLinear / RowParallelLinear: 张量并行线性层（类似 Megatron-LM）

目的：理解分布式训练框架的底层机制，不是为了替代 PyTorch。
"""

from mini_fsdp.wrapper import MiniFSDP
from mini_fsdp.parallel_linear import ColumnParallelLinear, RowParallelLinear

__all__ = ["MiniFSDP", "ColumnParallelLinear", "RowParallelLinear"]
