"""
MiniFSDP 包装器 — 将模型包装为参数分片训练模式。

工作流程：
    1. 初始化时将参数 flatten 并分片
    2. 前向时 all-gather 收集完整参数，注入模型，计算，释放
    3. 反向时 all-gather 收集参数，autograd 计算梯度，reduce-scatter 拆分
    4. 各 rank 只更新自己的参数分片

用法：
    model = MyModel()
    fsdp_model = MiniFSDP(model, device=device)

    # 训练循环
    output = fsdp_model(input_ids)
    output["loss"].backward()
    fsdp_model.step(optimizer)
"""

import torch
import torch.distributed as dist

from mini_fsdp.shard import all_gather_params, reduce_scatter_grads, shard_parameters


class MiniFSDP:
    """
    不依赖 PyTorch FSDP 的参数分片训练包装器。

    核心原理：
        - 参数 flatten 后按 element 数均匀分片到各 rank
        - 前向/反向时 all-gather 收集完整参数
        - 反向后 reduce-scatter 拆分梯度
        - 各 rank 只更新自己的参数分片
    """

    def __init__(self, model: torch.nn.Module, device: torch.device):
        """
        Args:
            model: 要包装的模型
            device: 计算设备
        """
        self.model = model
        self.device = device
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        # 收集所有参数
        self.all_params = list(model.parameters())
        self.param_shapes = [p.shape for p in self.all_params]
        self.param_numels = [p.numel() for p in self.all_params]

        # 分片
        self.param_shard = shard_parameters(self.all_params, self.world_size, self.rank)
        self.param_shard = self.param_shard.to(device).requires_grad_(True)

        # 记录分片信息
        total_elements = sum(self.param_numels)
        shard_size = total_elements // self.world_size
        print(f"[MiniFSDP] rank {self.rank}: shard {self.param_shard.numel()} / {total_elements} elements")

        # 注册 forward hook 用于注入参数
        self._hooks = []
        self._register_hooks()

    def _register_hooks(self):
        """注册 forward hook，在前向时注入 all-gather 后的参数。"""
        for param in self.all_params:
            param._original_data = param.data.clone()

    def _inject_params(self, full_flat: torch.Tensor):
        """将 flat tensor 注入回模型参数。"""
        offset = 0
        for param in self.all_params:
            numel = param.numel()
            param.data = full_flat[offset:offset + numel].view(param.shape).contiguous()
            offset += numel

    def _restore_params(self):
        """恢复参数为原始分片状态。"""
        for param in self.all_params:
            if hasattr(param, '_original_data'):
                param.data = param._original_data

    def forward(self, *args, **kwargs) -> dict:
        """
        前向传播。

        1. All-gather 收集完整参数
        2. 注入模型
        3. 计算 forward
        4. 释放完整参数（保留分片）

        Returns:
            模型的输出（包含 logits 和 loss）
        """
        # All-gather 收集完整参数
        full_flat = all_gather_params(
            self.param_shard, self.world_size, self.rank, self.device
        )

        # 截取有效长度（去掉 padding）
        total_elements = sum(self.param_numels)
        full_flat = full_flat[:total_elements]

        # 注入参数
        self._inject_params(full_flat)

        # 计算 forward
        output = self.model(*args, **kwargs)

        # 保存 loss 用于 backward
        if "loss" in output:
            self._loss = output["loss"]

        # 释放完整参数，保留分片
        del full_flat
        torch.cuda.empty_cache()

        return output

    def backward(self):
        """
        反向传播。

        1. All-gather 收集完整参数（autograd 需要）
        2. 计算 loss.backward()
        3. Reduce-scatter 拆分梯度
        4. 更新 param_shard 的梯度
        """
        if not hasattr(self, '_loss'):
            raise RuntimeError("没有 loss，请先调用 forward")

        # All-gather 收集完整参数（backward 需要）
        full_flat = all_gather_params(
            self.param_shard, self.world_size, self.rank, self.device
        )
        total_elements = sum(self.param_numels)
        full_flat = full_flat[:total_elements]

        # 注入参数
        self._inject_params(full_flat)

        # 计算 backward
        self._loss.backward()

        # 收集完整梯度
        full_grad = torch.zeros_like(full_flat)
        offset = 0
        for param in self.all_params:
            if param.grad is not None:
                numel = param.numel()
                full_grad[offset:offset + numel] = param.grad.reshape(-1)
            offset += param.numel()

        # Reduce-scatter 拆分梯度
        shard_grad = reduce_scatter_grads(
            full_grad, self.world_size, self.rank, self.device
        )

        # 更新 param_shard 的梯度
        if self.param_shard.grad is None:
            self.param_shard.grad = shard_grad
        else:
            self.param_shard.grad += shard_grad

        # 清理
        del full_flat, full_grad
        torch.cuda.empty_cache()

    def step(self, optimizer: torch.optim.Optimizer):
        """
        执行 optimizer step，更新参数分片。

        1. 用 param_shard 的梯度更新 param_shard
        2. 将更新后的分片写回模型参数

        Args:
            optimizer: 绑定到 param_shard 的优化器
        """
        optimizer.step()
        optimizer.zero_grad()

        # 将更新后的分片写回模型参数
        self._update_model_params()

    def _update_model_params(self):
        """将 param_shard 的数据写回模型参数。"""
        # All-gather 收集更新后的完整参数
        full_flat = all_gather_params(
            self.param_shard, self.world_size, self.rank, self.device
        )
        total_elements = sum(self.param_numels)
        full_flat = full_flat[:total_elements]

        # 注入参数
        self._inject_params(full_flat)

        # 释放
        del full_flat
        torch.cuda.empty_cache()

    def parameters(self):
        """返回 param_shard，用于构建 optimizer。"""
        return [self.param_shard]

    def __call__(self, *args, **kwargs):
        """使 MiniFSDP 实例可以像模型一样被调用。"""
        return self.forward(*args, **kwargs)
