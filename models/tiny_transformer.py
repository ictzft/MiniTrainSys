"""
TinyTransformer — 用于分布式训练实验的小型 Transformer 语言模型。

目标不是追求精度，而是在有限 GPU 资源下对比不同训练模式的性能。
参数量约 10M~50M，支持配置层数、隐藏维度、注意力头数等超参。
"""

import math
import torch
import torch.nn as nn


class TinyTransformer(nn.Module):
    """基于 Transformer Encoder 的小型语言模型。"""

    def __init__(
        self,
        vocab_size: int = 30522,
        d_model: int = 256,
        nhead: int = 8,
        num_layers: int = 4,
        dim_feedforward: int = 1024,
        max_seq_len: int = 512,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.d_model = d_model
        self.max_seq_len = max_seq_len

        # Token embedding + 可学习位置编码
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)
        self.embedding_dropout = nn.Dropout(dropout)

        # Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # 输出头：hidden → vocab_size
        self.output_head = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, vocab_size),
        )

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        """Xavier uniform 初始化，参考 GPT-2 风格。"""
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Args:
            input_ids: (batch_size, seq_len) token ids
            labels:    (batch_size, seq_len) 用于计算 loss，-100 表示忽略

        Returns:
            dict with keys:
                logits: (batch_size, seq_len, vocab_size)
                loss:   scalar (仅当 labels 不为 None 时返回)
        """
        batch_size, seq_len = input_ids.shape
        assert seq_len <= self.max_seq_len, (
            f"seq_len {seq_len} > max_seq_len {self.max_seq_len}"
        )

        # 位置编码
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)

        # Embedding: token + position
        x = self.token_embedding(input_ids) * math.sqrt(self.d_model)
        x = x + self.position_embedding(positions)
        x = self.embedding_dropout(x)

        # Causal mask：防止 attention 看到未来 token
        causal_mask = nn.Transformer.generate_square_subsequent_mask(
            seq_len, device=input_ids.device
        )

        # Transformer Encoder
        x = self.transformer_encoder(x, mask=causal_mask)

        # 输出 logits
        logits = self.output_head(x)

        result = {"logits": logits}

        # 计算 loss
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss(ignore_index=-100)
            # logits: (B, T, V) → (B*V, T)，labels: (B, T) → (B*T,)
            loss = loss_fn(
                logits.view(-1, logits.size(-1)),
                labels.view(-1),
            )
            result["loss"] = loss

        return result

    def count_parameters(self) -> int:
        """返回可训练参数总数。"""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(config: dict) -> TinyTransformer:
    """从配置字典构建模型。"""
    model_cfg = config["model"]
    return TinyTransformer(
        vocab_size=model_cfg["vocab_size"],
        d_model=model_cfg["d_model"],
        nhead=model_cfg["nhead"],
        num_layers=model_cfg["num_layers"],
        dim_feedforward=model_cfg["dim_feedforward"],
        max_seq_len=model_cfg["max_seq_len"],
        dropout=model_cfg.get("dropout", 0.1),
    )


if __name__ == "__main__":
    # 快速验证：用随机数据跑一次 forward
    model = TinyTransformer()
    print(f"模型参数量: {model.count_parameters():,}")

    input_ids = torch.randint(0, 30522, (2, 128))
    labels = torch.randint(0, 30522, (2, 128))

    output = model(input_ids, labels=labels)
    print(f"logits shape: {output['logits'].shape}")
    print(f"loss: {output['loss'].item():.4f}")
