import torch
import torch.nn as nn


class SREncoderLayer(nn.Module):

    def __init__(self, hidden_dim, num_heads):
        super().__init__()

        self.pre_linear = nn.Linear(hidden_dim, hidden_dim)


        self.mha = nn.MultiheadAttention(embed_dim=hidden_dim, num_heads=num_heads, batch_first=True)

        self.post_net = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, x):

        identity = x

        attn_input = self.pre_linear(x)
        attn_output, _ = self.mha(attn_input, attn_input, attn_input)


        combined = torch.cat([identity, attn_output], dim=-1)

        return self.post_net(combined)


class CLSEnhancer(nn.Module):

    def __init__(self, input_dim=768, hidden_dim=1024, num_layers=4, num_heads=8):
        super().__init__()


        self.upper_path = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, input_dim)
        )

        self.lower_input_proj = nn.Linear(input_dim, hidden_dim)


        self.sr_encoders = nn.Sequential(*[
            SREncoderLayer(hidden_dim, num_heads) for _ in range(num_layers)
        ])


        self.lower_output_proj = nn.Linear(hidden_dim, input_dim)

    def forward(self, x):

        x_reshaped = x.unsqueeze(1)

        upper_out = self.upper_path(x_reshaped)

        lower_out = self.lower_input_proj(x_reshaped)
        lower_out = self.sr_encoders(lower_out)
        lower_out = self.lower_output_proj(lower_out)
        out = upper_out + lower_out


        return out.squeeze(1)


# 验证代码
if __name__ == "__main__":
    model = CLSEnhancer()
    test_input = torch.randn(1, 768)
    output = model(test_input)
    print(f"输入维度: {test_input.shape}")
    print(f"输出维度: {output.shape}")