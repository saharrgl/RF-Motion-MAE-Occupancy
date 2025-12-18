import torch
import torch.nn as nn

# ---------- Patch Embedding ----------
class PatchEmbed(nn.Module):
    def __init__(self, dim_in=94, dim_patch=10, embed_dim=128):
        super().__init__()
        self.dim_patch = dim_patch
        self.proj = nn.Linear(dim_in * dim_patch, embed_dim)

    def forward(self, x):
        B, T, F = x.shape
        num_patches = T // self.dim_patch
        x = x[:, :num_patches*self.dim_patch, :]
        x = x.reshape(B, num_patches, self.dim_patch * F)
        return self.proj(x), num_patches


# ---------- MAE ----------
class MAE(nn.Module):
    def __init__(self, embed_dim=128, depth=4, heads=4, mask_ratio=0.4):
        super().__init__()
        self.mask_ratio = mask_ratio
        self.embed = PatchEmbed(embed_dim=embed_dim)

        self.encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(embed_dim, heads, dim_feedforward=256),
            num_layers=depth
        )

        self.decoder = nn.TransformerDecoder(
            nn.TransformerDecoderLayer(embed_dim, heads, dim_feedforward=256),
            num_layers=2
        )

        self.fc_recon = nn.Linear(embed_dim, 94 * 10)

    def forward(self, x):
        x_p, P = self.embed(x)
        B, P, D = x_p.shape
        x_p = x_p.transpose(0, 1)

        keep = int(P * (1 - self.mask_ratio))
        idx = torch.randperm(P)
        idx_keep = idx[:keep]
        idx_mask = idx[keep:]

        x_keep = x_p[idx_keep]
        enc = self.encoder(x_keep)

        mask_tokens = torch.zeros(len(idx_mask), B, D, device=x.device)
        dec_input = torch.cat([enc, mask_tokens], dim=0)

        dec = self.decoder(dec_input, enc)
        out = self.fc_recon(dec.transpose(0, 1))
        out = out.reshape(B, P, 10, 94).permute(0, 1, 3, 2)

        return out, idx_mask


# ---------- Classifier ----------
class MAEClassifier(nn.Module):
    def __init__(self, mae, n_classes=2, dropout_p=0.3):
        super().__init__()
        self.encoder = mae.encoder
        self.dropout = nn.Dropout(dropout_p)
        self.fc = nn.Linear(128, n_classes)

    def forward(self, x):
        x_p, _ = mae.embed(x)
        x_p = x_p.transpose(0, 1)
        enc = self.encoder(x_p)
        feat = enc.mean(dim=0)
        feat = self.dropout(feat)
        return self.fc(feat)
