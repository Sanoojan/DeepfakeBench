import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CosineTemporalClassifier(nn.Module):
    def __init__(self, num_patches, embed_dim=128, num_heads=4, num_layers=2, dropout=0.1):
        super().__init__()
        
        # Project per-patch similarities into a higher-dimensional embedding
        self.patch_embed = nn.Sequential(
            nn.Linear(num_patches, embed_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout)
        )
        
        # Temporal modeling (Transformer)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            batch_first=True,
            dropout=dropout
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Classification head (video-level)
        self.fc = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embed_dim // 2, 1)
        )
    
    def forward(self, x):
        """
        x: (B, T, N) - cosine similarity maps across frames and patches
        """
        # Step 1: Patch embedding
        x = self.patch_embed(x)    # (B, T, embed_dim)
        
        # Step 2: Temporal encoding
        x = self.transformer(x)    # (B, T, embed_dim)
        
        # Step 3: Temporal pooling (global average)
        x = x.mean(dim=1)          # (B, embed_dim)
        
        # Step 4: Classification
        logits = self.fc(x).squeeze(1)   # (B,)
        return logits

class LocalTemporalClassifier(nn.Module):
    def __init__(self, num_patches, hidden_dim=128):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(num_patches, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, x):
        x_t = x[:, :-1, :, :]
        x_tp1 = x[:, 1:, :, :]
        diff = (x_tp1 - x_t).pow(2).sum(dim=-1).sqrt()  # [B, T-1, P]
        motion = diff.mean(dim=1)  # [B, P]
        logits = self.fc(motion)
        return logits.squeeze(-1)

class SimpleTemporalClassifier(nn.Module):
    """
    Simple baseline classifier on top of pretrained XCLIP class embeddings.
    Input: [B, T, D] features
    Output: [B] logits
    """
    def __init__(self, input_dim=768, hidden_dim=512, dropout=0.3):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Binary classification

    def forward(self, x):
        """
        x: [B, T, D]
        """
        # Temporal pooling: mean over T frames
        x = x.mean(dim=1)  # [B, D]

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)  # [B, 1]
        return x.squeeze(1)  # [B]

class LearnableTemporalClassifier(nn.Module):
    """
    Temporal classifier with learnable pooling weights over T frames.
    Input: [B, T, D]
    Output: [B] logits
    """
    def __init__(self, input_dim=768, hidden_dim=512, dropout=0.3, max_frames=100):
        super().__init__()
        self.max_frames = max_frames  # maximum number of frames expected
        # Learnable temporal weights (initialized uniform)
        self.temporal_weights = nn.Parameter(torch.ones(max_frames) / max_frames)  # [T_max]
        
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: [B, T, D]
        """
        B, T, D = x.shape
        
        # Get first T weights and normalize
        w = self.temporal_weights[:T]
        w = F.softmax(w, dim=0)  # sum to 1
        
        # Weighted sum over temporal dimension
        x = (x * w.view(1, T, 1)).sum(dim=1)  # [B, D]
        
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        return x.squeeze(1)
    
class TransformerClassifier(nn.Module):
    def __init__(self, in_dim, num_heads=4, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=in_dim, nhead=num_heads, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.fc = nn.Linear(in_dim, 1)

    def forward(self, x):
        x = self.transformer(x)   # (B, T, D)
        x = x.mean(dim=1)         # global average pooling
        logits = self.fc(x)
        return logits.squeeze(1)

class SimpleClassifier(nn.Module):
    def __init__(self, D):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(D, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, z):
        return self.fc(z).squeeze(-1)
    
class PatchTemporalAttention(nn.Module):
    def __init__(self, D, num_patches=88, heads=4, layers=2):
        super().__init__()
        
        self.proj = nn.Linear(D, D)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=D, nhead=heads, batch_first=True
        )
        self.temp_encoder = nn.TransformerEncoder(encoder_layer, layers)
        
        self.fc = nn.Linear(num_patches * D, 1)

    def forward(self, x):
        # x: (B, T, N, D)
        B, T, N, D = x.shape
        
        # Process each patch across time
        x = self.proj(x)   # (B,T,N,D)
        
        x = x.permute(0,2,1,3)   # (B, N, T, D)
        x = x.reshape(B*N, T, D)

        encoded = self.temp_encoder(x)   # (B*N, T, D)
        encoded = encoded.mean(1)        # (B*N, D)

        encoded = encoded.reshape(B, N*D)
        logits = self.fc(encoded)
        return logits.squeeze(1)
    
class StatisticalTemporalClassifier(nn.Module):
    def __init__(self, D, max_lag=5, hidden_dim=256):
        super().__init__()
        self.max_lag = max_lag

        # number of statistics per (lag, N, D)
        # mean, std, energy, cosine mean, cosine std, max-min
        stats_per_lag = 6  

        # feature dimension after pooling
        self.feature_dim = stats_per_lag * max_lag * D

        self.fc = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

    def compute_cos(self, a, b):
        # a,b: [B, T-1, N, D]
        cos = F.cosine_similarity(a, b, dim=-1)  # -> [B, T-1, N]
        return cos

    def forward(self, x):
        """
        x: [B, T, N, D]
        """
        B, T, N, D = x.shape
        features = []

        for lag in range(1, self.max_lag+1):
            if T <= lag: break

            # First order difference with lag
            diff = x[:, lag:] - x[:, :-lag]     # [B, T-lag, N, D]

            # Second order difference for this lag
            if T > lag+1:
                diff2 = diff[:, 1:] - diff[:, :-1]  # [B, T-lag-1, N, D]
            else:
                diff2 = None

            # Cosine similarity differences
            cos = self.compute_cos(x[:, lag:], x[:, :-lag])  # [B, T-lag, N]

            # --- STATISTICS ---
            stats = []

            # mean & std
            stats.append(diff.mean(dim=(1,2)))        # [B, D]
            stats.append(diff.std(dim=(1,2)))         # [B, D]

            # energy (L2)
            stats.append((diff**2).mean(dim=(1,2)))   # [B, D]

            # cosine stats
            stats.append(cos.mean(dim=(1,2)))         # [B]
            stats.append(cos.std(dim=(1,2)))          # [B]

            # amplitude range
            stats.append((diff.max(dim=1).values.max(dim=1).values -
                          diff.min(dim=1).values.min(dim=1).values))  # [B, D]

            # concat lag stats
            lag_vec = torch.cat(
                [s if s.ndim == 2 else s.unsqueeze(-1) for s in stats],
                dim=-1
            )
            features.append(lag_vec)

            # second order
            if diff2 is not None:
                features.append(diff2.mean(dim=(1,2)))  # add 2nd diff mean

        # Collapse all lags → [B, M]
        feat = torch.cat(features, dim=-1)

        out = self.fc(feat).squeeze(-1)  # [B]
        return out
    
class StatisticalTemporalClassifier(nn.Module):
    def __init__(self, T=16, D=768, max_lag=5, hidden_dim=256):
        super().__init__()
        self.max_lag = max_lag
        self.T = T
       
        self.D = D
        self.hidden_dim = hidden_dim

        # ---- Precompute the feature dimension M ----
        M = 0
        for lag in range(1, max_lag + 1):
            if T <= lag:
                break

            # Always adds: s1,s2,s3,s4,s5,amp = 4D + 2
            M += (4 * D + 2)

            # diff2 exists only when T > lag + 1
            if T > lag + 1:
                M += D

        self.M = M

        # ---- Now initialize the classifier FC ----
        self.fc = nn.Sequential(
            nn.Linear(M, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
    
    def compute_cos(self, a, b):
        return F.cosine_similarity(a, b, dim=-1)

    def forward_features(self, x):
        B, T, N, D = x.shape
        features = []

        for lag in range(1, self.max_lag + 1):
            if T <= lag:
                break

            diff = x[:, lag:] - x[:, :-lag]

            if T > lag + 1:
                diff2 = diff[:, 1:] - diff[:, :-1]
            else:
                diff2 = None

            cos = self.compute_cos(x[:, lag:], x[:, :-lag])

            # stats
            s1 = diff.mean(dim=(1,2))
            s2 = diff.std(dim=(1,2))
            s3 = (diff**2).mean(dim=(1,2))
            s4 = cos.mean(dim=(1,2))
            s5 = cos.std(dim=(1,2))
            amp = diff.max(dim=1).values.max(dim=1).values - diff.min(dim=1).values.min(dim=1).values

            vec = torch.cat([s1, s2, s3, s4.unsqueeze(-1), s5.unsqueeze(-1), amp], dim=-1)
            features.append(vec)

            if diff2 is not None:
                features.append(diff2.mean(dim=(1,2)))

        return torch.cat(features, dim=-1)  # [B, M]

    def forward(self, x):
        feat = self.forward_features(x)
        return self.fc(feat).squeeze(-1)
    
    
class PatchTemporalClassifier(nn.Module):
    def __init__(self, D, hidden_dim=128, lag=1):
        super().__init__()
        self.lag = lag
        
        # This FC will be shared across all patches + transitions
        self.patch_fc = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, return_patch_scores=False):
        """
        x: [B, T, N, D]
        """
        B, T, N, D = x.shape
        
        # ----------------------------------------------
        # 1. Temporal difference features
        # ----------------------------------------------

        # diff[t] = x[t+lag] - x[t]
        # diff: [B, T-lag, N, D]
        diff = x[:, self.lag:] - x[:, :-self.lag]
        
        
        # ----------------------------------------------
        # 2. Apply shared FC to each patch transition
        # ----------------------------------------------

        # Flatten: [B*(T-lag)*N, D]
        diff_flat = diff.reshape(B*(T-self.lag)*N, D)
        
        # FC gives a score for each patch transition
        patch_scores_flat = self.patch_fc(diff_flat)  # [B*(T-lag)*N, 2]
        
        # Unflatten back: [B, T-lag, N, 2]
        patch_scores = patch_scores_flat.view(B, T-self.lag, N, 2)
        
        
        # ----------------------------------------------
        # 3. Aggregate patch & temporal scores per video
        # ----------------------------------------------

       
        video_scores = patch_scores.mean(dim=(1, 2))  # [B, 2]

        if return_patch_scores:
            return video_scores, patch_scores
        else:
            return video_scores
        
        
class PatchClassifier(nn.Module):
    def __init__(self, D, hidden_dim=128):
        super().__init__()
        
        
        # This FC will be shared across all patches + transitions
        self.patch_fc = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x, return_patch_scores=False):
        """
        x: [B, T, N, D]
        """
        B, T, N, D = x.shape
        
        # ----------------------------------------------
        # 1. Temporal difference features
        # ----------------------------------------------

        # diff[t] = x[t+lag] - x[t]
        # diff: [B, T-lag, N, D]
        
        
        
        # ----------------------------------------------
        # 2. Apply shared FC to each patch transition
        # ----------------------------------------------

        # Flatten: [B*(T-lag)*N, D]
        x = x.reshape(B*T*N, D)
        
        # FC gives a score for each patch transition
        patch_scores_flat = self.patch_fc(x)  # [B*(T-lag)*N, 2]
        
        # Unflatten back: [B, T-lag, N, 2]
        patch_scores = patch_scores_flat.view(B, T, N, 2)
        
        
        # ----------------------------------------------
        # 3. Aggregate patch & temporal scores per video
        # ----------------------------------------------

       
        video_scores = patch_scores.mean(dim=(1, 2))  # [B, 2]

        if return_patch_scores:
            return video_scores, patch_scores
        else:
            return video_scores

class PatchClassifierCNN3D(nn.Module):
    def __init__(self, D, hidden_dim=128):
        super().__init__()

        # FC per patch → 1 logit
        self.patch_fc = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 3D conv aggregator
        # input: B, 1, T, N1, N1
        self.agg3d = nn.Sequential(
            nn.Conv3d(
                in_channels=1,
                out_channels=8,
                kernel_size=(3, 3, 3),   
                padding=(1, 1, 1)      
            ),
            nn.ReLU(),
            nn.Conv3d(8, 2, kernel_size=1)
        )

    def forward(self, x, return_patch_scores=False):
        """
        x: [B, T, N, D]
        """
        B, T, N, D = x.shape
        N1 = int(math.sqrt(N))
        assert N1 * N1 == N, f"N={N} is not a perfect square!"

        # -----------------------------------------
        # 1. FC per patch → patch_logits: [B, T, N, 1]
        # -----------------------------------------
        x_flat = x.reshape(B*T*N, D)
        patch_flat = self.patch_fc(x_flat)  # [B*T*N, 1]
        patch_scores = patch_flat.view(B, T, N)   # [B, T, N]

        # -----------------------------------------
        # 2. Reshape into spatial grid: [B, T, N1, N1]
        # -----------------------------------------
        patch_grid = patch_scores.view(B, T, N1, N1)

        # Add channel dim → [B, 1, T, N1, N1]
        patch_grid = patch_grid.unsqueeze(1)

        # -----------------------------------------
        # 3. 3D conv aggregation
        # output: [B, 2, T, N1, N1]
        # -----------------------------------------
        out3d = self.agg3d(patch_grid)

        # -----------------------------------------
        # 4. Global average pool → [B, 2]
        # -----------------------------------------
        video_scores = out3d.mean(dim=(2, 3, 4))

        if return_patch_scores:
            # return patch_scores as [B, T, N]
            return video_scores, patch_scores

        return video_scores

class PatchTemporalClassifierCNN3D(nn.Module):
    def __init__(self, D, hidden_dim=128,lag=1):
        super().__init__()

        # FC per patch → 1 logit
        self.patch_fc = nn.Sequential(
            nn.Linear(D, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )

        # 3D conv aggregator
        # input: B, 1, T, N1, N1
        self.agg3d = nn.Sequential(
            nn.Conv3d(1, 8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv3d(8, 2, kernel_size=1)   # output channels = 2
        )
        self.lag = lag

    def forward(self, x, return_patch_scores=False):
        """
        x: [B, T, N, D]
        """
        B, T, N, D = x.shape
        N1 = int(math.sqrt(N))
        
        assert N1 * N1 == N, f"N={N} is not a perfect square!"

        # -----------------------------------------
        # 1. FC per patch → patch_logits: [B, T, N, 1]
        # -----------------------------------------
        x= x[:, self.lag:] - x[:, :-self.lag]  # temporal difference
        x_flat = x.reshape(B*(T-self.lag)*N, D)
        patch_flat = self.patch_fc(x_flat)  # [B*T*N, 1]
        patch_scores = patch_flat.view(B, T-self.lag, N)   # [B, T, N]

        # -----------------------------------------
        # 2. Reshape into spatial grid: [B, T, N1, N1]
        # -----------------------------------------
        patch_grid = patch_scores.view(B, T-self.lag, N1, N1)

        # Add channel dim → [B, 1, T, N1, N1]
        patch_grid = patch_grid.unsqueeze(1)

        # -----------------------------------------
        # 3. 3D conv aggregation
        # output: [B, 2, T, N1, N1]
        # -----------------------------------------
        out3d = self.agg3d(patch_grid)

        # -----------------------------------------
        # 4. Global average pool → [B, 2]
        # -----------------------------------------
        video_scores = out3d.mean(dim=(2, 3, 4))

        if return_patch_scores:
            # return patch_scores as [B, T, N]
            return video_scores, patch_scores

        return video_scores
        
        
class ClsDiffClassifier(nn.Module):
    def __init__(self, D, lag=1,diff_op="sq"):
        super().__init__()
        self.lag = lag
        self.diff_op=diff_op
        # This FC will be shared across all patches + transitions
        self.fc = nn.Linear(D, 2)

    def forward(self, x, return_patch_scores=False):
        '''
        Docstring for forward
        
        :param self: Description
        :param x: B,T,D
        :param return_patch_scores: Description
        '''
        B, T, D = x.shape
        # diff = x[:, self.lag:] - x[:, :-self.lag]  # B, T-lag, D
        if self.diff_op=="no":
            diff=x
        elif self.diff_op=="direct":
            diff = x[:, self.lag:] - x[:, :-self.lag]  # B, T-lag, D
        
        elif self.diff_op=="abs":
            diff = (x[:, self.lag:] - x[:, :-self.lag]).abs()  # B, T-lag, D
        elif self.diff_op=="sq":
            diff = (x[:, self.lag:] - x[:, :-self.lag]).pow(2)  # B, T-lag, D
        elif self.diff_op=="cos":
            diff = F.cosine_similarity(x[:, self.lag:], x[:, :-self.lag], dim=-1).unsqueeze(-1)  # B, T-lag, 1
            raise NotImplementedError("cosine diff not implemented yet")
        # Flatten: [B*(T-lag), D]
        diff_flat = diff.reshape(B*(T-self.lag), D)
        scores_flat = self.fc(diff_flat)  # [B*(T-lag), 2]
        scores = scores_flat.view(B, T - self.lag, 2)
        video_scores = scores.mean(dim=1)  # B,2
        
        if return_patch_scores:
            return video_scores, scores
        else:
            return video_scores