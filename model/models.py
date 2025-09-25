import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import torch.nn.init as init
from torchvision import transforms, utils

import numpy as np
from PIL import Image
import os
import math
import torchvision.models as models
from torchvision.ops import generalized_box_iou

from utils import box_cxcywh_to_xyxy, box_iou, bbox_iou, ciou_loss_xyxy, init_weights

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class FaceDetectionDataset(Dataset):
    def __init__(self, images, annotations):
        self.images = images
        self.annotations = annotations

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        images = self.images[idx]
        annotation = self.annotations[idx]

        # Convert numpy arrays to tensors
        # Process annotation to get boxes and labels in the format expected by the loss function
        # We need a list of dicts, one dict per image in the batch
        # Each dict should have 'boxes' (M, 4) and 'labels' (M,)
        # M is the number of objects in the image
        # 'boxes' should be in cxcywh normalized format
        # 'labels' should be raw 0..C-1

        # Remove padding and the class_id (which is always 0 for faces)
        # The 0th element in each annotation entry is the class_id. The following 4 are cx, cy, w, h.
        # We need to filter out entries where all box coordinates are zero (padding)
        annotation = torch.from_numpy(annotation).float()
        valid_annotations = annotation[(annotation[:, 1:] != 0).any(dim=1)]

        if valid_annotations.shape[0] > 0:
            # Extract bounding boxes (cx, cy, w, h) which are already normalized
            boxes = valid_annotations[:, 1:5].float()
            # Extract labels (which is the first element, class_id)
            # Since we only have face class (0), the label will always be 0
            labels = (valid_annotations[:, 0].long() + 1) # Should be 0..C-1
            # Ensure labels are 0-indexed if they represent class IDs
            # In this dataset, class_id is 0, so labels are already 0-indexed.
            # For DETR, we need class IDs 0...C, where 0 is the null class.
            # Since the original class_id is 0, adding 1 makes it 1.
            # If the null class is 0, then the face class should be 1.
            # The detr_loss function handles null class internally, so here we just need correct class ID (1 for face).

            # If class_id is 0 and null_class_idx is 0, and we want class IDs 0...C where 0 is null,
            # then the face class should be 1. The current dataset has class_id 0 for faces.
            # Let's assume the face class is indeed class 1, and class 0 is the null class.
            # So, the labels should be 1 for faces.
            labels = torch.ones_like(valid_annotations[:, 0].long()) # All valid annotations are faces (class 1)

        else:
            # If no objects, provide empty tensors
            boxes = torch.empty((0, 4), dtype=torch.float32)
            labels = torch.empty((0,), dtype=torch.int64)


        target = {'boxes': boxes, 'labels': labels}

        return images, target

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm1 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act1 = nn.PReLU(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.norm2 = nn.GroupNorm(num_groups=8, num_channels=out_channels)
        self.act2 = nn.PReLU(out_channels)
        self.residual = nn.Identity() if in_channels == out_channels else nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.residual_act = nn.PReLU(out_channels)

    def forward(self, x):
        res = self.residual(x)
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        x = self.act2(x)
        x = self.residual_act(x + res)
        return x

class PositionEncodingSine(nn.Module):
    def __init__(self, num_pos_feats=128, temperature=10000, normalize=True, scale=None):
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        if scale is not None and normalize is False:
            raise ValueError("normalize should be True if scale is passed")
        self.scale = scale if scale is not None else 2 * torch.pi

    def forward(self, x):
        """
        x: Tensor of shape (B, C, H, W)
        returns: positional encodings of shape (B, C, H, W)
        """
        b, c, h, w = x.shape
        mask = torch.zeros(b, h, w, dtype=torch.bool, device=x.device)  # no padding here
        not_mask = ~mask

        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)

        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(self.num_pos_feats, dtype=torch.float32, device=x.device)
        dim_t = self.temperature ** (2 * (dim_t // 2) / self.num_pos_feats)

        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t

        pos_x = torch.stack((pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()), dim=4).flatten(3)
        pos_y = torch.stack((pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()), dim=4).flatten(3)

        pos = torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)  # (B, C, H, W)
        return pos

class Backbone(nn.Module):
    def __init__(self, embedding_size=256):
        super().__init__()
        self.layer1 = nn.ModuleList([
            ResidualBlock(3, 32),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # ResidualBlock(32, 32),
        ])
        self.layer2 = nn.ModuleList([
            ResidualBlock(32, 64),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # ResidualBlock(64, 64),
        ])
        self.layer3 = nn.ModuleList([
            ResidualBlock(64, 128),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # ResidualBlock(128, 128),
        ])
        self.layer4 = nn.ModuleList([
            ResidualBlock(128, 256),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # ResidualBlock(256, 256),
        ])
        self.layer5 = nn.ModuleList([
            ResidualBlock(256, 256),
            nn.AvgPool2d(kernel_size=2, stride=2),
            # ResidualBlock(256, 256),
        ])
        self.fc = nn.Conv2d(256, embedding_size, kernel_size=3, stride=1, padding=1) if embedding_size != 256 else nn.Identity()
        self.apply(init_weights)

    def forward(self, x):
        for layer in self.layer1:
            x = layer(x)
        for layer in self.layer2:
            x = layer(x)
        for layer in self.layer3:
            x = layer(x)
        for layer in self.layer4:
            x = layer(x)
        for layer in self.layer5:
            x = layer(x)
        x = self.fc(x)
        return x


class MultiheadAttention(nn.Module):
    def __init__(self, embedding_size, num_heads, dropout=0.0):
        super(MultiheadAttention, self).__init__()
        assert embedding_size % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = embedding_size // num_heads
        self.q_proj = nn.Linear(embedding_size, embedding_size)
        self.k_proj = nn.Linear(embedding_size, embedding_size)
        self.v_proj = nn.Linear(embedding_size, embedding_size)
        self.fc_out = nn.Linear(embedding_size, embedding_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, q, k, v, mask=None):
        B, Nq, D = q.shape
        Nk = k.shape[1]
        q = self.q_proj(q).view(B, Nq, self.num_heads, self.head_dim).permute(0,2,1,3)
        k = self.k_proj(k).view(B, Nk, self.num_heads, self.head_dim).permute(0,2,1,3)
        v = self.v_proj(v).view(B, Nk, self.num_heads, self.head_dim).permute(0,2,1,3)

        attn_scores = torch.matmul(q, k.transpose(-2,-1)) / math.sqrt(self.head_dim)
        if mask is not None:
            attn_scores = attn_scores.masked_fill(mask[:,None,None,:]==0, float('-inf'))
        attn_probs = F.softmax(attn_scores, dim=-1)
        attn_probs = self.dropout(attn_probs)
        attn_output = torch.matmul(attn_probs, v) # (B, H, Nq, D_head)
        attn_output = attn_output.permute(0,2,1,3).contiguous().view(B,Nq,D)
        return self.fc_out(attn_output)

class TransformerEncoderBlock(nn.Module):
    def __init__(self, embedding_size, hidden_embedding_size, num_heads):
        super(TransformerEncoderBlock, self).__init__()
        # self.multihead_attention = MultiheadAttention(embedding_size, num_heads)
        self.multihead_attention = nn.MultiheadAttention(embedding_size, num_heads)
        self.norm1 = nn.LayerNorm(embedding_size)
        self.norm2 = nn.LayerNorm(embedding_size)
        self.fc = nn.Sequential(
            nn.Linear(embedding_size, hidden_embedding_size),
            nn.PReLU(),
            nn.Linear(hidden_embedding_size, embedding_size)
        )

    def forward(self, x, pos_encoding):
        '''
        attn_output = self.multihead_attention(x, x, x)
        x = self.norm1(x + attn_output)
        fc_output = self.fc(x)
        x = self.norm2(x + fc_output)
        return x
        '''
        # Pre-LN перед attention
        x_norm = self.norm1(x) + pos_encoding
        # Transpose for MultiheadAttention
        x_norm_t = x_norm.transpose(0, 1)
        attn_output_t, _ = self.multihead_attention(x_norm_t, x_norm_t, x_norm_t)
        # Transpose back
        attn_output = attn_output_t.transpose(0, 1)
        x = x + attn_output  # residual

        # Pre-LN перед FF
        x_norm = self.norm2(x) + pos_encoding
        fc_output = self.fc(x_norm)
        x = x + fc_output  # residual

        return x

class TransformerDecoderBlock(nn.Module):
    def __init__(self, embedding_size, hidden_embedding_size, num_heads, dropout=0.0):
        super(TransformerDecoderBlock, self).__init__()
        # Masked self-attention
        # self.self_attention = MultiheadAttention(embedding_size, num_heads, dropout)
        self.self_attention = nn.MultiheadAttention(embedding_size, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(embedding_size)

        # Cross-attention
        # self.cross_attention = MultiheadAttention(embedding_size, num_heads, dropout)
        self.cross_attention = nn.MultiheadAttention(embedding_size, num_heads, dropout=dropout)
        self.norm2 = nn.LayerNorm(embedding_size)

        # Feed-forward
        self.fc = nn.Sequential(
            nn.Linear(embedding_size, hidden_embedding_size),
            nn.PReLU(),
            nn.Linear(hidden_embedding_size, embedding_size)
        )
        self.norm3 = nn.LayerNorm(embedding_size)

    def forward(self, dec_input, enc_output, pos_encoding, self_mask=None, enc_mask=None):
        '''
        # 1. Masked self-attention
        attn_out = self.self_attention(dec_input,dec_input,dec_input, mask=self_mask)
        x = self.norm1(dec_input + attn_out)

        # 2. Cross-attention (Q=decoder, K,V=encoder)
        attn_out = self.cross_attention(x, enc_output + pos_encoding,enc_output + pos_encoding, mask=enc_mask)  # тут K,V берутся из enc_output
        x = self.norm2(x + attn_out)

        # 3. Feed-forward
        fc_out = self.fc(x)
        x = self.norm3(x + fc_out)
        '''

        # 1. Masked self-attention
        x_norm = self.norm1(dec_input)
        # Transpose for MultiheadAttention
        x_norm_t = x_norm.transpose(0, 1)
        attn_out_t, _ = self.self_attention(x_norm_t,x_norm_t,x_norm_t)
        # Transpose back
        attn_out = attn_out_t.transpose(0, 1)
        x = dec_input + attn_out

        # 2. Cross-attention (Q=decoder, K,V=encoder)
        x_norm = self.norm2(x)
        # Transpose for MultiheadAttention
        x_norm_t = x_norm.transpose(0, 1)
        enc_output_t = enc_output.transpose(0, 1)
        pos_encoding_t = pos_encoding.transpose(0, 1)
        attn_out_t, _ = self.cross_attention(x_norm_t, enc_output_t + pos_encoding_t,enc_output_t + pos_encoding_t)  # тут K,V берутся из enc_output
        # Transpose back
        attn_out = attn_out_t.transpose(0, 1)
        x = x + attn_out

        # 3. Feed-forward
        x_norm = self.norm3(x)
        fc_out = self.fc(x_norm)
        x = x + fc_out

        return x

class FullTransformer(nn.Module):
    def __init__(self, embedding_size, hidden_embedding_size, num_heads, num_encoder_layers, num_decoder_layers, fm_max_size=50, num_queries=100, dropout=0.0):
        super(FullTransformer, self).__init__()
        self.encoder = nn.ModuleList([
            TransformerEncoderBlock(embedding_size, hidden_embedding_size, num_heads)
            for _ in range(num_encoder_layers)
        ])
        self.decoder = nn.ModuleList([
            TransformerDecoderBlock(embedding_size, hidden_embedding_size, num_heads, dropout)
            for _ in range(num_decoder_layers)
        ])
        self.query_embed = nn.Embedding(num_queries, embedding_size)
        self.positional_embedding_decoder = nn.Embedding(num_queries, embedding_size)
        # Positional encoding for features
        # We need to figure out the size of the feature map from ResNet
        # ResNet18 with input (240, 240, 3) and removing the last two layers
        # Goes through conv1 (stride 2), maxpool (stride 2), layer1 (stride 1), layer2 (stride 2), layer3 (stride 2), layer4 (stride 2)
        # 240 -> 120 -> 60 -> 60 -> 30 -> 15 -> 8 approximately
        # The size should be (H/32, W/32) for a standard ResNet feature map before the last two layers.
        # 240 / 32 is not an integer. ResNet has stride 2, 2, 2, 2, 2 (from conv1, maxpool, layer2, layer3, layer4) total stride is 32
        # 240 / 32 = 7.5.  The actual output size depends on padding. Let's check the output shape from ResNet.
        # print(features_.shape) # (B, 512, 8, 8) -> 512 channels, 8x8 spatial size for 240x240 input
        self.fm_h = 8
        self.fm_w = 8
        self.pos_encoder = PositionEncodingSine(num_pos_feats=embedding_size//2)
        self.apply(init_weights)

    def forward(self, image_features, enc_mask=None):
        # image_features are now the precomputed features from ResNet
        B, C, H, W = image_features.shape

        # Generate positional embeddings for features
        pos_emb = self.pos_encoder(image_features)

        # Add positional embeddings to image features
        image_features = image_features + pos_emb

        image_features = image_features.permute(0, 2, 3, 1)  # (B, H, W, C)
        image_features = image_features.view(B, H * W, C) # Flatten spatial dimensions

        pos_emb = pos_emb.permute(0, 2, 3, 1)
        pos_emb = pos_emb.view(B, H * W, C)

        # Pass through encoder
        enc_output = image_features
        for i, encoder_block in enumerate(self.encoder):
            enc_output = encoder_block(enc_output, pos_emb)

        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        pos_emb_dec = self.positional_embedding_decoder.weight.unsqueeze(0).repeat(B, 1, 1)

        # Pass through decoder
        dec_output = queries
        for i, decoder_block in enumerate(self.decoder):
            dec_output = decoder_block(dec_output + pos_emb_dec, enc_output, pos_emb, enc_mask=enc_mask)

        return dec_output
    
    def forward_lite(self, image_features, encoder_layers=-1, decoder_layers=-1, enc_mask=None):
        if encoder_layers == -1:
            encoder_layers = self.num_encoder_layers
        if decoder_layers == -1:
            decoder_layers = self.num_decoder_layers

        B, C, H, W = image_features.shape

        # Generate positional embeddings for features
        pos_emb = self.pos_encoder(image_features)

        # Add positional embeddings to image features
        image_features = image_features + pos_emb

        image_features = image_features.permute(0, 2, 3, 1)  # (B, H, W, C)
        image_features = image_features.view(B, H * W, C) # Flatten spatial dimensions

        pos_emb = pos_emb.permute(0, 2, 3, 1)
        pos_emb = pos_emb.view(B, H * W, C)

        # Pass through encoder
        enc_output = image_features
        for i, encoder_block in enumerate(self.encoder):
            if i < encoder_layers:
                enc_output = encoder_block(enc_output, pos_emb)

        queries = self.query_embed.weight.unsqueeze(0).repeat(B, 1, 1)
        pos_emb_dec = self.positional_embedding_decoder.weight.unsqueeze(0).repeat(B, 1, 1)

        # Pass through decoder
        dec_output = queries
        for i, decoder_block in enumerate(self.decoder):
            if i < decoder_layers:
                dec_output = decoder_block(dec_output + pos_emb_dec, enc_output, pos_emb, enc_mask=enc_mask)

        return dec_output

class ShrinkingMLP(nn.Module):
    def __init__(self, hidden_dim, min_dim=4, alpha=0.01):
        super().__init__()
        layers = []
        dim = hidden_dim
        while dim > min_dim:
            next_dim = max(min_dim, dim // 2)
            layers.append(nn.Linear(dim, next_dim))
            layers.append(nn.PReLU())
            layers.append(nn.LayerNorm(next_dim))
            dim = next_dim
        self.net = nn.Sequential(*layers)

        self.net.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_uniform_(m.weight, a=0.2, nonlinearity='leaky_relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)

class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MLP, self).__init__()
        # Simplify MLP for bbox regression
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.PReLU()
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.relu2 = nn.PReLU()
        self.fc3 = nn.Linear(hidden_dim, 4) # Output 4 values for cxcywh

        self.sigmoid = nn.Sigmoid()
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0.0, std=1e-3)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu2(x)
        x = self.fc3(x)
        x = self.sigmoid(x) # Ensure output is between 0 and 1 for normalized boxes
        return x

class Classifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, classes_count):
        super(Classifier, self).__init__()
        output_dim = classes_count + 1 # +1 for the null class
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.PReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        # Softmax will be applied in the loss function (CrossEntropyLossWithLogits)
        # self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        # x = self.softmax(x)
        return x

class FullModel(nn.Module):
    def __init__(self, embedding_size, hidden_embedding_size, num_heads, num_encoder_layers, num_decoder_layers, classes_count, num_queries=100, dropout=0.0):
        super(FullModel, self).__init__()
        # The backbone is not part of this module anymore as features are precomputed
        self.backbone = Backbone(embedding_size)

        self.transformer = FullTransformer(embedding_size, hidden_embedding_size, num_heads, num_encoder_layers, num_decoder_layers, num_queries=num_queries, dropout=dropout) # Adjusted fm_max_size
        self.classifier = Classifier(embedding_size, embedding_size, classes_count)
        self.mlp = MLP(embedding_size, embedding_size)
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers

    def forward(self, image, enc_mask=None):
        image_features = self.backbone(image)
        dec_output = self.transformer(image_features.float(), enc_mask) # dec_output shape (B, num_queries, embedding_size)

        classifier_output = self.classifier(dec_output) # (B, num_queries, classes_count + 1)
        mlp_output = self.mlp(dec_output) # (B, num_queries, 4)

        return classifier_output, mlp_output
    
    def forward_lite(self, image, encoder_layers=-1, decoder_layers=-1, enc_mask=None):
        if encoder_layers == -1:
            encoder_layers = self.num_encoder_layers
        if decoder_layers == -1:
            decoder_layers = self.num_decoder_layers
        image_features = self.backbone(image)
        dec_output = self.transformer.forward_lite(image_features.float(), encoder_layers, decoder_layers, enc_mask)
        classifier_output = self.classifier(dec_output)
        mlp_output = self.mlp(dec_output)
        return classifier_output, mlp_output