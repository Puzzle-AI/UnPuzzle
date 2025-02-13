import os
import torch
import timm
import torch.nn as nn


# ------------------- Multiple Modality Fusion -------------------
# The text embeddings (query) are passed into the attention mechanism to attend to the image embeddings (key/value).
# The multi-head attention layer computes the attention weights that help the model focus on relevant visual features
# based on the textual query.
# The attended image and text features are concatenated together to form a unified representation of both modalities.
class MultiHeadAttention(nn.Module):
    # In the attention mechanism (both single and multi-head), the core idea is to let the model focus on
    # different parts of the input sequence or different inputs to capture relationships
    # In this Visual Question Answering (VQA) model,
    # the attention mechanism helps the text (the query) focus on the image (to answer the question).
    def __init__(self, embed_size=768, heads=8, dropout_rate=0.1):
        super(MultiHeadAttention, self).__init__()
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embed_size, num_heads=heads)
        self.dropout = nn.Dropout(dropout_rate)

    def forward(self, query, key, value):
        '''
        Query (Q): Represents what you are trying to match or attend to (here it is the text embeddings).
        Key (K): Represents the features to compare against (here it is the image embeddings).
        Value (V): Holds the actual data that will be output after attention is applied (here it is image embeddings).
        The key tells the model where to attend, and the value gives the information for those attended locations.
        '''
        # query, key, value should be [seq_len, batch, embed_size]
        query = query.transpose(0, 1)
        key = key.transpose(0, 1)
        value = value.transpose(0, 1)

        attn_output, attn_weights = self.multihead_attn(query, key, value)
        attn_output = self.dropout(attn_output)

        # Transpose back to [batch, seq_len, embed_size]
        attn_output = attn_output.transpose(0, 1)
        return attn_output, attn_weights


class MultipleModalityFusion(nn.Module):
    # In the attention mechanism (both single and multi-head), the core idea is to let the model focus on
    # different parts of the input sequence or different inputs to capture relationships
    # In this Visual Question Answering (VQA) model,
    # the attention mechanism helps the text (the query) focus on the image (to answer the question).
    def __init__(self, fusion_method='MHSA', embed_size=768, heads=8, dropout_rate=0.1):
        super(MultipleModalityFusion, self).__init__()
        self.fusion_method = fusion_method
        if self.fusion_method == 'MHSA':
            self.attention = MultiHeadAttention(embed_size=embed_size, heads=heads, dropout_rate=dropout_rate)
        elif self.fusion_method == 'clip':
            raise NotImplementedError
        else:
            raise NotImplementedError

    def forward(self, text_features, image_features):
        if self.fusion_method == 'MHSA':
            # Attention between image and text
            query = text_features.unsqueeze(1)  # Text features as query
            key_value = image_features.unsqueeze(1)  # Image features as key/value
            attended_features, _ = self.attention(query, key_value, key_value)

            # Combine attended features with text features [B, 2 * embed_size]
            combined_features = torch.cat((attended_features.squeeze(1), text_features), dim=1)
            # The attended_features (the output of the attention mechanism) is combined with the original text_features
            # attended_features.squeeze(1): Removes the extra dimension added by unsqueeze(1) earlier

            return combined_features
        elif self.fusion_method == 'clip':
            raise NotImplementedError
        else:
            raise NotImplementedError
