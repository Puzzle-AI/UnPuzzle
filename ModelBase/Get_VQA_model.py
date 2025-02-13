"""
Build VQA models    Script  ver： Sep 25th 15:00
"""
import torch.nn as nn
from MLLM_modules.Get_Language_model import build_language_model
from MLLM_modules.Fusion import MultipleModalityFusion
from Get_ROI_model import ImageEncoder


# ------------------- Text Encoder -------------------
# After tokenisation, the query (question tokens) is passed through the GPT-2 model,
# generating a sequence of hidden states (intermediate representations of input text after learning)
# The last CLS token from the last hidden state from the sequence is selected as the question's vector representation.
# A dropout layer is applied to the text embeddings to prevent overfitting.

class TextEncoder(nn.Module):
    # todo future use a prompt based llm
    # this obtains the question embedding (GPT CLS token)
    def __init__(self, tokenizer_name='gpt2', embed_size=768, dropout_rate=0.1):
        super(TextEncoder, self).__init__()

        self.Text_Encoder = build_language_model(tokenizer_name=tokenizer_name)
        self.dropout = nn.Dropout(dropout_rate)

        self.embed_convert = nn.Linear(self.Text_Encoder.embed_dim, embed_size) \
            if self.Text_Encoder.embed_dim != embed_size else nn.Identity()

    def forward(self, input_ids, attention_mask):
        # Process text through GPT-2 to generate a seq of hidden state
        Text_outputs = self.Text_Encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        Text_cls_embedding = Text_outputs[:, -1, :]  # GPT-2 uses the last token embedding as CLS representation
        Text_cls_embedding = self.dropout(Text_cls_embedding)

        return self.embed_convert(Text_cls_embedding)


# ------------------- Answer Decoder (VQAbyCLS Classifier) -------------------
class AnswerDecoder_VQAbyCLS(nn.Module):
    '''
    The VQAbyCLS is task design that align and train the multiple modal output in a classification manner

    in the output langurage decoding stage:
    The combined features (which now include both the attended image information and the text representation)
    are passed into the answer decoder, which is a linear classifier predicts
    the final answer by producing logits for each possible answer class.

    The output, logits, is a tensor of size [batch_size, num_classes],
    it represents the raw scores for each possible answer class,
    where num_classes is the total number of possible answer classes

    '''

    def __init__(self, embed_size=768, num_classes=None):
        assert num_classes is not None
        super(AnswerDecoder_VQAbyCLS, self).__init__()
        self.classifier = nn.Linear(embed_size * 2, num_classes)

    def forward(self, combined_features):
        # Classification to predict the answer
        logits = self.classifier(combined_features)
        return logits


# ------------------- Full VQA Model -------------------
class VQAModel_VQAbyCLS(nn.Module):
    def __init__(self, image_encoder, text_encoder, fusion_method='MHSA',
                 num_classes=None, embed_size=768, heads=8, dropout_rate=0.1):
        assert num_classes is not None
        super(VQAModel_VQAbyCLS, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        # fusion with clip for future
        self.fusion = MultipleModalityFusion(fusion_method=fusion_method,
                                             embed_size=embed_size, heads=heads, dropout_rate=dropout_rate)

        self.answer_decoder = AnswerDecoder_VQAbyCLS(embed_size=embed_size, num_classes=num_classes)

    def forward(self, images, input_ids, attention_mask):
        # Image encoding
        image_features = self.image_encoder(images)
        # Text encoding
        text_features = self.text_encoder(input_ids, attention_mask)
        # fusion
        combined_features = self.fusion(text_features, image_features)
        # todo can we ask this model to speak its answer?

        # Answer classification [B, 2 * embed_size] -> logits [B, N(num cls)]
        logits = self.answer_decoder(combined_features)

        return logits


def get_VQA_model(model_idx='uni', tokenizer_name='gpt2', fusion_method='MHSA', embed_size=768, dropout_rate=0.1,
                  heads=8, num_classes=None):
    assert num_classes is not None
    # Initialize model
    image_encoder = ImageEncoder(model_idx=model_idx, embed_size=embed_size)
    text_encoder = TextEncoder(tokenizer_name=tokenizer_name, embed_size=embed_size, dropout_rate=dropout_rate)
    VQAModel = VQAModel_VQAbyCLS(image_encoder, text_encoder, fusion_method=fusion_method,
                                 embed_size=embed_size, heads=heads, dropout_rate=dropout_rate,
                                 num_classes=num_classes)
    return VQAModel
