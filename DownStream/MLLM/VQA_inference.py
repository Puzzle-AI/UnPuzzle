"""
Run VQA inference for ROI/WSI fixme nos its dummy model   Script  ver： Oct 3rd 21:00
"""
import os
import timm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Model
import torch
import torch.nn as nn


# ------------------- Image Encoder (ViT) -------------------
# Pre-processed image tensor is passed through the Vision Transformer (ViT), to obtain image embedding (ViT CLS token)
class ImageEncoder(nn.Module):
    def __init__(self, embed_size=768):
        super(ImageEncoder, self).__init__()

        # Pre-trained Vision Transformer (ViT)
        self.Image_Encoder = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
        self.embed_convert = nn.Linear(self.Image_Encoder.embed_dim, embed_size) \
            if self.Image_Encoder.embed_dim != embed_size else nn.Identity()

    def forward(self, images):
        # Process image through Image_Encoder to get the embeddings
        Image_cls_embedding = self.Image_Encoder(images)  # CLS token output from ViT [B,D]
        return self.embed_convert(Image_cls_embedding)


# ------------------- Text Encoder -------------------
# After tokenisation, the query (question tokens) is passed through the GPT-2 model,
# generating a sequence of hidden states (intermediate representations of input text after learning)
# The last CLS token from the last hidden state from the sequence is selected as the question's vector representation.
# A dropout layer is applied to the text embeddings to prevent overfitting.

class TextEncoder(nn.Module):
    # this obtains the question embedding (GPT CLS token)
    def __init__(self, tokenizer_name='gpt2', embed_size=768, dropout_rate=0.1):
        super(TextEncoder, self).__init__()
        # Pre-trained GPT-2 (768)
        self.Text_Encoder = GPT2Model.from_pretrained('gpt2') if tokenizer_name == 'gpt2' else None
        self.dropout = nn.Dropout(dropout_rate)

        self.embed_convert = nn.Linear(self.Text_Encoder.embed_dim, embed_size) \
            if self.Text_Encoder.embed_dim != embed_size else nn.Identity()

    def forward(self, input_ids, attention_mask):
        # Process text through GPT-2 to generate a seq of hidden state
        Text_outputs = self.Text_Encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        Text_cls_embedding = Text_outputs[:, -1, :]  # GPT-2 uses the last token embedding as CLS representation
        Text_cls_embedding = self.dropout(Text_cls_embedding)

        return self.embed_convert(Text_cls_embedding)


def get_tokenizer(tokenizer_name='gpt2'):
    if tokenizer_name == 'gpt2':
        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        # Padding is used to ensure that all input sequences in a batch are of the same length.
        # EOS (End of sequence model) make the end of a seq to let model know input text is done
        tokenizer.pad_token = tokenizer.eos_token  # pad with eos, (use eos_token as pad_token)
    else:
        raise NotImplementedError
    return tokenizer


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


# ------------------- Answer Generation Decoder (GPT-2 for Text Generation) -------------------
class AnswerGenerator(nn.Module):
    '''
    This class uses GPT-2 to generate a natural language answer, conditioned on image-text fused features.
    '''

    def __init__(self, gpt2_model_name='gpt2', embed_size=768):
        super(AnswerGenerator, self).__init__()
        # GPT-2 model for text generation
        self.answer_generator = GPT2LMHeadModel.from_pretrained(gpt2_model_name)
        self.embed_size = embed_size

        # Linear layer to transform the fused image-text features to GPT-2 hidden state size
        self.feature_transform = nn.Linear(embed_size * 2, self.answer_generator.config.n_embd)

    def forward(self, fused_features, input_ids, attention_mask):
        '''
        Takes the fused image and text features and uses GPT-2 to generate text based on those features.
        '''
        # Transform the fused features to the hidden state size of GPT-2
        transformed_features = self.feature_transform(fused_features)

        # Instead of using past_key_values, we concatenate the transformed features to input IDs
        # This is a simple approach to condition the generation on the fused features
        expanded_features = transformed_features.unsqueeze(1).expand(-1, input_ids.size(1), -1)
        input_embeds = self.answer_generator.transformer.wte(input_ids) + expanded_features

        outputs = self.answer_generator(inputs_embeds=input_embeds, attention_mask=attention_mask)
        return outputs

    def generate_answer(self, fused_features, input_ids, attention_mask, max_length=50, temperature=0.7, top_k=50):
        '''
        Generates an answer token-by-token, conditioned on the fused image-text features.
        '''
        input_ids = input_ids.to(fused_features.device)
        attention_mask = attention_mask.to(fused_features.device)

        generated = input_ids  # Start with the input_ids as the initial sequence
        current_length = input_ids.size(1)  # Track the current sequence length

        for _ in range(max_length):
            # Get the logits for the next token
            outputs = self.forward(fused_features, generated, attention_mask)
            next_token_logits = outputs.logits[:, -1, :]

            # Apply top-k sampling to the logits
            filtered_logits = self.top_k_sampling(next_token_logits, top_k=top_k)

            # Sample the next token
            next_token = torch.multinomial(torch.softmax(filtered_logits, dim=-1), num_samples=1)

            # Append the sampled token to the generated sequence
            generated = torch.cat((generated, next_token), dim=1)

            # Update attention mask: add a 1 for the new token to attend to all previous tokens
            attention_mask = torch.cat(
                [attention_mask, torch.ones((attention_mask.size(0), 1), device=attention_mask.device)], dim=1)

            # Increment the length
            current_length += 1

            # Stop when EOS token is generated
            if next_token.item() == self.answer_generator.config.eos_token_id:
                break

        return generated

    def top_k_sampling(self, logits, top_k=50):
        '''
        Filters the logits to keep only the top_k tokens with the highest probability, and set others to -inf.
        '''
        top_k = min(top_k, logits.size(-1))  # Safety check
        indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
        logits[indices_to_remove] = -float('Inf')
        return logits


# ------------------- Full VQA Model with Answer Generation -------------------
class VQAModel_GPT2AnswerGeneration(nn.Module):
    def __init__(self, image_encoder, text_encoder, fusion_method='MHSA', embed_size=768, heads=8, dropout_rate=0.0):
        super(VQAModel_GPT2AnswerGeneration, self).__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.fusion = MultipleModalityFusion(fusion_method=fusion_method, embed_size=embed_size,
                                             heads=heads, dropout_rate=dropout_rate)
        self.answer_generator = AnswerGenerator(embed_size=embed_size)

    def forward(self, images, input_ids, attention_mask):
        # Encode the image
        image_features = self.image_encoder(images)
        # Encode the text (question)
        text_features = self.text_encoder(input_ids, attention_mask)
        # Fuse the image and text features
        fused_features = self.fusion(text_features, image_features)
        return fused_features

    def generate_answer(self, images, input_ids, attention_mask, max_length=50, temperature=0.7, top_k=50):
        # Forward pass to get the fused image-text features
        fused_features = self.forward(images, input_ids, attention_mask)
        # Generate the natural language answer conditioned on the fused features
        generated_answer_tokens = self.answer_generator.generate_answer(fused_features, input_ids, attention_mask,
                                                                        max_length, temperature, top_k)
        return generated_answer_tokens


# ------------------- Testing Function for Answer Generation -------------------
def test_model_with_input_generation(model, image_input, text, tokenizer, device,
                                     max_length=50, tries=5, temperature=0.7, top_k=50):
    """
    Test the model with a single input (image and question).
    The function will print the generated natural language answer.
    """
    # Set the model to evaluation mode
    model.eval()

    for index in range(tries):
        image = image_input.unsqueeze(0).to(device)  # Add batch dimension

        # Tokenize the question using GPT-2 tokenizer
        question_tk = tokenizer(
            text,
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=top_k
        )
        input_ids = question_tk['input_ids'].to(device)
        attention_mask = question_tk['attention_mask'].to(device)

        with torch.no_grad():
            # Generate the answer using the model
            generated_answer_tokens = model.generate_answer(image, input_ids, attention_mask,
                                                            max_length=max_length, temperature=temperature, top_k=top_k)

        # Decode the generated tokens into a human-readable text
        generated_answer = tokenizer.decode(generated_answer_tokens.squeeze(0), skip_special_tokens=True)

        # Decode and print the question and generated answer
        question = tokenizer.decode(input_ids.squeeze(0), skip_special_tokens=True)
        print(f"Question: {question}")
        print(f"Generated Answer: {generated_answer}")


if __name__ == '__main__':
    EMBED_SIZE = 768
    DROP_RATE = 0.01
    HEADS = 8
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # ------------------- Model components -------------------
    # Initialize the new model for answer generation
    image_encoder = ImageEncoder(embed_size=EMBED_SIZE)
    text_encoder = TextEncoder(embed_size=EMBED_SIZE, dropout_rate=DROP_RATE)
    # Load the pre-trained weights from the VQAbyCLS model
    # image_encoder.load_state_dict(torch.load('image_encoder.pth'))
    # text_encoder.load_state_dict(torch.load('text_encoder.pth'))
    tokenizer = get_tokenizer(tokenizer_name='gpt2')

    # Freeze the encoders to avoid updating them during answer generation
    for param in image_encoder.parameters():
        param.requires_grad = False

    for param in text_encoder.parameters():
        param.requires_grad = False

    # ------------------- Testing model -------------------
    # Initialize the answer generation model with the pre-trained encoders
    model = VQAModel_GPT2AnswerGeneration(image_encoder, text_encoder, fusion_method='MHSA',
                                          embed_size=EMBED_SIZE, heads=HEADS, dropout_rate=DROP_RATE)
    model = torch.compile(model)
    model.to(device)

    # ------------------- Testing Function for Answer Generation -------------------
    image = torch.randn([3, 224, 224])  # pseudo image
    text = 'Hey is this not a question but just for fun?'

    # Test the model with an input from the dataset and generate a natural language answer
    test_model_with_input_generation(model, image, text, tokenizer, device)
