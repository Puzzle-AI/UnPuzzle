import argparse
import torch
import os
import json
from tqdm import tqdm
import shortuuid
import torch
from PIL import Image
from MLLM_modules.llava.model.builder import load_pretrained_model  # Assuming this function loads the model properly
from MLLM_modules.llava.mm_utils import process_images  # Utility for processing images
from transformers import AutoTokenizer
from MLLM_modules.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images
import json
from MLLM_modules.llava.constants import IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN, DEFAULT_IMAGE_PATCH_TOKEN
from MLLM_modules.llava.conversation import conv_templates, SeparatorStyle
from MLLM_modules.llava.model.builder import load_pretrained_model
from MLLM_modules.llava.utils import disable_torch_init
from MLLM_modules.llava.mm_utils import tokenizer_image_token, get_model_name_from_path, KeywordsStoppingCriteria, process_images


# ------------------- DataLoader -------------------
def llava_med_dataloader(data_file_path, image_processor, device):
    """
    A data loader for the LLaVA-Med dataset.

    Args:
        data_file_path (str): Path to the dataset file.
        image_processor: The image processor from the LLaVA-Med model.
        device (torch.device): The device to process the images.

    Returns:
        list: A list of dictionaries with keys `id`, `image`, and `conversation`.
    """
    dataset = []
    
    # Load dataset
    with open(data_file_path, "r") as file:
        data = json.load(file)

    for entry in data:
        entry_id = entry["id"]
        image_path = entry["image"]
        conversations = entry["conversations"]

        # Open and preprocess image
        try:
            image = Image.open(image_path).convert("RGB")
            processed_image = image_processor(images=image, return_tensors="pt")["pixel_values"].to(device)
        except FileNotFoundError:
            print(f"Image file not found: {image_path}")
            continue
        except Exception as e:
            print(f"Error processing image {image_path}: {e}")
            continue

        # Build conversation text
        conversation_text = ""
        for msg in conversations:
            role = msg["role"]
            content = msg["content"]

            if role == "USER":
                conversation_text += f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{content}\n"
        
        # Add a placeholder for assistant's response
        conversation_text += f"{DEFAULT_IMAGE_TOKEN}\n"

        # Append to dataset
        dataset.append({
            "id": entry_id,
            "image": processed_image,
            "conversation": conversation_text
        })

    return dataset


# ------------------- Model Loader -------------------
def get_llava_med_model(model_path, device):
    """
    Load the LLaVA-Med model, tokenizer, and image processor.
    """
    tokenizer, model, image_processor, _ = load_pretrained_model(
        model_path=model_path,
        model_base=None,
        model_name=model_path.split("/")[-1],
        load_8bit=False,
        load_4bit=False,
        device=device
    )
    return tokenizer, model, image_processor


# ------------------- Main Script -------------------
def main():
    # Device setup
    device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load the model
    model_path = "microsoft/llava-med-v1.5-mistral-7b"
    tokenizer, model, image_processor = get_llava_med_model(model_path, device)
    model.to(device)

    # Load dataset using the dataloader
    data_file_path = "/mnt/data1/changhan/llavamed/LLaVA-Med/Data.txt"
    dataset = llava_med_dataloader(data_file_path, image_processor, device)

    # Iterate over preprocessed dataset
    for entry in dataset:
        image = entry["image"]
        conversation_text = entry["conversation"]

        # Tokenize input text
        input_ids = tokenizer(conversation_text, return_tensors="pt", truncation=True, padding=True)["input_ids"].to(device)

        # Model inference
        try:
            with torch.no_grad():
                generated_ids = model.generate(
                    inputs=input_ids,
                    images=image,
                    do_sample=True,
                    temperature=1.0,
                    top_p=0.9,
                    num_beams=1,
                    max_new_tokens=1024,
                    use_cache=True
                )
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            # Display the generated output
            print(f"Generated Text for ID {entry['id']}:")
            print(generated_text)
            print("=" * 50)
        except Exception as e:
            print(f"Error processing entry ID {entry['id']}: {e}")
            continue


if __name__ == "__main__":
    main()
# def main():
#     # Set up the device
#     device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
#     print(f"Using device: {device}")

#     # Define model path and load tokenizer, model, and image processor
#     model_path = "microsoft/llava-med-v1.5-mistral-7b"
#     tokenizer, model, image_processor, context_len = load_pretrained_model(
#         model_path=model_path,
#         model_base=None,          # Set if there's a specific base model
#         model_name=model_path.split("/")[-1],
#         load_8bit=False,          # Set to True if you want to use 8-bit precision
#         load_4bit=False,          # Set to True if you want to use 4-bit precision
#         device=device
#     )

#     model.to(device)

#     conv = conv_templates["llava_llama_2"].copy()
#     conversation_history = []

#     # Load the dataset
#     with open("/mnt/data1/changhan/llavamed/LLaVA-Med/Data.txt", "r") as file:
#         data = json.load(file)

#     # Process each entry in the dataset
#     for entry in data:
#         image_path = entry["image"]

#         # Load and preprocess the image for this entry
#         image = Image.open(image_path).convert("RGB")
#         processed_image = image_processor(images=image, return_tensors="pt")['pixel_values'].to(device)
#         processed_image = processed_image.half()  # Convert to float16 if required

#         # Initialize the conversation
#         conv.messages = []

#         # Build the conversation up to the last USER message
#         for msg in entry["conversations"]:
#             role = msg['role']
#             content = msg['content']

#             # Validate the role
#             if role not in conv.roles:
#                 raise ValueError(f"Unexpected role {msg['role']}")

#             # Adjust content if the role is USER
#             if role == conv.roles[0]:  # "USER"
#                 # Include image tokens if necessary
#                 if model.config.mm_use_im_start_end:
#                     content = f"{DEFAULT_IM_START_TOKEN}{DEFAULT_IMAGE_TOKEN}{DEFAULT_IM_END_TOKEN}\n{content}"
#                 else:
#                     content = f"{DEFAULT_IMAGE_TOKEN}\n{content}"

#                 # Append the message to the conversation
#                 conv.append_message(role, content)
#             else:
#                 # Skip assistant's previous responses
#                 pass  # Do not include assistant's responses

#         # Append assistant's placeholder
#         conv.append_message(conv.roles[1], None)

#         # Generate the prompt from the conversation
#         prompt = conv.get_prompt()

#         # Debug: Print the conversation messages and the prompt
#         print("Conversation Messages:", conv.messages)
#         print("Generated Prompt:")
#         print(prompt)
#         print("-------------------")

#         # Tokenize the prompt with image tokens
#         input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)


#         # Generate the response with the model
#         with torch.no_grad():
#             output_ids = model.generate(
#                 inputs=input_ids,
#                 images=processed_image,
#                 do_sample=True,
#                 temperature=1.0,
#                 top_p=0.9,
#                 num_beams=1,
#                 max_new_tokens=1024,
#                 use_cache=True
#             )

#         # Decode and print the generated text
#         generated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
#         print(f"Generated Text for image {image_path}:\n{generated_text}")
#         print("=====================================================")

# if __name__ == "__main__":
#     main()



