import argparse
import pandas as pd
from PIL import Image
from tqdm import tqdm
import torch

# Example: Using transformers pipeline for VQA (replace with your model as needed)
from transformers import BlipProcessor, BlipForQuestionAnswering
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_dir', type=str, required=True, help='Path to image folder')
    parser.add_argument('--csv_path', type=str, required=True, help='Path to image-metadata CSV')
    args = parser.parse_args()

    # Load metadata CSV
    df = pd.read_csv(args.csv_path)

    # Load model and processor, move model to GPU if available
    # MODEL_PATH = r"blip_finetuned_model"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


    BASE_MODEL = "Salesforce/blip-vqa-base"
    ADAPTER_PATH = "blip_finetuned_model"

    # Load base model and processor
    processor = BlipProcessor.from_pretrained(BASE_MODEL)
    model = BlipForQuestionAnswering.from_pretrained(BASE_MODEL).to(device)

    # Load adapter
    model.load_adapter(ADAPTER_PATH)


    generated_answers = []
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        image_path = f"{args.image_dir}/{row['image_id']}"
        question = str(row['question'])
        try:
            image = Image.open(image_path).convert("RGB")
            encoding = processor(image, question, return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**encoding)
                logits = outputs.logits
                predicted_idx = logits.argmax(-1).item()
                answer = model.config.id2label[predicted_idx]
        except Exception as e:
            answer = "error"
        # Ensure answer is one word and in English (basic post-processing)
        answer = str(answer).split()[0].lower()
        generated_answers.append(answer)

    df["generated_answer"] = generated_answers
    df.to_csv("results.csv", index=False)


if __name__ == "__main__":
    main()