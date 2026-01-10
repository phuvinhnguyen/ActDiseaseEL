from transformers import AutoModel, AutoTokenizer
import torch
import os
import fitz
import pathlib
import glob
# import shutil
from datetime import datetime

# os.environ["CUDA_VISIBLE_DEVICES"] = '0' # Set project to use only one GPU

# 1) load model
model_name = 'deepseek-ai/DeepSeek-OCR'
# _attn_implementation='flash_attention_2' # If available, use to speed up inference
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

# The different scaling options for the model, more information available at:
# https://huggingface.co/deepseek-ai/DeepSeek-OCR
size_configs = {
    "Tiny": {"base_size": 512, "image_size": 512, "crop_mode": False},
    "Small": {"base_size": 640, "image_size": 640, "crop_mode": False},
    "Base": {"base_size": 1024, "image_size": 1024, "crop_mode": False},
    "Large": {"base_size": 1280, "image_size": 1280, "crop_mode": False},
    "Gundam": {"base_size": 1024, "image_size": 640, "crop_mode": True},
}

model_size = "Large"

config = size_configs.get(model_size, size_configs["Gundam"])

pdf_list = [
    # Absolute paths to PDFs / Images to OCR
    # Could be improved to load a text/CSV file with paths for flexibility
]

# tell the model how to convert the documents, more information available at:
# https://huggingface.co/deepseek-ai/DeepSeek-OCR
prompt = "<image>\n<|grounding|>Convert the document to text."

# Store the converted results
results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

# Store temporary images when processing PDFs
temp_image_dir = "./_temp_page_images"
os.makedirs(temp_image_dir, exist_ok=True)

# Used for logging, if the script crashes, check logs.txt for guidance
log_path = "./logs.txt"
def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")

log("--- Run started ---")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}

for pdf_path in pdf_list:
    log(f"Starting: {pdf_path}")
    if not os.path.isfile(pdf_path):
        log(f"file not found, skipping: {pdf_path}")
        continue

    try:
        pdf_base = pathlib.Path(pdf_path).stem
        txt_path = os.path.join(results_dir, f"{pdf_base}.txt")
        all_text = []

        ext = pathlib.Path(pdf_path).suffix.lower()

        if ext in IMAGE_EXTS:
            text = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=pdf_path,
                output_path=results_dir,
                base_size=config["base_size"],
                image_size=config["image_size"],
                crop_mode=config["crop_mode"],
                save_results=True,
                test_compress=True,
                eval_mode=True,
            )
            all_text.append(text)

        else:
            doc = fitz.open(pdf_path)

            for i, page in enumerate(doc):
                img_path = os.path.join(temp_image_dir, f"{pdf_base}_page_{i+1}.png")
                pix = page.get_pixmap(dpi=300)
                pix.save(img_path)

                text = model.infer(
                    tokenizer,
                    prompt=prompt,
                    image_file=img_path,
                    output_path=results_dir,
                    base_size=config["base_size"],
                    image_size=config["image_size"],
                    crop_mode=config["crop_mode"],
                    save_results=True,
                    test_compress=True,
                    eval_mode=True,
                )
                all_text.append(text)

            doc.close()

        with open(txt_path, "w", encoding="utf-8") as f:
            f.write("\n\n".join(all_text))

        log(f"finished: {pdf_path} → {txt_path}")

    except Exception as e:
        log(f"error processing {pdf_path}: {str(e)}")
        log("skipping to next file")

    finally:
        for file in glob.glob(os.path.join(temp_image_dir, "*.png")):
            try:
                os.remove(file)
            except:
                pass

log("----- Run finished -----")

