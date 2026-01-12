import argparse
import os
import fitz
import pathlib
import glob
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, required=True, help="GPU ID (0-3)")
parser.add_argument("--total_gpus", type=int, default=4, help="Total number of GPUs")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

from transformers import AutoModel, AutoTokenizer
import torch
import os
model_name = 'deepseek-ai/DeepSeek-OCR'

# _attn_implementation='flash_attention_2'
tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModel.from_pretrained(model_name, trust_remote_code=True, use_safetensors=True)
model = model.eval().cuda().to(torch.bfloat16)

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
]

# split files across GPUs
my_files = pdf_list[args.gpu::args.total_gpus]

prompt = "Convert the document, include images and tables."

results_dir = "./results"
os.makedirs(results_dir, exist_ok=True)

temp_image_dir = f"./_temp_page_images_gpu_{args.gpu}"
os.makedirs(temp_image_dir, exist_ok=True)

log_path = f"./logs_gpu_{args.gpu}.txt"

def log(message):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(f"[{timestamp}] {message}\n")
    print(f"[GPU {args.gpu}] {message}")

log(f"----- Run started on GPU {args.gpu} -----")
log(f"Processing {len(my_files)} files out of {len(pdf_list)} total")

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}

for pdf_path in my_files:

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

log(f"----- Run finished on GPU {args.gpu} -----")