# Very similar script to deepseekocr.py, but designed to work with folders of images.
# Script defined to work with the A40 GPUs on Alvis, defined by the jobscriptA40
import argparse
import os
from transformers import AutoModel, AutoTokenizer
import torch
import random
import fitz
import pathlib
import glob
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, required=True, help="GPU ID (0-3)")
parser.add_argument("--total_gpus", type=int, default=4, help="Total number of GPUs")
args = parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

model_name = 'deepseek-ai/DeepSeek-OCR'

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

# Settings CONSTANTS
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".tif", ".tiff", ".bmp", ".gif"}
PDF_EXTS = {".pdf"}
VALID_EXTS = IMAGE_EXTS | PDF_EXTS
SAMPLES_PER_FOLDER = 15
RANDOM_SEED = 42

folder_list = [
    # Absolute paths to folders of PDFs/images to OCR
]

random.seed(RANDOM_SEED)

all_files = []
for folder in folder_list:
    if os.path.isdir(folder):
        folder_files = []
        for entry in os.listdir(folder):
            filepath = os.path.join(folder, entry)
            if os.path.isfile(filepath) and pathlib.Path(filepath).suffix.lower() in VALID_EXTS:
                folder_files.append(filepath)
        
        if len(folder_files) <= SAMPLES_PER_FOLDER:
            all_files.extend(folder_files)
        else:
            sampled = random.sample(folder_files, SAMPLES_PER_FOLDER)
            all_files.extend(sampled)

all_files.sort()

my_files = all_files[args.gpu::args.total_gpus]

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
log(f"Processing {len(my_files)} files out of {len(all_files)} total")

for file_path in my_files:

    log(f"Starting: {file_path}")

    if not os.path.isfile(file_path):
        log(f"file not found, skipping: {file_path}")
        continue

    try:
        file_base = pathlib.Path(file_path).stem
        txt_path = os.path.join(results_dir, f"{file_base}.txt")
        all_text = []

        ext = pathlib.Path(file_path).suffix.lower()

        if ext in IMAGE_EXTS:
            text = model.infer(
                tokenizer,
                prompt=prompt,
                image_file=file_path,
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
            doc = fitz.open(file_path)

            for i, page in enumerate(doc):
                img_path = os.path.join(temp_image_dir, f"{file_base}_page_{i+1}.png")
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

        log(f"finished: {file_path} → {txt_path}")

    except Exception as e:
        log(f"error processing {file_path}: {str(e)}")
        log("skipping to next file")

    finally:
        for temp_file in glob.glob(os.path.join(temp_image_dir, "*.png")):
            try:
                os.remove(temp_file)
            except:
                pass

log(f"----- Run finished on GPU {args.gpu} -----")