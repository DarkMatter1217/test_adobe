import os
import re
import json
import fitz
from PIL import Image
import torch
from transformers import LayoutLMv3Processor, LayoutLMv3ForTokenClassification

MODEL_PATH = "layoutlmv3-finetuned"
DEVICE = torch.device("cpu")

processor = LayoutLMv3Processor.from_pretrained(MODEL_PATH, local_files_only=True, apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(MODEL_PATH, local_files_only=True).to(DEVICE)
model.eval()

def normalize_bbox(bbox, width, height):
    x0, y0, x1, y1 = bbox
    nx0 = int(1000 * x0 / width)
    ny0 = int(1000 * y0 / height)
    nx1 = int(1000 * x1 / width)
    ny1 = int(1000 * y1 / height)
    return [max(0, nx0), max(0, ny0), min(1000, nx1), min(1000, ny1)]

def extract_words_and_boxes(page):
    page_width, page_height = page.rect.width, page.rect.height
    raw_words = page.get_text("words")
    results = []
    for word_tuple in raw_words:
        x0, y0, x1, y1, word = word_tuple[:5]
        if not word.strip():
            continue
        bbox = normalize_bbox((x0, y0, x1, y1), page_width, page_height)
        results.append((word, bbox))
    return results

def aggregate_label_spans(words, labels):
    results = []
    prev_label = None
    chunk = []
    chunk_indices = []
    for idx, (t, l) in enumerate(zip(words, labels)):
        base = l[2:] if l.startswith("B-") or l.startswith("I-") else l
        if base == "O":
            if chunk and prev_label:
                text = " ".join(chunk)
                results.append({"label": prev_label, "text": text, "indices": chunk_indices})
                chunk, chunk_indices = [], []
                prev_label = None
            continue
        if prev_label != base and chunk and prev_label:
            text = " ".join(chunk)
            results.append({"label": prev_label, "text": text, "indices": chunk_indices})
            chunk, chunk_indices = [t], [idx]
            prev_label = base
        else:
            chunk.append(t)
            chunk_indices.append(idx)
            prev_label = base
    if chunk and prev_label:
        results.append({"label": prev_label, "text": " ".join(chunk), "indices": chunk_indices})
    return results

def predict_page_structure(page, processor, model, device):
    words_and_boxes = extract_words_and_boxes(page)
    if not words_and_boxes:
        return [], []
    words, boxes = zip(*words_and_boxes)
    if not words or not boxes or len(words) != len(boxes):
        return [], []
    image = page.get_pixmap()
    pil_image = Image.frombytes("RGB", [image.width, image.height], image.samples)
    encoding = processor(
        images=pil_image,
        words=list(words),
        boxes=list(boxes),
        return_tensors="pt",
        truncation=True
    )
    encoding = {k: v.to(device) for k, v in encoding.items()}
    with torch.no_grad():
        output = model(**encoding)
    pred_ids = output.logits.argmax(dim=2)[0].tolist()
    id2label = model.config.id2label
    labels = [id2label[_id] if _id in id2label else "O" for _id in pred_ids]
    return list(words), labels

def consolidate_outline(pages_label_spans):
    found = set()
    outline = []
    for entry in pages_label_spans:
        page_num, word_spans = entry['page'], entry['spans']
        for span in word_spans:
            label = span['label'].upper()
            if label in {"H1", "H2", "H3"}:
                txt = re.sub(r"\s+", " ", span["text"].replace("##", "")).strip()
                key = (label, txt, page_num)
                if txt and key not in found:
                    outline.append({"level": label, "text": txt, "page": page_num})
                    found.add(key)
    return outline

def extract_outline(pdf_path, processor, model, device):
    doc = fitz.open(pdf_path)
    title = ""
    pages_label_spans = []
    for page_num, page in enumerate(doc):
        words, labels = predict_page_structure(page, processor, model, device)
        if not words:
            continue
        word_spans = aggregate_label_spans(words, labels)
        pages_label_spans.append({'page': page_num, 'spans': word_spans})
    for entry in pages_label_spans:
        for span in entry['spans']:
            if span['label'].upper() == "TITLE" and not title:
                candidate = re.sub(r"\s+", " ", span["text"]).strip()
                if len(candidate) > 0:
                    title = candidate
                    break
        if title:
            break
    outline = consolidate_outline(pages_label_spans)
    return {"title": title, "outline": outline}

def process_pdfs(input_dir="input", output_dir="output"):
    os.makedirs(output_dir, exist_ok=True)
    pdfs = [f for f in os.listdir(input_dir) if f.lower().endswith(".pdf")]
    for fname in pdfs:
        in_path = os.path.join(input_dir, fname)
        out_path = os.path.join(output_dir, f"{os.path.splitext(fname)[0]}.json")
        result = extract_outline(in_path, processor, model, DEVICE)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        print(f"Processed {fname} -> {os.path.basename(out_path)}")

if __name__ == "__main__":
    process_pdfs("input", "output")
