import os
import fitz  # PyMuPDF
import json

# Updated to include H4 support
label2id = {"O": 0, "Title": 1, "H1": 2, "H2": 3, "H3": 4, "H4": 5}

def make_label_dict(json_file):
    with open(json_file, "r", encoding="utf-8") as f:
        d = json.load(f)
    label_dict = {}
    
    # Debug: Print what we found in the JSON
    print(f"Processing {json_file}:")
    if d.get("title"):
        print(f"  Found title: '{d['title']}'")
        label_dict[(0, d["title"].strip())] = "Title"
        # Also try variations of the title
        title_words = d["title"].strip().split()
        for word in title_words:
            if len(word) > 3:  # Skip short words
                label_dict[(0, word)] = "Title"
    
    for o in d.get("outline", []):
        print(f"  Found {o['level']}: '{o['text']}' on page {o['page']}")
        label_dict[(o["page"], o["text"].strip())] = o["level"]
        # Also try individual words for better matching
        outline_words = o["text"].strip().split()
        for word in outline_words:
            if len(word) > 3:  # Skip short words
                label_dict[(o["page"], word)] = o["level"]
    
    print(f"  Total label entries: {len(label_dict)}")
    return label_dict

def pdf_to_examples(pdf_file, label_dict):
    doc = fitz.open(pdf_file)
    examples = []
    # Updated to include H4 in found_labels
    found_labels = {"Title": 0, "H1": 0, "H2": 0, "H3": 0, "H4": 0, "O": 0}
    
    for i, page in enumerate(doc):
        w, h = page.rect.width, page.rect.height
        for word_tuple in page.get_text("words"):
            x0, y0, x1, y1, text = word_tuple[:5]
            text = text.strip()
            if not text:
                continue
            
            box = [
                int(1000 * x0 / w), int(1000 * y0 / h),
                int(1000 * x1 / w), int(1000 * y1 / h)
            ]
            
            # Try exact match first
            label = label_dict.get((i, text), "O")
            
            # If no exact match, try partial matching
            if label == "O":
                for (page_num, label_text), label_val in label_dict.items():
                    if page_num == i and (text.lower() in label_text.lower() or label_text.lower() in text.lower()):
                        label = label_val
                        print(f"    Partial match: '{text}' -> '{label_text}' ({label})")
                        break
            
            found_labels[label] += 1
            examples.append({
                "words": [text],
                "boxes": [box],
                "labels": [label2id[label]]
            })
    
    print(f"  Label distribution: {found_labels}")
    return examples

def get_training_examples(pdf_dir, json_dir):
    data = []
    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        json_file = os.path.join(json_dir, fname.replace(".pdf", ".json"))
        pdf_file = os.path.join(pdf_dir, fname)
        if not os.path.exists(json_file):
            print(f"Warning: No JSON found for {fname}")
            continue
        
        print(f"\n=== Processing {fname} ===")
        label_dict = make_label_dict(json_file)
        if not label_dict:
            print(f"Warning: No labels found in {json_file}")
            continue
        
        examples = pdf_to_examples(pdf_file, label_dict)
        data.extend(examples)
    
    return data

if __name__ == "__main__":
    train_examples = get_training_examples("input", "labels")
    
    # Updated label counts to include H4
    label_counts = {"O": 0, "Title": 0, "H1": 0, "H2": 0, "H3": 0, "H4": 0}
    for ex in train_examples:
        label_name = [k for k, v in label2id.items() if v == ex["labels"][0]][0]
        label_counts[label_name] += 1
    
    print(f"\n=== Final Label Distribution ===")
    for label, count in label_counts.items():
        print(f"{label}: {count}")
    
    # Filter out bad examples
    filtered_examples = [ex for ex in train_examples if "words" in ex and ex["words"] and isinstance(ex["words"], list)]
    
    with open("train_examples.json", "w", encoding="utf-8") as f:
        json.dump(filtered_examples, f, ensure_ascii=False)
    
    print(f"\nSaved {len(filtered_examples)} training examples to train_examples.json")
    
    # Check if we have any non-O labels
    non_o_count = sum(1 for ex in filtered_examples if ex["labels"][0] != 0)
    if non_o_count == 0:
        print("\n❌ WARNING: No Title/H1/H2/H3/H4 labels found!")
        print("Check your ground-truth JSON files and ensure text matches PDF content.")
    else:
        print(f"✅ Found {non_o_count} non-O labels for training.")
