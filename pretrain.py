import torch
from torch.utils.data import DataLoader
from torch.optim import AdamW
from transformers import BlipProcessor, BlipForQuestionAnswering
from datasets import load_dataset
from tqdm import tqdm
import os

# --- 설정 (A6000 최적화) ---
# 사전 학습 데이터 양을 10만 개로 대폭 늘려 성능 극대화
TRAIN_SAMPLES = 100_000 
MODEL_NAME = "Salesforce/blip-vqa-base"
PRETRAINED_MODEL_PATH = "./saved_model_pretrained"
BATCH_SIZE = 128 # 👈 A6000의 VRAM을 활용하여 배치 사이즈 대폭 증가!
NUM_EPOCHS = 1
LEARNING_RATE = 5e-5

class VQADataset(Dataset):
    def __init__(self, processor, num_samples):
        print("Loading VQA v2 dataset from Hugging Face... (This may take a significant amount of time and disk space)")
        self.dataset = load_dataset("HuggingFaceM4/VQAv2", split="train", streaming=True).take(num_samples)
        self.processor = processor
        print(f"Preparing {num_samples} samples for pre-training. This is a one-time process.")
        self.dataset_iterator = list(tqdm(self.dataset, total=num_samples, desc="Pre-caching dataset samples"))

    def __len__(self):
        return len(self.dataset_iterator)

    def __getitem__(self, idx):
        item = self.dataset_iterator[idx]
        question = item['question']
        answer = max(item['answers'], key=lambda x: x['answer_confidence'])['answer']
        image = item['image']
        
        try:
            inputs = self.processor(images=image, text=question, padding="max_length", truncation=True, return_tensors="pt")
            labels = self.processor.tokenizer(text=answer, padding="max_length", truncation=True, max_length=32, return_tensors="pt")
            inputs['labels'] = labels.input_ids
            return {k: v.squeeze(0) for k, v in inputs.items()}
        except Exception as e:
            print(f"Skipping item {idx} due to error: {e}")
            return None

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device for pre-training: {device}")

    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)

    dataset = VQADataset(processor, TRAIN_SAMPLES)
    
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch: return None
        return torch.utils.data.dataloader.default_collate(batch)

    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print("✅ Pre-training setup complete. Starting...")
    model.train()
    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(dataloader, desc=f"Pre-training Epoch {epoch+1}")
        for batch in pbar:
            if batch is None: continue
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            outputs = model(**batch)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())

    if not os.path.exists(PRETRAINED_MODEL_PATH):
        os.makedirs(PRETRAINED_MODEL_PATH)
        
    model.save_pretrained(PRETRAINED_MODEL_PATH)
    processor.save_pretrained(PRETRAINED_MODEL_PATH)
    print(f"🎉 Pre-trained model successfully saved to {PRETRAINED_MODEL_PATH}")

if __name__ == '__main__':
    main()