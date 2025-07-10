import torch
import pandas as pd
from PIL import Image
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import BlipProcessor, BlipModel
from tqdm import tqdm
import os

# --- 설정 (A6000 최적화) ---
TRAIN_DATA_PATH = './data/train_augmented.csv'
PRETRAINED_PATH = './saved_model_pretrained'
FINAL_MODEL_PATH = './saved_model'
NUM_EPOCHS = 10
BATCH_SIZE = 32 # 👈 A6000에 맞게 배치 사이즈 증가
LEARNING_RATE = 2e-5

class BlipForMCQA(torch.nn.Module):
    def __init__(self, model_path):
        super().__init__()
        self.blip = BlipModel.from_pretrained(model_path)
        self.classifier = torch.nn.Linear(self.blip.config.text_config.hidden_size, 1)

    def forward(self, input_ids, attention_mask, pixel_values):
        outputs = self.blip(input_ids=input_ids, attention_mask=attention_mask, pixel_values=pixel_values)
        cls_output = outputs.pooler_output
        logits = self.classifier(cls_output)
        return logits

class MCQADataset(Dataset):
    def __init__(self, df, processor):
        self.df = df
        self.processor = processor
        self.processed_data = self._process_data()
        print(f"Created {len(self.processed_data)} samples for the final classification task.")

    def _process_data(self):
        processed = []
        for _, row in tqdm(self.df.iterrows(), desc="Formatting data for classification"):
            for choice in ['A', 'B', 'C', 'D']:
                text = f"Question: {row['Question']} Answer: {row[choice]}"
                label = 1.0 if choice == row['answer'] else 0.0
                processed.append({"img_path": row['img_path'], "text": text, "label": label})
        return processed

    def __len__(self):
        return len(self.processed_data)

    def __getitem__(self, idx):
        item = self.processed_data[idx]
        img_path = os.path.join('data', item['img_path'].lstrip('.\\/'))
        
        try:
            image = Image.open(img_path).convert("RGB")
            inputs = self.processor(images=image, text=item['text'], padding="max_length", truncation=True, return_tensors="pt")
            return {
                "input_ids": inputs.input_ids.squeeze(0), "attention_mask": inputs.attention_mask.squeeze(0),
                "pixel_values": inputs.pixel_values.squeeze(0), "labels": torch.tensor(item['label'], dtype=torch.float32)
            }
        except Exception:
            return None

def train_and_inference():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    print("--- 1. Starting Final Training (Classification Task) ---")
    processor = BlipProcessor.from_pretrained(PRETRAINED_PATH)
    model = BlipForMCQA(PRETRAINED_PATH).to(device)
    
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    train_dataset = MCQADataset(train_df, processor)
    
    def collate_fn(batch):
        batch = list(filter(lambda x: x is not None, batch))
        if not batch: return None
        return torch.utils.data.dataloader.default_collate(batch)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn, num_workers=4)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)
    loss_fn = torch.nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(NUM_EPOCHS):
        pbar = tqdm(train_dataloader, desc=f"Fine-tuning Epoch {epoch+1}/{NUM_EPOCHS}")
        for batch in pbar:
            if batch is None: continue
            batch = {k: v.to(device) for k, v in batch.items()}
            
            optimizer.zero_grad()
            logits = model(input_ids=batch['input_ids'], attention_mask=batch['attention_mask'], pixel_values=batch['pixel_values'])
            loss = loss_fn(logits.squeeze(-1), batch['labels'])
            loss.backward()
            optimizer.step()
            pbar.set_postfix(loss=loss.item())
            
    if not os.path.exists(FINAL_MODEL_PATH):
        os.makedirs(FINAL_MODEL_PATH)
        
    model.blip.save_pretrained(FINAL_MODEL_PATH)
    torch.save(model.classifier.state_dict(), os.path.join(FINAL_MODEL_PATH, "classifier.pth"))
    processor.save_pretrained(FINAL_MODEL_PATH)
    print(f"🎉 Final classifier model saved to {FINAL_MODEL_PATH}")

    print("\n--- 2. Starting Inference ---")
    del model
    torch.cuda.empty_cache()
    
    final_processor = BlipProcessor.from_pretrained(FINAL_MODEL_PATH)
    final_model = BlipForMCQA(FINAL_MODEL_PATH).to(device)
    final_model.classifier.load_state_dict(torch.load(os.path.join(FINAL_MODEL_PATH, "classifier.pth")))
    final_model.eval()

    test_df = pd.read_csv('./data/test.csv')
    submission = pd.read_csv('./data/sample_submission.csv')
    predictions = []

    with torch.no_grad():
        for _, row in tqdm(test_df.iterrows(), desc="Predicting final answers"):
            img_path = os.path.join('data', row['img_path'].lstrip('.\\/'))
            try:
                image = Image.open(img_path).convert("RGB")
            except FileNotFoundError:
                predictions.append('A')
                continue

            scores = []
            for choice in ['A', 'B', 'C', 'D']:
                text = f"Question: {row['Question']} Answer: {row[choice]}"
                inputs = final_processor(images=image, text=text, return_tensors="pt").to(device)
                logits = final_model(**inputs)
                scores.append(logits.item())
            
            predictions.append(['A', 'B', 'C', 'D'][np.argmax(scores)])
    
    submission['answer'] = predictions
    submission.to_csv('submission.csv', index=False)
    print("🎉 Final submission file 'submission.csv' created successfully!")

if __name__ == '__main__':
    train_and_inference()