import torch
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from transformers import BlipProcessor, BlipForQuestionAnswering, AdamW
from tqdm import tqdm
import os

# --- 설정 ---
TRAIN_DATA_PATH = './data/train.csv'
SAVED_MODEL_PATH = './saved_model'
MODEL_NAME = 'Salesforce/blip-vqa-large'
NUM_EPOCHS = 5  # 학습 횟수 (에폭)
BATCH_SIZE = 4  # 한 번에 처리할 데이터 수
LEARNING_RATE = 5e-6 # 학습률

# --- 데이터셋 클래스 정의 ---
class VQADataset(Dataset):
    def __init__(self, df, processor, img_base_path):
        self.df = df
        self.processor = processor
        self.img_base_path = img_base_path

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_base_path, os.path.basename(row['img_path']))
        question = row['Question']
        
        # 정답 텍스트 생성 (A, B, C, D 중 정답에 해당하는 내용)
        answer_letter = row['answer']
        answer_text = row[answer_letter]

        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: File not found {img_path}, skipping.")
            return None

        # VQA 모델 학습을 위해 이미지, 질문, 답변을 모두 인코딩
        encoding = self.processor(image, question, text=answer_text, padding="max_length", truncation=True, return_tensors="pt")
        
        # 불필요한 차원 제거
        encoding = {k: v.squeeze() for k, v in encoding.items()}
        return encoding

# --- 메인 학습 함수 ---
def main():
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 데이터 로드 및 전처리기 준비
    train_df = pd.read_csv(TRAIN_DATA_PATH)
    processor = BlipProcessor.from_pretrained(MODEL_NAME)
    
    # 데이터셋 및 데이터로더 생성
    # train_input_images 폴더 경로를 정확히 지정해야 합니다.
    img_base_path = os.path.dirname(train_df['img_path'][0])
    train_dataset = VQADataset(train_df, processor, img_base_path)
    
    # None 값을 필터링하기 위한 collate_fn
    def collate_fn(batch):
        batch = [item for item in batch if item is not None]
        if not batch:
            return None, None
        return torch.utils.data.dataloader.default_collate(batch)

    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=collate_fn)

    # 모델 로드 및 최적화 설정
    model = BlipForQuestionAnswering.from_pretrained(MODEL_NAME).to(device)
    optimizer = AdamW(model.parameters(), lr=LEARNING_RATE)

    print("✅ Setup complete. Starting training...")

    # 학습 루프
    model.train()
    for epoch in range(NUM_EPOCHS):
        print(f"\n--- Epoch {epoch+1}/{NUM_EPOCHS} ---")
        total_loss = 0
        
        for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch+1}"):
            if batch is None:
                continue

            input_ids = batch['input_ids'].to(device)
            pixel_values = batch['pixel_values'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)
            
            optimizer.zero_grad()
            
            # 모델 출력 및 손실 계산
            outputs = model(input_ids=input_ids,
                              pixel_values=pixel_values,
                              attention_mask=attention_mask,
                              labels=labels)
            
            loss = outputs.loss
            
            # 역전파 및 가중치 업데이트
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()

        avg_loss = total_loss / len(train_dataloader)
        print(f"Epoch {epoch+1} Average Loss: {avg_loss:.4f}")

    # 학습된 모델 저장
    if not os.path.exists(SAVED_MODEL_PATH):
        os.makedirs(SAVED_MODEL_PATH)
        
    model.save_pretrained(SAVED_MODEL_PATH)
    processor.save_pretrained(SAVED_MODEL_PATH)
    print(f"🎉 Model successfully saved to {SAVED_MODEL_PATH}")

if __name__ == '__main__':
    main()