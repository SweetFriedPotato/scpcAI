import torch
import pandas as pd
from PIL import Image
from transformers import BlipProcessor, BlipForQuestionAnswering
from tqdm import tqdm
import os

# --- 설정 ---
TEST_DATA_PATH = './data/test.csv'
SUBMISSION_PATH = './data/sample_submission.csv'
SAVED_MODEL_PATH = './saved_model' # 학습된 모델이 저장된 경로
OUTPUT_FILENAME = 'submission.csv' # 최종 제출 파일명

# --- 메인 추론 함수 ---
def main():
    # 장치 설정
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🚀 Using device: {device}")

    # 파인튜닝된 모델과 전처리기 로드
    processor = BlipProcessor.from_pretrained(SAVED_MODEL_PATH)
    model = BlipForQuestionAnswering.from_pretrained(SAVED_MODEL_PATH).to(device)
    model.eval() # 모델을 평가 모드로 설정
    
    print("✅ Model loaded successfully.")

    # 테스트 데이터 로드
    test_df = pd.read_csv(TEST_DATA_PATH)
    predictions = []
    
    # 테스트 데이터에 대해 추론 실행
    for _, row in tqdm(test_df.iterrows(), total=len(test_df), desc="🧠 Predicting answers"):
        img_path = row['img_path']
        question = row['Question']
        
        try:
            raw_image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: File not found {img_path}. Predicting 'A' as default.")
            predictions.append('A')
            continue
            
        # 이미지와 질문을 모델 입력 형식으로 변환
        inputs = processor(raw_image, question, return_tensors="pt").to(device)

        # 모델을 통해 답변 생성
        with torch.no_grad(): # 그래디언트 계산 비활성화
            out = model.generate(**inputs, max_new_tokens=10) # 답변 생성 최대 길이 설정
        
        predicted_text = processor.decode(out[0], skip_special_tokens=True).strip().lower()

        # 생성된 답변과 선택지를 비교하여 가장 유사한 답안 선택
        choices = {
            "A": row['A'].lower(),
            "B": row['B'].lower(),
            "C": row['C'].lower(),
            "D": row['D'].lower()
        }

        # 가장 유사한 선택지를 찾는 간단한 방법 (겹치는 단어 수 기준)
        final_answer = 'A' # 기본값
        max_overlap = -1
        
        for letter, text in choices.items():
            overlap = len(set(predicted_text.split()) & set(text.split()))
            if overlap > max_overlap:
                max_overlap = overlap
                final_answer = letter

        predictions.append(final_answer)

    # 제출 파일 생성
    submission_df = pd.read_csv(SUBMISSION_PATH)
    submission_df['answer'] = predictions
    submission_df.to_csv(OUTPUT_FILENAME, index=False)
    
    print(f"\n🎉 Submission file '{OUTPUT_FILENAME}' created successfully!")
    print("Sample of the submission file:")
    print(submission_df.head())

if __name__ == '__main__':
    main()