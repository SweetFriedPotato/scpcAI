import torch
import pandas as pd
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import re
import os

# --- 설정 ---
MODEL_ID = "meta-llama/Meta-Llama-3-8B-Instruct"

def parse_generated_text(text):
    """Llama-3 모델이 생성한 텍스트를 파싱하여 딕셔너리로 반환합니다."""
    try:
        q_match = re.search(r"New Question:\s*(.*)", text)
        a_match = re.search(r"New A:\s*(.*)", text)
        b_match = re.search(r"New B:\s*(.*)", text)
        c_match = re.search(r"New C:\s*(.*)", text)
        d_match = re.search(r"New D:\s*(.*)", text)
        ans_match = re.search(r"New Answer:\s*([A-D])", text)

        if all([q_match, a_match, b_match, c_match, d_match, ans_match]):
            return {
                "Question": q_match.group(1).strip(), "A": a_match.group(1).strip(),
                "B": b_match.group(1).strip(), "C": c_match.group(1).strip(),
                "D": d_match.group(1).strip(), "answer": ans_match.group(1).strip().upper(),
            }
        return None
    except Exception:
        return None

def augment_data(original_df, model, tokenizer, num_new_examples=4):
    augmented_rows = []
    
    for _, row in tqdm(original_df.iterrows(), total=len(original_df), desc="🤖 Augmenting data with Full Llama-3"):
        augmented_rows.append(row.to_dict())

        prompt_template = """<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are a data augmentation assistant. Based on the provided example, create ONE new, diverse, and plausible multiple-choice question.
The new question must be different but related to the original.
Your output format must be *exactly* as follows, without any additional text, comments, or explanations.

New Question: [The new question text]
New A: [Text for new option A]
New B: [Text for new option B]
New C: [Text for new option C]
New D: [Text for new option D]
New Answer: [A single letter: A, B, C, or D]<|eot_id|><|start_header_id|>user<|end_header_id|>

Original Question: {question}
Original Choices: A: {choice_a}, B: {choice_b}, C: {choice_c}, D: {choice_d}
Correct Answer Text: {answer_text}<|eot_id|><|start_header_id|>assistant<|end_header_id|>
"""
        
        for i in range(num_new_examples):
            try:
                prompt = prompt_template.format(
                    question=row['Question'], choice_a=row['A'], choice_b=row['B'],
                    choice_c=row['C'], choice_d=row['D'], answer_text=row[row['answer']]
                )
                
                inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
                outputs = model.generate(**inputs, max_new_tokens=256, eos_token_id=tokenizer.eos_token_id, do_sample=True, temperature=0.6, top_p=0.9)
                response_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
                
                assistant_response = response_text.split('<|start_header_id|>assistant<|end_header_id|>')[-1].strip()
                
                parsed_example = parse_generated_text(assistant_response)

                if parsed_example:
                    new_row = {"ID": f"{row['ID']}_aug_{i}", "img_path": row["img_path"], **parsed_example}
                    augmented_rows.append(new_row)
                else:
                    print(f"\nSkipping due to parsing failure for ID {row['ID']}_aug_{i}")

            except Exception as e:
                print(f"\nSkipping due to generation error for ID {row['ID']}_aug_{i}. Error: {e}")
                continue
            
    return pd.DataFrame(augmented_rows)

if __name__ == "__main__":
    output_path = "./data/train_augmented.csv"
    if os.path.exists(output_path):
        os.remove(output_path)
        print(f"Removed old augmented file: {output_path}")

    print("🚀 Loading Llama-3 model locally (Full Precision for A6000)...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    # A6000의 VRAM을 활용하여 양자화 없이 모델의 최대 성능으로 로드
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16, # bfloat16은 A6000에서 지원하는 고성능 데이터 타입
        device_map="auto",
    )
    print("✅ Llama-3 model loaded successfully.")
    
    train_df = pd.read_csv("./data/train.csv")
    # 원본 1개당 4개의 새로운 데이터 생성 -> 총 300개 데이터 확보
    augmented_df = augment_data(train_df, model, tokenizer, num_new_examples=4)
    
    augmented_df.to_csv(output_path, index=False, encoding='utf-8-sig')
    
    print(f"\n🎉 Augmentation complete!")
    print(f"Original samples: {len(train_df)}")
    print(f"Total samples (original + augmented): {len(augmented_df)}")
    print(f"Saved to {output_path}")