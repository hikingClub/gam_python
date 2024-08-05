import torch
from flask_cors import CORS
from transformers import AutoModelForCausalLM, AutoTokenizer

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)
CORS(app)  # 모든 도메인에서의 요청을 허용

# 모델과 토크나이저 로드
model_name = "skt/kogpt2-base-v2"
model = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def autocomplete_search(query, max_length=7, num_return_sequences=5):
    # 입력 시퀀스를 토큰화
    input_ids = tokenizer.encode(query, return_tensors='pt')
    
    # 모델을 사용해 예측
    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_length=max_length,
            num_return_sequences=num_return_sequences,
            no_repeat_ngram_size=2,
            early_stopping=True,
            top_p=0.95,
            top_k=50,
            do_sample=True
        )
    
    # 예측된 토큰을 텍스트로 변환
    suggestions = [tokenizer.decode(output, skip_special_tokens=True).strip() for output in outputs]
    return suggestions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    data = request.get_json()
    query = data.get('query', '')
    suggestions = autocomplete_search(query)
    return jsonify(suggestions)

if __name__ == '__main__':
    app.run(debug=True)