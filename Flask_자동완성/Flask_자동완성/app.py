from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForMaskedLM, AutoTokenizer
import torch

app = Flask(__name__)

# 모델과 토크나이저 로드
model_name = "lots-o/ko-albert-large-v1"
model = AutoModelForMaskedLM.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def get_autocomplete_suggestions(prefix):
    inputs = tokenizer(prefix + " [MASK]", return_tensors="pt")
    with torch.no_grad():
        outputs = model(**inputs)

    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]
    mask_token_logits = outputs.logits[0, mask_token_index, :].squeeze()

    top_k = 5
    top_k_tokens = torch.topk(mask_token_logits, top_k, dim=-1).indices.tolist()
    suggestions = [tokenizer.decode([token]).strip() for token in top_k_tokens]
    
    return suggestions

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/autocomplete', methods=['GET'])
def autocomplete():
    query = request.args.get('query', '')
    base_suggestions = get_autocomplete_suggestions(query)
    full_suggestions = [query + suggestion for suggestion in base_suggestions]
    return jsonify(full_suggestions)

if __name__ == '__main__':
    app.run(debug=True)