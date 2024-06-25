from flask import Flask, request, jsonify
from kobert_tokenizer import KoBERTTokenizer
from transformers import BertModel

app = Flask(__name__)

# 토크나이저 및 모델 로드
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
model = BertModel.from_pretrained('skt/kobert-base-v1')

@app.route('/tokenize', methods=['POST'])
def tokenize():
    data = request.json
    text = data['text']
    tokens = tokenizer.tokenize(text)
    token_ids = tokenizer.convert_tokens_to_ids(tokens)
    return jsonify({'tokens': tokens, 'token_ids': token_ids})

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data['text']
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    outputs = model(**inputs)
    return jsonify({'outputs': outputs[0].detach().cpu().numpy().tolist()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5001)
