import datetime
import json
import pandas as pd
import requests
import datetime
from datetime import timedelta
import json
from flask_cors import CORS
from sklearn.metrics.pairwise import cosine_similarity
from sqlalchemy import create_engine
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer
import torch
from flask import Flask, jsonify, render_template, request

today = datetime.datetime.now().strftime("%Y-%m-%d")
startDate = (datetime.datetime.now()- 2*timedelta(365)).strftime("%Y-%m-%d")

app = Flask(__name__)
CORS(app)

#--------------------------------------추천 검색-------------------------------------------------------
# MySQL 데이터베이스 연결 정보 설정
engine = create_engine("mysql+pymysql://root:mysql123@mysql1.cr2ool1q9cto.ap-northeast-2.rds.amazonaws.com/hikingclub")

url = "https://metalink.k-knowledge.kr/search/openapi/search"
api_key = "bvazg6wcmctebosw3ho20bf9lpk7mpl32hcs8sbut4xsgu2zrie57dgu3pqy83ou"

headers = {
    "Content-Type": "application/json",
    "api_key": api_key
}

def get_searchapi(keyword, pageNum, pagePer):
    params = {
        "searchKeyword": keyword,
        "startDate": startDate,
        "endDate": today,
        "pageNum": pageNum,
        "pagePer": pagePer
    }
    response = requests.post(url, headers=headers, data=json.dumps(params))

    # 응답 처리
    if response.status_code == 200:
        data = response.json()
        pretty_json = json.dumps(data, indent=4, ensure_ascii=False) # pretty print
        return pd.DataFrame(data['result'])
    else:
        print("데이터 조회 실패:", response.status_code, response.text)

# 데이터 프레임 로드
def load_data():
    # 데이터베이스 연결
    with engine.connect() as connection:
        # SQL 쿼리 실행 및 데이터프레임으로 변환
        search_df = pd.read_sql("SELECT * FROM SEARCHHISTORY", connection)
        view_df = pd.read_sql("SELECT * FROM VIEWHISTORY", connection)
        favorite_df = pd.read_sql("SELECT * FROM EMPATHY", connection)
        profile_df = pd.read_sql("SELECT * FROM MEMBER", connection)
        recommendedfield_df = pd.read_sql("SELECT * FROM RECOMMENDEDFIELD", connection)
    return search_df, view_df, favorite_df, profile_df, recommendedfield_df

def get_recommend(recIndexes, recommendedfield_df):
    if recIndexes == None:
        return []
    else:
        recIndex_list = [int(recIndex) for recIndex in recIndexes.split(',') if recIndex]
        category = [recommendedfield_df[recommendedfield_df['recIndex'] == idx]['category1'].values[0] + ' > ' + recommendedfield_df[recommendedfield_df['recIndex'] == idx]['category2'].values[0] for idx in recIndex_list]
        return category
    
def get_recommendSearch(user_id, title_df, profile_df, recommendedfield_df):
    profile_df['recIndexes'] = profile_df['recIndexes'].apply(lambda x : get_recommend(x, recommendedfield_df))
    title_df['category_set'] = title_df['map_path'].apply(set)
    reindexes = profile_df[profile_df['seq']==user_id]['recIndexes'].values[0]
    interest = profile_df[profile_df['seq']==user_id]['interest'].values[0].replace(' ', '').split(',')

    filtered_df = title_df[title_df['type_name'].isin(interest)]
    filtered_df = filtered_df[filtered_df['category_set'].apply(lambda x: not x.isdisjoint(set(reindexes)))]
    return filtered_df


with engine.connect() as connection:
    user_agejob_df = pd.read_sql("SELECT * FROM embedding_table", connection)

embedding_loaded_list = [json.loads(embedding) for embedding in user_agejob_df['embedding']]
user_agejob_df['embedding'] = [torch.tensor(i) for i in embedding_loaded_list]

tokenizer1 = AutoTokenizer.from_pretrained("snunlp/KR-SBERT-V40K-klueNLI-augSTS")
model1 = AutoModel.from_pretrained("snunlp/KR-SBERT-V40K-klueNLI-augSTS")

def get_embeddings(text):
    inputs = tokenizer1(text, return_tensors="pt", truncation=True, padding=True)
    with torch.no_grad():
        outputs = model1(**inputs)
    embeddings = outputs.last_hidden_state
    sentence_embedding = embeddings.mean(dim=1).squeeze()
    return sentence_embedding

def get_user_profile_embeddings(user_id, search_df, view_df, favorite_df, profile_df):
    search_df['embedding'] = search_df['keyword'].apply(get_embeddings)
    view_df['embedding'] = view_df['viewTitle'].apply(get_embeddings)
    favorite_df['embedding'] = favorite_df['favoriteTitle'].apply(get_embeddings)

    search_embeddings_list = search_df[search_df['seq'] == user_id]['embedding'].tolist()
    view_embeddings_list = view_df[view_df['seq'] == user_id]['embedding'].tolist()
    favorite_embeddings_list = favorite_df[favorite_df['seq'] == user_id]['embedding'].tolist()
    age = profile_df[profile_df['seq']== user_id]['ageRange'].values[0]
    job = profile_df[profile_df['seq'] == user_id]['jobRange'].values[0]
    user_agejob_embeddings_list = user_agejob_df[(user_agejob_df['ageRange'] == age) & (user_agejob_df['jobRange'] == job)]['embedding'].tolist()

    search_embeddings = torch.stack(search_embeddings_list) if search_embeddings_list else torch.zeros(1, model1.config.hidden_size)
    view_embeddings = torch.stack(view_embeddings_list) if view_embeddings_list else torch.zeros(1, model1.config.hidden_size)
    favorite_embeddings = torch.stack(favorite_embeddings_list) if favorite_embeddings_list else torch.zeros(1, model1.config.hidden_size)
    user_agejob_embeddings = torch.stack(user_agejob_embeddings_list) if user_agejob_embeddings_list else torch.zeros(1, model1.config.hidden_size)
    
    user_profile_embedding  = search_embeddings.mean(dim=0) * 0.2 + view_embeddings.mean(dim=0) * 0.2 + favorite_embeddings.mean(dim=0) * 0.4 + user_agejob_embeddings.mean(dim=0) * 0.2
    
    return user_profile_embedding

def calculate_similarity(embedding1, embedding2):
    return cosine_similarity(embedding1.unsqueeze(0), embedding2.unsqueeze(0)).item()

def get_similarities(user_id, search_df, view_df, favorite_df, profile_df, titles_embedding):
    user_profile_embedding = get_user_profile_embeddings(user_id, search_df, view_df, favorite_df,  profile_df)
    return [calculate_similarity(user_profile_embedding, title_embedding) for title_embedding in titles_embedding]

def get_searchresult(user_id, keyword, pageNum, pagePer):
    title_df = get_searchapi(keyword, pageNum, pagePer)
    search_df, view_df, favorite_df, profile_df, recommendedfield_df = load_data()

    df = get_recommendSearch(user_id, title_df, profile_df, recommendedfield_df)
    # 각 문서와의 유사도 계산
    titles_embedding = [get_embeddings(title) for title in df['title'].values]
    df['similarities'] = get_similarities(user_id, search_df, view_df, favorite_df, profile_df, titles_embedding)
    re_df = df.sort_values('similarities', ascending=False).drop(['category_set', 'similarities'], axis=1)
    return re_df

@app.route('/recommend_search', methods=['POST'])
def recommend_search():
    try:
        data = request.get_json()
        user_id = data['user_id']
        keyword = data['keyword']
        pageNum = data.get('pageNum', 1)
        pagePer = data.get('pagePer', 10)
        
        result_df = get_searchresult(user_id, keyword, pageNum, pagePer)
        result = result_df.to_dict(orient='records')
        
        return jsonify({"status": "success", "result": result})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})

# -------------------------------------------검색어 자동완성----------------------------------------------
# 모델과 토크나이저 로드
model_name = "skt/kogpt2-base-v2"
model2 = AutoModelForCausalLM.from_pretrained(model_name)
tokenizer2 = AutoTokenizer.from_pretrained(model_name)

def autocomplete_search(query, max_length=7, num_return_sequences=5):
    # 입력 시퀀스를 토큰화
    input_ids = tokenizer2.encode(query, return_tensors='pt')
    
    # 모델을 사용해 예측
    with torch.no_grad():
        outputs = model2.generate(
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
    suggestions = [tokenizer2.decode(output, skip_special_tokens=True).strip() for output in outputs]
    return suggestions

#@app.route('/')
#def home():
#    return render_template('index.html')

@app.route('/autocomplete', methods=['POST'])
def autocomplete():
    data = request.get_json()
    query = data.get('query', '')
    suggestions = autocomplete_search(query)
    return jsonify(suggestions)

#-------------------------------------------------------------금일 증분수정데이터-------------------------------------------------------
# 데이터베이스 연결
with engine.connect() as connection:
    # SQL 쿼리 실행 및 데이터프레임으로 변환
    meta_df = pd.read_sql("SELECT * FROM METADATA", connection)

@app.route('/get_data', methods=['POST'])
def get_data():
    return jsonify(meta_df.to_dict(orient='records'))

if __name__ == '__main__':
    app.run(debug=True)