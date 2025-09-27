#Build an LLM chatbot for news text classification and keyword extraction
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
import faiss
import pymupdf
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

#1.放文档 Add the news documents
files_for_classification     = ["bbc_text_cls.csv"]
files_for_keyword_extraction = ["NewsCategorizer.csv"]

#2.读取文档 Read the documents
def get_news_classification_from_files(files):
    news_classification_list = {}
    for file in files:
        try:
            if file.endswith(".csv"):   #read the csv files
                try:
                    df = pd.read_csv(file, encoding='UTF-8')  #如果不是utf-8或者latin1，打不开csv
                except UnicodeDecodeError:
                        df = pd.read_csv(file, encoding='latin1')
                df_length     = len(df['text'])
                news_texts    = df['text']
                news_labels   = df['label']
                for x in range(df_length):   #将新闻正文和标签塞进字典里
                    text      = news_texts[x]
                    label     = news_labels[x]
                    if label in news_classification_list.keys():
                        news_classification_list[label].append(text)
                    else:
                        news_classification_list[label]=[text]
    
            if file.endswith(".txt"):   #read the txt files
                try:
                    df = pd.read_csv(file, encoding='UTF-8')
                except UnicodeDecodeError:
                        df = pd.read_csv(file, encoding='latin1')
                df           = pd.read_csv(file)
                df_readlines = df.readlines()
                text         = df_readlines[1:]   #txt的第一行是标签“商业，娱乐”等，第二行开始是正文
                label        = df_readlines[0]
                if label in news_classification_list.keys():
                    news_classification_list[label].append(text) #将文档内容塞入list里面，用于后续向量化
                else:
                    news_classification_list[label]=[text]
                
        except (FileNotFoundError, pd.errors.EmptyDataError, pymupdf.FileDataError) as e:
            print(f"Error during processing {file}: {e}")
    return news_classification_list

def get_news_keywords_from_files(files):
    news_keyword_extraction_list = {}
    for file in files:
            if file.endswith(".csv"):
                try:
                    df = pd.read_csv(file, encoding='latin1')
                except UnicodeDecodeError:
                        df = pd.read_csv(file, encoding='UTF-8')
                df_length     = len(df['keywords'])
                news_keywords = df['keywords']
                news_texts    = df['short_description']
                for x in range(df_length):   #将新闻正文和标签塞进字典里
                    text      = news_texts[x]
                    keyword   = news_keywords[x]
                    news_keyword_extraction_list[keyword] = text   #将文档内容塞入list里面，用于后续向量化
    return news_keyword_extraction_list

#3.内容向量化 vectorize the contents
def convert_to_tfidf_vectors(vectorizer, docs):  # 将新闻列表向量化
    texts = []
    for doc_list in docs.values():
        if isinstance(doc_list, list):
            texts.extend(doc_list) #extend the list of strings to the texts list.
        else:
            texts.append(doc_list) #if the value is a string, then append.
    texts_vectors = vectorizer.fit_transform(texts).toarray()
    return texts, texts_vectors

def create_faiss_index(texts_vectors):    #创建一个《相似度搜索》的索引 create index
    dimension = texts_vectors.shape[1]
    index     = faiss.IndexFlatL2(dimension)
    index.add(texts_vectors.astype('float32'))
    return index

def search_documents(vectorizer, index, query, top_k=3):    #利用索引搜索文本 search texts via index
    if not query:
        print("Empty query received.")
        return None, None
    query_vector = vectorizer.transform([query]).toarray().astype('float32')
    distances, indices = index.search(query_vector, top_k)
    return distances, indices



def indices_to_documents(distances, indices, texts, original_dict):
    results = []
    for i in range(len(indices[0])):
        text = texts[indices[0][i]]
        distance = distances[0][i]
        similarity = 1 / (1 + distance)
        for key, value in original_dict.items():
            if isinstance(value, list) and text in value:
                label = key
                break
            elif isinstance(value, str) and text == value:
                label = key
                break
        results.append((label, text, similarity))
    return results

vectorizer = TfidfVectorizer(max_features=8000, min_df=5, max_df=0.5)    #看看有无必要调低max_features
news_classification_list = get_news_classification_from_files(files_for_classification)
class_texts = []
for doc_list in news_classification_list.values():
    if isinstance(doc_list, list):
        class_texts.extend(doc_list)
    else:
        class_texts.append(doc_list)
news_keywords_extraction = get_news_keywords_from_files(files_for_keyword_extraction)
keyword_texts = []

for text in news_keywords_extraction.values():
    keyword_texts.append(text)
all_texts = class_texts + keyword_texts
vectorizer.fit(all_texts) #train vectorizer with all texts.
news_classification_list_faiss_index = create_faiss_index(vectorizer.transform(class_texts).toarray())
news_keywords_extraction_faiss_index = create_faiss_index(vectorizer.transform(keyword_texts).toarray())


#让chatbot把用户要处理的新闻向量化，并在文档库里寻找相似的文本，生成标签；如果没有，那就让模型自己生成一个标签
#4.建立chatbot build chatbot
def Chatbot(model_name):
    chat_tokenizer = AutoTokenizer.from_pretrained(model_name)
    chat_model     = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16, device_map="auto")
    chat_streamer  = TextIteratorStreamer(chat_tokenizer, timeout=60.0, skip_prompt=True, skip_special_tokens=True)

    def retrieve(query):
        # 分类检索 search by sorting
        distances_classification, indices_classification = search_documents(vectorizer, news_classification_list_faiss_index, query, top_k=3)
        retrieved_knowledge_classification = indices_to_documents(distances_classification, indices_classification, class_texts, news_classification_list)
    
        # 关键词提取检索 extract keywords
        distances_keywords, indices_keywords = search_documents(vectorizer, news_keywords_extraction_faiss_index, query, top_k=3)
        retrieved_knowledge_keywords = indices_to_documents(distances_keywords, indices_keywords, keyword_texts, news_keywords_extraction)
    
        return retrieved_knowledge_classification, retrieved_knowledge_keywords

    def run():
        input_query = ""
        while input_query != "exit":
            input_query = input('Input a news for classification and keywords extraction: \n')
            if input_query != "exit":
                retrieved_knowledge_classification, retrieved_knowledge_keywords = retrieve(input_query)

                print('--------- Retrieved Classification Knowledge ---------')
                for label, text, similarity in retrieved_knowledge_classification:
                    print(f'{label} similarity: {similarity:.2f}')
            
                print('--------- Retrieved Keyword Extraction Knowledge ---------')
                for keyword, text, similarity in retrieved_knowledge_keywords:
                    print(f'{keyword} similarity: {similarity:.2f}')
            
                context_classification = '\n'.join([f' - (similarity: {similarity:.2f}) {label} \n\n {text}' for label, text, similarity in retrieved_knowledge_classification])
                context_keywords = '\n'.join([f' - (similarity: {similarity:.2f}) {keyword} \n\n {text}' for keyword, text, similarity in retrieved_knowledge_keywords])
            
                instruction_prompt = f"""
                                        You are a helpful chatbot for news classification and keywords extraction.
                                        
                                        Datasets for classification:
                                        {context_classification}
                                        
                                        Datasets for keyword extraction:
                                        {context_keywords}
                                        
                                        ---
                                        
                                        User's News:
                                        {input_query}
                                        
                                        ---
                                        Your mission is:
                                        1. News classification: Give the most similar label of the news that user inputs. The output format is 'News Label:____'
                                        2. News keyword extraction: Analyze the User's News above and extract the most relevant keywords. 
                                        These keywords MUST be specific to the User's News, not the datasets.
                                        The output format is 'News keywords:____'
                                        """
                messages = [{
                        "role": "system",
                        "content": instruction_prompt,
                    },{
                        "role": "user",
                        "content": input_query,
                    }]
                tokenized_chat = chat_tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt", return_dict=True, return_attention_mask=True)
                print(f"prompt size: {len(tokenized_chat['input_ids'][0])}")
                generation_kwargs = tokenized_chat | {
                    "pad_token_id":   chat_tokenizer.eos_token_id,
                    "max_new_tokens": 1000,
                    "temperature": 0.2, 
                    "top_k": 1, 
                    "top_p": 0.8, 
                    "streamer": chat_streamer}

                thread = Thread(target=chat_model.generate, kwargs=generation_kwargs)
                thread.start()
                for new_text in chat_streamer:
                    if new_text.endswith(chat_tokenizer.eos_token) and new_text != chat_tokenizer.eos_token:
                        new_text = new_text.replace(chat_tokenizer.eos_token, "")
                    if len(new_text) > 0:
                        print(new_text, end="", flush=True)
                thread.join()
                print('\n')

    run()


if __name__ == "__main__":
    model_name = "meta-llama/Llama-3.2-3B-Instruct" 
    from huggingface_hub import login
    login(token = '????????????????????????')  #input the valid tokens from hugging face
    chatbot = Chatbot(model_name)
    chatbot.run()
