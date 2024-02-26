from transformers import pipeline
import pandas as pd

def test_huggingface_pipeline_use():

    text = """Dear Amazon, last week I ordered an Optimus Prime action figure \
    fromyour online store in Germany. Unfortunately, when I pened the package, \
    I discovered to my horror that I had been sent an action figure of Megatron\
    instead! As a lifelong enemy of the Decepticons, I hope you can understand my \
    dilemma. To resolve the issue, I emand an exchange of Megatron for the \
    Optimus Prime figure I ordered. Enclosed are copies of my records concerning \
    this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""

    print("\n## text-classification")
    classifier = pipeline("text-classification")
    outputs = classifier(text)
    print(" classification result: ", pd.DataFrame(outputs))

    print("\n## ner");
    ner_tagger = pipeline("ner", aggregation_strategy="simple")
    outputs = ner_tagger(text)
    print(pd.DataFrame(outputs))

    print("\n## question-answering")
    reader = pipeline("question-answering")
    question = "What does the customer want?"
    outputs = reader(question=question, context=text)
    print(" question: ", question)
    print(" answer  : ", pd.DataFrame([outputs]))

    print("\n## summarization")
    summarizer = pipeline("summarization")
    outputs = summarizer(text, max_length=56, clean_up_tokenization_spaces=True)
    print(" summarization : ",outputs[0]['summary_text'])

    print('\n## translation_en_to_de')
    translator = pipeline('translation_en_to_de', model="Helsinki-NLP/opus-mt-en-de")
    outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
    print(' Deutsch translation: ' , outputs[0]['translation_text'])

def test_translate_kr2en():

    translastor = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")

    try:
        print('## Korean to English Translation')
        while True:
            kr_text = input("> ")
            outputs = translastor(kr_text)
            print(' [English translated]: ', outputs[0]['translation_text'])
    except KeyboardInterrupt:
        print(" .. Quit")

from datasets import list_datasets
from datasets import load_dataset

'''
Trouble shooting for datasets.
 * Error message: NotImplementedError: Loading a dataset cached in a LocalFileSystem is not supported.
   - Solution: pip install -U datasets  //conda 환경에서도 pip install을 쓸 수 있나봐..
  
'''

def test_datasets():
    all_datasets = list_datasets()
    print('number of data set', len(all_datasets))
    print('first 10 datas ets:', all_datasets[:10])

    emotions = load_dataset("emotion")
    print(emotions)


import torch
import numpy as np
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from transformers import Trainer, TrainingArguments
from sklearn.metrics import accuracy_score, f1_score
from transformers import Trainer
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch.nn.functional import cross_entropy

#torch.cuda.set_device('cpu')

'''
hugging face에서 distilbert-base-uncased의 checkpoint를 내려 받고,
emotion 데이터 넷을 다시 내려 받아, 학습 시켜 미세 튜닝을 수행한다.
'''
def test_model_tuning():
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    print("cuda is available" if torch.cuda.is_available() else "cpu is available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    num_labels = 6
    model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels=num_labels)
         .to(device))

    # 허깅페이스 허브에서 데이터 셋을 다운로드 해서, 미세 튜닝의 학습 데이터로 사용한다. 
    # 허브에 약 5K개의 데이터 넷이 있는데 이중에서 emotion 데이터셋을 로딩한다.
    emotions = load_dataset("emotion")

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

    # GPU 메모리가 부족한 경우, batch_size를 줄여 본다.

    #batch_size = 64
    batch_size = 32

    logging_steps = len(emotions_encoded["train"])
    model_name = f"{model_ckpt}-finetuned-emotion"
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=2,
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    weight_decay=0.01,
                                    evaluation_strategy="epoch",
                                    disable_tqdm=False,
                                    logging_steps=logging_steps,
                                    push_to_hub=True,
                                    save_strategy="epoch",
                                    load_best_model_at_end=True,
                                    log_level="error")

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    '''
    Trainer를 쓰기위해 hugging face access token이 필요하다.
    https://huggingface.co/settings/tokens 에서 access token을 할당 받고,
    `huggingface-cli login` 명령을 이용하여 acces token을 입력하자.
    '''
    trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)

    trainer.train()

    def extra_hidden_states(batch):
        inputs = {k:v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}

        with torch.no_grad():
            last_hidden_state = model(**inputs).last_hidden_state
            return {"hidden_state": last_hidden_state[:, 0].cpu().numpy()}

    def plot_confusion_matrix(y_preds, y_true, labels):
        cm = confusion_matrix(y_true, y_preds, normalize="true")
        fig, ax = plt.subplots(figsize = (6, 6))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        disp.plot(cmap="Blues", values_format=".2f", ax = ax, colorbar=False)
        plt.title("Normalized confusion matrix")
        plt.show()

    preds_output = trainer.predict(emotions_encoded["validation"])
    print('preds_output.metrics: ', preds_output.metrics)
    y_preds = np.argmax(preds_output.predictions, axis = 1)
    print('y_preds: ', y_preds)

    def forward_pass_with_label(batch):
        inputs = {k:v.to(device) for k, v in batch.items() if k in tokenizer.model_input_names}

        with torch.no_grad():
            output = model(**inputs)
            pred_label = torch.argmax(output.logits, axis=-1)
            loss = cross_entropy(output.logits, batch["label"].to(device), reduction="none")
        return {"loss": loss.cpu().numpy(), "predicted_label": pred_label.cpu().numpy()}

    emotions_encoded.set_format("torch", columns=["input_ids", "attention_mask", "label"])
    emotions_encoded["validation"] = emotions_encoded["validation"].map(forward_pass_with_label, batched=True, batch_size=16)

    def label_int2str(row):
        return emotions["train"].features["label"].int2str(row)

    emotions_encoded.set_format("pandas")
    cols = ["text", "label", "predicted_label", "loss"]
    df_test = emotions_encoded["validation"][:][cols]
    df_test["label"] = df_test["label"].apply(label_int2str)
    df_test["predicted_label"] = (df_test["predicted_label"].apply(label_int2str))

    print(df_test.sort_values("loss", ascending=False).head(10))
    print(df_test.sort_values("loss", ascending=True).head(10))

    # 학습된 내용은 다시 hub에 commit 한다.
    #trainer.push_to_hub(commit_message="Training completed")

import pandas as pd
import matplotlib.pyplot as plt

def test_use_tuned_model():

    model_id = "jaegon-kim/distilbert-base-uncased-finetuned-emotion"
    classifier = pipeline("text-classification", model=model_id)

    #custom_tweet = "I saw a movie today and it was really good."
    custom_tweet = "OpenAI will now let you create videos from verbal cues"
    preds = classifier(custom_tweet, return_all_scores=True)
    print(preds)

    #from datasets import load_dataset
    #emotions = load_dataset("emotion")
    #labels = emotions["train"].features["label"].names
    labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

    preds_df = pd.DataFrame(preds[0])
    plt.bar(labels, 100 * preds_df["score"], color='C0')
    plt.title(f'"{custom_tweet}"')
    plt.ylabel("Class probability (%)")
    plt.show()



def test_model_tuning_local_store(local_model_path):
    model_ckpt = "distilbert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_ckpt)

    print("cuda is available" if torch.cuda.is_available() else "cpu is available")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.empty_cache()

    num_labels = 6
    model = (AutoModelForSequenceClassification
         .from_pretrained(model_ckpt, num_labels=num_labels)
         .to(device))

    # 허깅페이스 허브에서 데이터 셋을 다운로드 해서, 미세 튜닝의 학습 데이터로 사용한다. 
    # 허브에 약 5K개의 데이터 넷이 있는데 이중에서 emotion 데이터셋을 로딩한다.
    emotions = load_dataset("emotion")

    def tokenize(batch):
        return tokenizer(batch["text"], padding=True, truncation=True)

    emotions_encoded = emotions.map(tokenize, batched=True, batch_size=None)

    # GPU 메모리가 부족한 경우, batch_size를 줄여 본다.

    #batch_size = 64
    batch_size = 32

    logging_steps = len(emotions_encoded["train"])
    model_name = f"{model_ckpt}-finetuned-emotion"
    training_args = TrainingArguments(output_dir=model_name,
                                    num_train_epochs=1,
                                    learning_rate=2e-5,
                                    per_device_train_batch_size=batch_size,
                                    per_device_eval_batch_size=batch_size,
                                    weight_decay=0.01,
                                    evaluation_strategy="epoch",
                                    disable_tqdm=False,
                                    logging_steps=logging_steps,
                                    push_to_hub=True,
                                    save_strategy="epoch",
                                    load_best_model_at_end=True,
                                    log_level="error")

    def compute_metrics(pred):
        labels = pred.label_ids
        preds = pred.predictions.argmax(-1)
        f1 = f1_score(labels, preds, average="weighted")
        acc = accuracy_score(labels, preds)
        return {"accuracy": acc, "f1": f1}

    '''
    Trainer를 쓰기위해 hugging face access token이 필요하다.
    https://huggingface.co/settings/tokens 에서 access token을 할당 받고,
    `huggingface-cli login` 명령을 이용하여 acces token을 입력하자.
    '''
    trainer = Trainer(model=model, args=training_args,
                  compute_metrics=compute_metrics,
                  train_dataset=emotions_encoded["train"],
                  eval_dataset=emotions_encoded["validation"],
                  tokenizer=tokenizer)

    trainer.train()
    
    trainer.model.save_pretrained(local_model_path)
    tokenizer.save_pretrained(local_model_path)

    # 학습된 내용은 다시 hub에 commit 한다.
    #trainer.push_to_hub(commit_message="Training completed")

from transformers import AutoTokenizer, AutoModelForSequenceClassification, Pipeline

def test_use_tuned_model_local_store(local_model_path):

    tokenizer = AutoTokenizer.from_pretrained(local_model_path)
    model = AutoModelForSequenceClassification.from_pretrained(local_model_path)
    classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

    #custom_tweet = "I saw a movie today and it was really good."
    custom_tweet = "OpenAI will now let you create videos from verbal cues"
    preds = classifier(custom_tweet, return_all_scores=True)
    print(preds)

    #from datasets import load_dataset
    #emotions = load_dataset("emotion")
    #labels = emotions["train"].features["label"].names
    labels = ['sadness', 'joy', 'love', 'anger', 'fear', 'surprise']

    preds_df = pd.DataFrame(preds[0])
    plt.bar(labels, 100 * preds_df["score"], color='C0')
    plt.title(f'"{custom_tweet}"')
    plt.ylabel("Class probability (%)")
    plt.show()

from transformers import BertModel, BertTokenizer

def test_bert_wordembedding():

    # BERT 모델과 토크나이저 불러오기
    model_name = 'bert-base-uncased'  # 또는 원하는 다른 BERT 모델 이름
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    text = "Example text to extract word embeddings."

    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**tokens)

    word_embeddings = outputs.last_hidden_state
    print(word_embeddings)


def test_bert_multilang_wordembedding():

    model_name = 'bert-base-multilingual-cased'
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertModel.from_pretrained(model_name)

    text = "[CLS] " + "한국어 워드 임베딩 지원" + " [SEP]"

    tokens = tokenizer(text, return_tensors='pt', padding=True, truncation=True)

    with torch.no_grad():
        outputs = model(**tokens)

    word_embeddings = outputs.last_hidden_state

    print(tokenizer.tokenize(text))
    print(word_embeddings.shape)
    print(word_embeddings)

from transformers import AutoTokenizer, AutoModelForCausalLM

# https://huggingface.co/google/gemma-2b
# https://luvris2.tistory.com/865#article-3-1--hugging-face-cli-%EB%9D%BC%EC%9D%B4%EB%B8%8C%EB%9F%AC%EB%A6%AC-%EC%84%A4%EC%B9%98-%EB%B0%8F-%EB%A1%9C%EA%B7%B8%EC%9D%B8
def test_gemma_2b():
    tokenizer = AutoTokenizer.from_pretrained("google/gemma-2b")
    model = AutoModelForCausalLM.from_pretrained("google/gemma-2b")
    input_text = "Write me a poem about Machine Learning."
    input_ids = tokenizer(input_text, return_tensors="pt")
    outputs = model.generate(**input_ids)
    print(tokenizer.decode(outputs[0]))


#test_huggingface_pipeline_use()
#test_translate_kr2en()
#test_datasets()
#test_model_tuning()
#test_use_tuned_model()
#test_bert_wordembedding()
#test_bert_multilang_wordembedding()
test_gemma_2b()
local_model_path = "/home/sdn/Workspace/my_model"
#test_model_tuning_local_store(local_model_path)
#test_use_tuned_model_local_store(local_model_path)
