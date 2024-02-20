from transformers import pipeline
import pandas as pd


def hf_translator_de():
    '''
    text = """Dear Amazon, last week I ordered an Optimus Prime action figure \
    fromyour online store in Germany. Unfortunately, when I pened the package, \
    I discovered to my horror that I had been sent an action figure of Megatron\
    instead! As a lifelong enemy of the Decepticons, I hope you can understand my \
    dilemma. To resolve the issue, I emand an exchange of Megatron for the \
    Optimus Prime figure I ordered. Enclosed are copies of my records concerning \
    this purchase. I expect to hear from you soon. Sincerely, Bumblebee."""
    print("text-classification")
    classifier = pipeline("text-classification")
    outputs = classifier(text)
    print(pd.DataFrame(outputs))

    print("ner");
    ner_tagger = pipeline("ner", aggregation_strategy="simple")
    outputs = ner_tagger(text)
    print(pd.DataFrame(outputs))

    print("## question-answering")
    reader = pipeline("question-answering")
    question = "What does the customer want?"
    outputs = reader(question=question, context=text)
    print(pd.DataFrame([outputs]))

    print("## summarization")
    summarizer = pipeline("summarization")
    outputs = summarizer(text, max_length=56, clean_up_tokenization_spaces=True)
    print(outputs[0]['summary_text'])

    print('## translation_en_to_de')
    translator = pipeline('translation_en_to_de', model="Helsinki-NLP/opus-mt-en-de")
    outputs = translator(text, clean_up_tokenization_spaces=True, min_length=100)
    print(outputs[0]['translation_text'])
    '''
    
    print('## translation_kr_to_en')
    translastor = pipeline("translation", model="Helsinki-NLP/opus-mt-ko-en")
    outputs = translastor("한국어를 영어로 번역해 주세요")
    print(outputs[0]['translation_text'])
    

hf_translator_de()