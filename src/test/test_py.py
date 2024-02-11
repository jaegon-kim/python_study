

def test_reversed():
    for i in range(3):
        print(i, end=", ")
    print()
    for i in reversed(range(3)):
        print(i, end=", ")
    print()

def test_tokenizing():
    text = "Tokenizing is a core task of NLP"
    tokenized_text = list(text)
    print(tokenized_text)
    s = sorted(set(tokenized_text))
    print(s)
    for idx, ch in  enumerate(s):
        print(idx, ':', ch, end=", ")
    print()

test_reversed()
test_tokenizing()