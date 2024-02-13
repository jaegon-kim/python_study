

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

import copy
def test_deep_copy():
    original_list = [[1, 2, 3], [4, 5, 6]]
    copied_list = copy.deepcopy(original_list)
    original_list[0][0] = 100
    print(original_list)
    print(copied_list)

test_reversed()
test_tokenizing()
test_deep_copy()
