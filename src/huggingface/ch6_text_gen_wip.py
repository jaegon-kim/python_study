import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F

#device = "cuda" if torch.cuda.is_available() else "cpu"
device = "cpu"

# 코랩의 경우 gpt2-xl을 사용하면 메모리 부족 에러가 발생합니다.
# 대신 "gpt" 또는 "gpt2-large"로 지정하거나 코랩 프로를 사용하세요.
#model_name = "gpt2-xl"
model_name = "gpt2-large"

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name).to(device)

import pandas as pd

#input_txt = "Transformers are the"
input_txt = """In a shocking finding, scientist discovered \
a herd of unicorns living in a remote, previously unexplored \
valley, in the Andes Mountains. Even more surprising to the \
researchers was the fact that the unicorns spoke perfect English.\n\n
"""
n_steps = 128

input_ids = tokenizer(input_txt, return_tensors="pt")["input_ids"].to(device)


def log_probs_from_logits(logits, labels):
    logp = F.log_softmax(logits, dim=-1)
    logp_label = torch.gather(logp, 2, labels.unsqueeze(2)).squeeze(-1)
    return logp_label

def sequence_logprob(model, labels, input_len=0):
    with torch.no_grad():
        output = model(labels)
        log_probs = log_probs_from_logits(
            output.logits[:, :-1, :], labels[:, 1:])
        seq_log_prob = torch.sum(log_probs[:, input_len:])
    return seq_log_prob.cpu().numpy()

output_greedy = model.generate(input_ids, max_new_tokens=n_steps, do_sample=False)

logp = sequence_logprob(model, output_greedy, input_len=len(input_ids[0]))
print(tokenizer.decode(output_greedy[0]))
print(f"\nGreedy Search 로그 확률: {logp:.2f}")


output_beam = model.generate(input_ids, max_length=n_steps, num_beams=5, do_sample=False)
logp = sequence_logprob(model, output_beam, input_len=len(input_ids[0]))
print(tokenizer.decode(output_beam[0]))
print(f"\nBeam Search 로그 확률: {logp:.2f}")

output_beam = model.generate(input_ids, max_length=n_steps, num_beams=5, do_sample=False, no_repeat_ngram_size=2)
logp = sequence_logprob(model, output_beam, input_len=len(input_ids[0]))
print(tokenizer.decode(output_beam[0]))
print(f"\Beam Searh + 반복 방지 로그 확률: {logp:.2f}")

