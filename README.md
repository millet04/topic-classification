# Topic Classification

## 1. Introduction

__Topic Classification__, the task of classifying the topic of a sequence, is a simple but useful task in practice. Since the development of Pretrained Language Models (PLMs),
fine-tuning them with a classifier has become a common method for solving topic classification tasks. However, there are other effective methods to solve the task without using a
classifier. This post experiments with various methods, including general classification, to explore their performance on topic classification. Korean BERT (klue/bert-base) and
KLUE Topic Classification dataset are used for the experiment. 

## 2. Methodology 

### (1) Classification

### (2) Masked Language Modeling

Masked Language Modeling (MLM) involves filling a masked token with the proper token to recover the meaning of the sentence. (Schick and Schütze, 2021a) converted classification 
task into MLM task by using verbalizer. When the MLM model finds the token with the highest probability for the masked token in the prompt (pattern), the verbalizer maps it to
the original classification label. 





### (3) Matching

### (4) Seq2Seq

---
