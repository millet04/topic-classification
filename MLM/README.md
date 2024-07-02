## (2) Masked Language Modeling

### ■ Introduction

Masked Language Modeling (MLM) involves filling a masked token with the proper token to recover the meaning of the sentence. (Schick and Schütze, 2021a) converted classification task into MLM task by using verbalizer. When the MLM model finds the token with the highest probability for the masked token in the prompt (pattern), the verbalizer maps it to the original classification label.


For example, assume that model has to classify sentiment of the given text 'The Pizza is '. Following (Schick and Schütze, 2021a), the prompt is made by entering the text into pattern      

### ■ Performance

### ■ Implementation
