---
title: "Text Preprocessing"
tags: [preprocessing, nlp, bag-of-words, tf-idf, n-grams]
---

# Text Preprocessing

# 1. Text Classification

## Token-level Tokenization:
- Squence Labeling: Word Segmentation, Part Of Speech Tagging (POS), Named Entity Recognition (NER).
![Pasted image 20260807211921](../../../05_Assets/Images/Pasted%20image%2020260807211921.png)

## Document-level Tokenization
![Pasted image 20260807211957](../../../05_Assets/Images/Pasted%20image%2020260807211957.png)

## Basic Pipeline

```
Dataset -> Preprocessing and EDA -> Representation -> Modeling -> Evaluation
```

# 2. Text Preprocessing

- Ta chỉ quan tâm đến các 
## Các cách cơ bản

- Removal of URLs and HTML tags
- Text Standardizing
- Lowercasing
- Number and Punctuation Handling
- Removal Stop Words and Rare Words
- Handle Emoju and Emoticons
- Spelling Correction
- Tokenization: Sentence, Word, Character, Subwords
- Stemming
- Lemmatization

### Removal URLs, HTML Tags
- Extract text based on the structure of an HTML document
- URLs: imgae links, reference links,...
- HTML tags: p, div, ....

![Pasted image 20260807214035](../../../05_Assets/Images/Pasted%20image%2020260807214035.png)
### Text Standardizing
- Lowercasing: Use lower() function in Python
- Using short words and abbreviations to represent the same meaning
- Contrastions: I'm, isn't, can't, ....
![Pasted image 20260807215708](../../../05_Assets/Images/Pasted%20image%2020260807215708.png)
### Number and Punctuation Handling
- Removal: Text Classification
- As token: Machine Translation, POS tagging, Named Entity Recognition
![Pasted image 20260807215802](../../../05_Assets/Images/Pasted%20image%2020260807215802.png)![Pasted image 20260807215810](../../../05_Assets/Images/Pasted%20image%2020260807215810.png)
### Stop/ Rare Words Handling
- Focus on the importance keywords
- Stop words: Common words - no meaning or less meaning compared to keywords
	- English: a, an, that, for,...
	- Vietnamese: à, ừ, vậy, thế,...
- Rare words: words that appear only a few times in corpus
![Pasted image 20260807215956](../../../05_Assets/Images/Pasted%20image%2020260807215956.png)

### Emoji and Emoticons Handling
![Pasted image 20260807220022](../../../05_Assets/Images/Pasted%20image%2020260807220022.png)
### Steamming and Lemmatization
#### Lemmatization
- Từ có cùng một ngữ nghĩa nhưng hình thái khác nhau $\rightarrow$ chuyển về chung dạng
- Ví dụ: 
	- am, is, are $\rightarrow$ be
	- buy, bought, buying $\rightarrow$ buy
- Morphological parsing (phân tách tiền tố, hậu tố)
	- Morphology: The small meaningful units that make up words
		- Stems: The core meaning-bearing units
		- Affixes: Parts that adhere to stems, often with grammatical functions
	- Morphological Parsers:
		![Pasted image 20260807220515](../../../05_Assets/Images/Pasted%20image%2020260807220515.png)
#### Stemming
- Stemming - Simple Lemmatization
	- Naive version of morphological analysis
	- Chopping of words-final stemming affixes
	![Pasted image 20260807220618](../../../05_Assets/Images/Pasted%20image%2020260807220618.png)

### Tokenization
- Split paragraph, document into sentences
- Use RegEx or library: nltk, genism, ...
	-  nltk.sent_tokenize() $\rightarrow$  dựa vào dấu câu để tách
	- ![Pasted image 20260807221013](../../../05_Assets/Images/Pasted%20image%2020260807221013.png)
	- nltk.word_tokenize() $\rightarrow$ tách theo từ
	- ![Pasted image 20260807221020](../../../05_Assets/Images/Pasted%20image%2020260807221020.png)
- Các cách tách:
	- Word level : nhược điểm từ điển lớn
	- Char level : nhược điểm sẽ tạo một vector dài
	- Sub-word level: khắc phục nhược điểm của hai phương pháp trên.
![Pasted image 20260807221117](../../../05_Assets/Images/Pasted%20image%2020260807221117.png)

# 3. Text Representation
## Numeric Representation
![Pasted image 20260807221302](../../../05_Assets/Images/Pasted%20image%2020260807221302.png)

## One-hot Encoding
- Token-level
- Represented by a V-dimensional binary vector of 0s and 1s
	- All 0s barring the index, $index = w_{id}$
	- At this index, put 1
	![Pasted image 20260807221431](../../../05_Assets/Images/Pasted%20image%2020260807221431.png)
	![Pasted image 20260807221449](../../../05_Assets/Images/Pasted%20image%2020260807221449.png)

## Bag of Word (BoW)
- Document-Level: Consider text as a bag (collection) of words
- Represented by a V-dimensional
- Use: the number of occurrences of the word in the document
- Nhược điểm:
	- Không xem xét thứ tự của các từ trong câu, làm sai lệch ý nghĩa của văn bản
	- Chống lại cấu trúc ngữ pháp
	- Chống lại tính đa dạng
![Pasted image 20260807221646](../../../05_Assets/Images/Pasted%20image%2020260807221646.png)
![Pasted image 20260807221654](../../../05_Assets/Images/Pasted%20image%2020260807221654.png)


## TF-IDF
$$tf_{t,d} = count(t, d)$$
- Some ways to reduct the raw frequency:
	- Using log space + add 1:
	$$tf_{t,d} = \log(count(t,d) + 1)$$
	- Divide the number of occurrences by the length of document:
	$$tf_{t,d} = \frac{count(t,d)}{len(d)}$$
	![Pasted image 20260807222230](../../../05_Assets/Images/Pasted%20image%2020260807222230.png)

$$idf_i = \frac{N}{df_t}$$
- Measures the importane of the word across a corpus
	- N: The total number of documents in the corpus
	- $df_t$ : The number of documents with term t in them
	- Using log space:
$$
idf_t = \log \frac{N}{df_t} \quad ; \quad idf_t = \log \frac{N}{df_t} + 1 \quad ; \quad idf_t = \log \frac{N + 1}{df_t + 1} + 1
$$
- The weighted value $w_{t,d}$ for word t in document d
$$w_{t,d} = tf_{t,d} \times idf_{t,d} $$
- IDF weighs down the terms: very common across a corpus and rare terms
- The TF-IDF vector representation for a document is then simply TF-IDF score for each term in that document.