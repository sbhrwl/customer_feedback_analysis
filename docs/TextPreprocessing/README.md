# Text Preprocessing
## Terms
- Corpus ~ Book
  - Corpora is a singular
- Document ~ Page
- Token ~ Word

* Original Feedback
  ```
  "It offers all your essential services in one, and sells energy bundled with broadband, home phone, mobile, home insurance and boiler care."
  ```
* Punctuations removed
  ```
  "it offers all your essential services in one and sells energy bundled with broadband home phone mobile home insurance and boiler care"
  ```
* Tokenized feedback
  - The unit into which text is decomposed is called **token**
  - The process of decomposing text into token is called tokenization.
  ```
  ['it', 'offers', 'all', 'your', 'essential', 'services', 'in', 'one', 'and', 'sells', 'energy', 'bundled', 'with', 'broadband', 'home', 'phone', 'mobile', 'home', 'insurance', 'and', 'boiler', 'care']
  ```
* Stop words removed from feedback
  ```
  ['offers', 'essential', 'services', 'one', 'sells', 'energy', 'bundled', 'broadband', 'home', 'phone', 'mobile', 'home', 'insurance', 'boiler', 'care']
  ```
* n GRAM
  * Unigram
    ```python
    import jieba
 
    text = "I will go to the United States"
    cut = jieba.cut(text)
    sent = list(cut)
    print(sent)
    ```
    `['I', ' ', 'will', ' ', 'go', ' ', 'to', ' ', 'the', ' ', 'United', ' ', 'States']`
  * Bigram
    ```python
    Sent = "I will go to United States"
    lst_sent = Sent.split (" ")
    of_bigrams_in = []
    for i in range(len(lst_sent)- 1):
       of_bigrams_in.append(lst_sent[i]+ " " + lst_sent[ i + 1])

    print(of_bigrams_in)
    ```
    `['I will', 'will go', 'go to', 'to United', 'United States']`
  * Trigram
    ```python
    import re
    punctuation_pattern = re.compile(r"" "[.,!? ""] "" " )

    sent = "I will go to United States"
    no_punctuation_sent = re.sub(punctuation_pattern , " " , sent )
    lst_sent = no_punctuation_sent.split (" ")
    trigram = []
    for i in range(len(lst_sent)- 2):
       trigram.append(lst_sent[i] + " " + lst_sent[i + 1] + " " +lst_sent[i + 2])
    ```
    `['I will go', 'will go to', 'go to United', 'to United States']`

## Next step will be convert these token into Vectors using Word Embedding techniques
