# NaiveBayes Classifier for Japanese Language

## File Directory


* /data(ignore) --- Place for training data, test data and augmented data. (CSV format)
* /result(ignore) --- Place for classified reports.(XLSX format)
* /models(ignore) --- Place for generated models.
* requirement.txt --- Environment requirement. (strictly required with newest Anaconda)
* .gitignore --- gitignore
* *CreateDataset.py* --- Read CSV file and setting class.
* *MorphologicalAnalysis.py* --- Morphological analyze with *MeCab*
* *GenModel.py* --- Generate *word2vec* model. (Maybe used to generate other models)
* *main.py* --- Run program by `python main.py`.
* README.md --- You are reading this.
---------------------------------------------------
CSV file requirement
```
 "Article1", "Category1"
 "Article2", "Category1"
 "Article3", "Category2" 
 ...
 ```

CSV Example 
```
"独女通信", dokujo-tsushin
"ITライフハック", it-life-hack
"家電チャンネル", kaden-channel
"livedoor HOMME", livedoor-homme
"MOVIE ENTER", movie-enter
"Peachy", peachy
"エスマックス", smax
"Sports Watch", sports-watch
"トピックニュース", topic-news
```
| Article         | Category        |
| --------------- | --------------- | 
| 独女通信  | dokujo-tsushin | 
| ITライフハック | it-life-hack | 
| 家電チャンネル | kaden-channel | 
| livedoor HOMME | livedoor-homme |
| MOVIE ENTER | movie-enter |
| Peachy | peachy |
| エスマックス | smax |
| Sports Watch | sports-watch |
| トピックニュース | topic-news |
---------------------------------------------------
 * result/result_report.xlsx

Report including Confuse Matrix, and Accuray, Precision, Recall, K-measure by *sklean.metrics*

---------------------------------------------------
## All you need to customize

1. `train_path`, `test_path`, `model_path`, `label` in *main.py* for work.
2. `corpus_*` in *main.py* for generate models.
3. `method` in *main.py* for change method between *tfidf*, *Bag Of word* and *tfidfvector*.
4. `if elif` function the same as `label`.
5. `MeCab.Tagger("mecabrc")` in *MorphologicalAnalysis.py* if you want to use other dictionary.
   
---------------------------------------------------
## License
```
chiVe    (Apache License 2.0)
```