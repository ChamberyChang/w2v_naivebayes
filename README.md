# NaiveBayes Classifier for Japanese Language

## File Directory


* /data(ignore) --- Place for training data, test data and augmented data. (CSV format)
* /result(ignore) --- Place for classified reports.(XLSX format)
* /models(ignore) --- Place for generated models.
* config.json(ignore) --- configuration file
* config.template.json --- configuration file template
* requirement.txt --- Environment requirement. (strictly required with newest Anaconda)
* .gitignore --- gitignore
* *Config.py* --- Read configuration file.
* *ProcessData.py* --- Read CSV file.
* *ReprocessData.py* --- Read augmented CSV file and re-process new augmented data.
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

Report including Confusion Matrix, and Accuray, Precision, Recall, K-measure by *sklean.metrics*

---------------------------------------------------
## Configuration
All you need to customize is the `config.json`, and `config.template.json` is the template.
You can run the program to generate the default structure.
1. `train_path`, `test_path`, `model_path`, `augment_path`, `label`, `result_path`
2. `corpus_*` for generate models.
3. `method` for change method between *tfidf*, *Countvector* and *tfidfvector*.
4. `dataaugment` to switch the word augmenting with model option.
5. `MeCab.Tagger("mecabrc")` in *MorphologicalAnalysis.py* if you want to use other dictionary.
   
---------------------------------------------------
## License
```
chiVe    (Apache License 2.0)
```