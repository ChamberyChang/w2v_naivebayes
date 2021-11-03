# ---naivebayes program---

ナイーブベイズ分類器の学習のために作成したプログラム。
写経に近いかも

## システム概要

ナイーブベイズ分類器をpythonのパッケージskleanを利用して作成する。

## ファイル構成

* /CreateDataset --- 作成したcsvファイルを読み取りデータセットを作成するclass。
* /dataset(ignore) --- 学習用のデータセットと、テストのデータセットを入れておく。
* /Morphological --- データを読み取り形態素解析を行うclass。
* /result(ignore) --- classification_reportなどの実行結果ログを保存する場所
* /setting --- pipのrequire.txtを入れてある。バージョン管理用テキストファイル
* /word2vec_model(ignore) --- word2vecの学習済みモデルが格納されている
* .gitignore     --- gitignoreファイル
* *data_augment.py(現在動きません)* --- googletransのデータ拡張用のファイル。
* *naivebayse.py* --- 実行のメインファイル。これを実行することでナイーブベイズが行われる。
* README.mb     --- readmeファイル

---------------------------------------------------
* **gitignoreしたもの**
 * /detaset

train用とtest用のcsvファイルとtrainを拡張したときに生成されるcsvファイル。
これは
` 文章　| 分類 `
が連続してなるcsvで文章をデータ、分類を答えとして学習を行うようになっている。

 * result/result_report.xlsx

testdataをpredictしてどれだけの精度なのか確かめるためのエクセルファイル。

* /word2vec_model

word2vecのモデルが格納されているフォルダ

---------------------------------------------------
## 実行方法

Pythonファイルのnaivebayse.pyがあるディレクトリで

`python __init__.py`

を実行することでプログラムが動作する。

---------------------------------------------------
## 処理の手順

1. まずデータセットを呼び出す。
2. 形態素解析を行う。
3. tfidf, Bag Of word, tfidfvectorの中から選択したもので数値に変換する。
4. skleanのMultinomialNBにかける。
5. predictを行って、結果からログをエクセル形式で作成する

---------------------------------------------------
## データ拡張について

### `word2vecを利用したデータ拡張`

上気した通りデータ数が少ないため、データ拡張を行う。
こっちの方法はword2vecを利用して

1. 一つの文章を形態素解析する。
2. 形態素解析した結果の品詞一つをword2vecにかけ類似した単語(文や人の場合もある)を出力する。
3. 類似した単語を連結し、データ拡張に追加。
4. 学習を行う。

---------------------------------------------------
## 評価方法について

ナイーブベイズ分類器にかけた際に、accuracyがあまり上がらないという現象が発生した。
これはカテゴリーを分ける際に複数のカテゴリーをまたいでいるデータが多数あったため起きた問題だということがわかった。
そこで複数のカテゴリーにまたいだものはその中からどれか一つでも判別することができていた場合、accuracyに数値を与えることにする。

ただこの評価法は適切ではない部分があるため選択肢として与えることとした。
`evoluationの数値変更で評価方法の変更が可能になった。`
(まあ数値が低かったので救済的な部分が大きい。)

近似しているカテゴリーが存在している場合、この評価方法の変更によって近似している部分での誤判定か確認をすることができる。
