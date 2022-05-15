## 民泊サービスの宿泊料金予測
### 10th Solution

#### モデル
 - LightGBM,NNなど8つのモデル → SVRでスタッキング(5foldの予測平均が最終予測)

#### バリデーション
 - KFold(5fold)<br>
 ※host_idでGroupKFoldしたが、LBが下がったためシンプルにKFoldを採用<br>

#### 特徴量
 - name列(小文字化、不要な記号除去をした後以下の方法で特徴量化)
   - Bertの学習済みモデル(bert-base-multilingual-uncased)から特徴量化：768次元
   - Universal sentence encoderによる特徴量化：512次元
   - TF-IDF → SVDによる次元削減：50次元
   - TF-IDF → NMFによる次元削減：50次元
 - 位置情報系
   - 経度、緯度を混合ガウスモデルでソフトクラスタリング：10次元
   - 施設間の距離行列 → 多次元尺度法による次元削減：10次元
   - 各駅との距離 → PCAによる次元削減：10次元
   - 最寄り駅をラベルエンコード
 - 日付
   - 最終レビュー日から2020.4.30までの経過日数
 - カテゴリ列
   - 決定木ベース → labelencoder
   - 線形モデル or 距離ベースモデル → one-hot化をstandardscaler
   - neighbourhood ✖️ roomtypeのカテゴリ化
 
#### 欠損値補完
 - 全て0埋め<br>
 ※モデルによる予測補完をしたがスコアが下がったため、不採用<br>
   
#### ハイパーパラメーターチューニング
 - チューニングなし<br>
 ※optunaを使ったが、スコアが下がったため不採用(おそらくCVとLBのデータの特徴が異なるため？)<br>
 
#### 試したがうまくいかなったこと
 - データオーギュメンテーション
   - name列を【英 → 日 → 英】に再翻訳し、学習データに追加(単純に過学習となった)
 - 言語判定列
   - 文章が短い場合に【un】になってしまいうまく推定できなかった(fasttextを使えばよかった？)
 - neighbourhoodの分散表現(word2vec)
 - その他細かい実験をしたが、覚えていない...
 
#### 試せばよかったこと、その他反省
 - Bertで学習
   - name列が重要だとわかっていたため、英語や日本語などに統一してモデリングするべきだった
 - 実験方法
   - mlflowなどの実験管理ライブラリを使うべきだった
   - mlflowじゃなくても、excelなどで実験内容やコメントを残すべきだった
 - CV法の決定
   - 結局最後まで、何がベストかわからず消去法でKFoldとしていた
   - KFoldでもあてにならなかったので、LBを頼りにしていた
 - アンサンブル実装のタイミング
   - コンペ中盤でモデルをいくつも作ってのアンサンブルを行ったため実験に時間がかかってしまった
   - アンサンブルはコンペ終盤から始める。それまでは単体モデルの実験でスコア改善を行うべきだった