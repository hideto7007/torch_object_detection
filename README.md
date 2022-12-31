# torch_object_detection

### Pytorchで物体検出(SSD)

- version
  - torch 1.13.0
  - torchvision 0.14.1


### 仮想環境
- 仮想環境作成時の参考
  - source env/Scripts/activate
  - pip install -r requirements.txt


### 実行スクリプト

- train.py
  - train.sh

### 実行手順

- data,weightがない場合
  - prior_preparation/make_folders_and_data_downloads.pyを実行しdataとweightを用意する

- train.pyのエポック数を各マシンの性能に合わせて変更
  - cpuでは10エポックで約30時間かかる
  - gpuが使える場合、train.pyのL141, 181, 182のコメントアウトを解除する
  - gpuで50エポックにした場合、学習時間は約6時間程

- train.sh実行(学習)

- test.sh実行(推論)
  - test.pyのL45のロードモデルを今回実行したモデルに変更
     - train.shで実行した学習済みモデルはssd300_〇〇.pthでweightに出力される

- test.shで実行すると、推論結果が出力される
  - 現状は、馬の推論のみしか出力されない
