"""

物体検出の事前準備

フォルダー作成とファイルダウンロード実施

"""

# 外部ライブラリ
import os
import urllib.request
import zipfile
import tarfile

# 変数定義
dir_list = ["./data", "./weights"]
url_list = [
    "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar", 
    "https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth",
    "https://s3.amazonaws.com/amdegroot-models/ssd300_mAP_77.43_v2.pth",
    "https://drive.google.com/open?id=1_zTTYQ2j0r-Qe3VBbHzvURD0c1P2ZSE9"
]
output_dir = ["./data", "./weights", "./weights", "./weights"]
name_list = [
    "VOCtrainval_11-May-2012.tar",
    "vgg16_reducedfc.pth",
    "ssd300_mAP_77.43_v2.pth",
    "ssd300_50.pth"
]

# フォルダ「data, weights」が存在しない場合は作成

print("フォルダー作成開始")
for dirs in dir_list:
    if not os.path.exists(dirs):
        os.mkdir(dirs)
        
        
print("フォルダー作成完了")
# 各データをダウンロード
for urls, name, output in zip(url_list, name_list, output_dir):
    target_path = os.path.join(output, name) 

    if not os.path.exists(target_path):
        urllib.request.urlretrieve(urls, target_path)
        print("{}のダウンロード".format(target_path))
        
        if output == "./data":
            print("{}の解凍開始".format(target_path))
            tar = tarfile.TarFile(target_path)  # tarファイルを読み込み
            tar.extractall(output)  # tarを解凍
            tar.close()  # tarファイルをクローズ
            print("{}の解凍終了".format(target_path))