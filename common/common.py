# 外部ライブラリ
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 自作モジュール
from dataset.dataset import make_datapath_list, Anno_xml2list, DataTransform
from config import Config


# インスタンス化
# 前処理クラスの作成
config = Config()
transform = DataTransform(config.input_size, config.mean)


# ファイルパスのリストを作成
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(
    config.path)


# 画像の読み込み OpenCVを使用
ind = 1
image_file_path = val_img_list[ind]
img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
height, width, channels = img.shape  # 画像のサイズを取得

# アノテーションをリストで表示
transform_anno = Anno_xml2list(config.voc_classes)
anno_list = transform_anno(train_anno_list[0], width, height)
transform_anno_list = transform_anno(val_anno_list[ind], width, height)


# 動作の確認
# 1. 画像読み込み
image_file_path = train_img_list[0]
img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
height, width, channels = img.shape  # 画像のサイズを取得

# 2. アノテーションをリストに
transform_anno = Anno_xml2list(config.voc_classes)
anno_list = transform_anno(train_anno_list[0], width, height)


# 4. 前処理クラスの作成
transform = DataTransform(config.input_size, config.mean)


def img_show():
    """確認用の画像表示"""
    
    # 3. 元画像の表示
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show()
    
    
    # 5. train画像の表示
    phase = "train"
    img_transformed, boxes, labels = transform(
        img, phase, anno_list[:, :4], anno_list[:, 4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.show()


    # 6. val画像の表示
    phase = "val"
    img_transformed, boxes, labels = transform(
        img, phase, anno_list[:, :4], anno_list[:, 4])
    plt.imshow(cv2.cvtColor(img_transformed, cv2.COLOR_BGR2RGB))
    plt.show()