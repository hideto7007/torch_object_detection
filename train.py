

# 外部ライブラリ
import os.path as osp
import random
# XMLをファイルやテキストから読み込んだり、加工したり、保存したりするためのライブラリ
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import numpy as np

# 機械学習ライブラリ
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import torch.utils.data as data


# 自作モジュール
from dataset.dataset import make_datapath_list, Anno_xml2list, DataTransform, VOCDataset, od_collate_fn
from config import Config
from common.common import img_show
from models.SSD_model import SSD


# 乱数のシードを設定
torch.manual_seed(1234)
np.random.seed(1234)
random.seed(1234)


# インスタンス化
config = Config()
transform_anno = Anno_xml2list(config.voc_classes)


# ファイルパスのリストを作成
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(config.path)

# アノテーションをリストに
transform_anno = Anno_xml2list(config.voc_classes)

# 前処理クラスの作成
transform = DataTransform(config.input_size, config.mean)

if config.img_flg:
    img_show()
    

# datasetの作成

train_dataset = VOCDataset(train_img_list, train_anno_list, phase="train", transform=DataTransform(
    config.input_size, config.mean), transform_anno=Anno_xml2list(config.voc_classes))

val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
    config.input_size, config.mean), transform_anno=Anno_xml2list(config.voc_classes))


# dataloderの作成

train_dataloader = data.DataLoader(
    train_dataset, batch_size=config.batch_size, shuffle=True, collate_fn=od_collate_fn)

val_dataloader = data.DataLoader(
    val_dataset, batch_size=config.batch_size, shuffle=False, collate_fn=od_collate_fn)

# 辞書型変数にまとめる
dataloaders_dict = {"train": train_dataloader, "val": val_dataloader}

# 動作の確認
batch_iterator = iter(dataloaders_dict["val"])  # イタレータに変換
images, targets = next(batch_iterator)  # 1番目の要素を取り出す


if config.print_debug:
    print(images.size())  # torch.Size([4, 3, 300, 300])
    print(len(targets))
    print(targets[1].size())  # ミニバッチのサイズのリスト、各要素は[n, 5]、nは物体数

    print(train_dataset.__len__())
    print(val_dataset.__len__())


# バッチサイズは全て一律の値でないと、モデル学習時に値が一致してなくエラーとなる
# バッチサイズが一致してるか確認
# for i in train_dataloader:
#     print(i[1][2].size())


# 動作確認
net = SSD(phase="train", cfg=config.ssd_cfg)


# SSDの初期の重みを設定

# loadデータ
load_list = ['./weights/vgg16_reducedfc.pth', './weights/ssd300_50.pth', './weights/ssd300_mAP_77.43_v2.pth']

vgg_weights = torch.load(load_list[0])
net.vgg.load_state_dict(vgg_weights)


# ssdのその他のネットワークの重みはHeの初期値で初期化

def weights_init(m):
    if isinstance(m, nn.Conv2d):
        init.kaiming_normal_(m.weight.data)
        if m.bias is not None:  # バイアス項がある場合
            nn.init.constant_(m.bias, 0.0)
            
            
# Heの初期値を適用
net.extras.apply(weights_init)
net.loc.apply(weights_init)
net.conf.apply(weights_init)

# GPUが使えるかを確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)

print('ネットワーク設定完了：学習済みの重みをロードしました')