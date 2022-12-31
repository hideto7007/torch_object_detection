
# 外部ライブラリ
import cv2
import matplotlib.pyplot as plt
import numpy as np
import matplotlib
matplotlib.use('TkAgg')


# 機械学習ライブラリ
import torch


# 自作モジュール
from dataset.dataset import make_datapath_list, Anno_xml2list, DataTransform, VOCDataset, od_collate_fn
from config import Config
from models.ssd_model import SSD
from models.ssd_prediction import SSDPredictShow



# インスタンス化
config = Config()
transform_anno = Anno_xml2list(config.voc_classes)


# ファイルパスのリストを作成
train_img_list, train_anno_list, val_img_list, val_anno_list = make_datapath_list(config.path)

# datasetの作成

train_dataset = VOCDataset(train_img_list, train_anno_list, phase="val", transform=DataTransform(
    config.input_size, config.mean), transform_anno=Anno_xml2list(config.voc_classes))

val_dataset = VOCDataset(val_img_list, val_anno_list, phase="val", transform=DataTransform(
    config.input_size, config.mean), transform_anno=Anno_xml2list(config.voc_classes))



# SSDネットワークモデル
net = SSD(phase="inference", cfg=config.ssd_cfg)
net.eval()

# SSDの学習済みの重みを設定
net_weights = torch.load('./weights/ssd300_10.pth', map_location={'cuda:0': 'cpu'})

#net_weights = torch.load('./weights/ssd300_mAP_77.43_v2.pth', map_location={'cuda:0': 'cpu'})

net.load_state_dict(net_weights)

# GPUが使えるかを確認
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("使用デバイス：", device)

print('ネットワーク設定完了：学習済みの重みをロードしました')

# 結果の描画
ssd = SSDPredictShow(img_list=train_img_list, dataset=train_dataset, eval_categories=config.voc_classes,
                     net=net, dataconfidence_level=0.6)
img_index = 10
ssd.show(img_index, "predict")
ssd.show(img_index, "ans")


# 結果の描画
ssd = SSDPredictShow(img_list=val_img_list, dataset=val_dataset, eval_categories=config.voc_classes,
                     net=net, dataconfidence_level=0.6)
img_index = 10
ssd.show(img_index, "predict")
ssd.show(img_index, "ans")