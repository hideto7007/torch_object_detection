
# 外部ライブラリ
import cv2
import matplotlib.pyplot as plt
import numpy as np


# 機械学習ライブラリ
import torch


# 自作モジュール
from config import Config
from models.ssd_model import SSD
from dataset.dataset import DataTransform
from models.ssd_predict_show import SSDPredictShow



# インスタンス化
config = Config()


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


# 1. 画像読み込み
image_file_path = "./data/VOCdevkit/VOC2012/cowboy-757575_640.jpg"
img = cv2.imread(image_file_path)  # [高さ][幅][色BGR]
height, width, channels = img.shape  # 画像のサイズを取得

# 2. 元画像の表示
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
plt.show()

# 3. 前処理クラスの作成
transform = DataTransform(config.input_size, config.mean)

# 4. 前処理
phase = "val"
img_transformed, boxes, labels = transform(
    img, phase, "", "")  # アノテーションはないので、""にする
img = torch.from_numpy(img_transformed[:, :, (2, 1, 0)]).permute(2, 0, 1)

# 5. SSDで予測
net.eval()  # ネットワークを推論モードへ
x = img.unsqueeze(0)  # ミニバッチ化：torch.Size([1, 3, 300, 300])
detections = net(x)

print(detections.shape)
# print(detections)

# output : torch.Size([batch_num, 21, 200, 5])
#  =（batch_num、クラス、confのtop200、規格化されたBBoxの情報）
#   規格化されたBBoxの情報（確信度、xmin, ymin, xmax, ymax）

# 予測と、予測結果を画像で描画する
ssd = SSDPredictShow(eval_categories=config.voc_classes, net=net)
ssd.show(image_file_path, data_confidence_level=0.6)