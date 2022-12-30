

# 外部ライブラリ
import os.path as osp
import random
import time
# XMLをファイルやテキストから読み込んだり、加工したり、保存したりするためのライブラリ
import xml.etree.ElementTree as ET
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

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
from models.loss_function import MultiBoxLoss


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


# 損失関数の設定
criterion = MultiBoxLoss(jaccard_thresh=0.5, neg_pos=3, device=device)

# 最適化手法の設定
optimizer = optim.SGD(net.parameters(), lr=1e-3,
                      momentum=0.9, weight_decay=5e-4)



# モデルを学習させる関数を作成


def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):

    # GPUが使えるかを確認
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("使用デバイス：", device)

    # ネットワークをGPUへ
    # net.to(device)

    # ネットワークがある程度固定であれば、高速化させる
    torch.backends.cudnn.benchmark = True

    # イテレーションカウンタをセット
    iteration = 1
    epoch_train_loss = 0.0  # epochの損失和
    epoch_val_loss = 0.0  # epochの損失和
    logs = []

    # epochのループ
    for epoch in range(num_epochs+1):

        # 開始時刻を保存
        t_epoch_start = time.time()
        t_iter_start = time.time()

        print('-------------')
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-------------')

        # epochごとの訓練と検証のループ
        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()  # モデルを訓練モードに
                print('(train)')
            else:
                if((epoch+1) % 10 == 0):
                    net.eval()   # モデルを検証モードに
                    print('-------------')
                    print('(val)')
                else:
                    # 検証は10回に1回だけ行う
                    continue

            # データローダーからminibatchずつ取り出すループ
            for images, targets in dataloaders_dict[phase]:

                # GPUが使えるならGPUにデータを送る
                # images = images.to(device)
                # targets = [ann.to(device) for ann in targets]  # リストの各要素のテンソルをGPUへ

                # optimizerを初期化
                optimizer.zero_grad()

                # 順伝搬（forward）計算
                with torch.set_grad_enabled(phase == 'train'):
                    # 順伝搬（forward）計算
                    outputs = net(images)

                    # 損失の計算
                    loss_l, loss_c = criterion(outputs, targets)
                    loss = loss_l + loss_c

                    # 訓練時はバックプロパゲーション
                    if phase == 'train':
                        loss.backward()  # 勾配の計算

                        # 勾配が大きくなりすぎると計算が不安定になるので、clipで最大でも勾配2.0に留める
                        nn.utils.clip_grad_value_(
                            net.parameters(), clip_value=2.0)

                        optimizer.step()  # パラメータ更新

                        if (iteration % 10 == 0):  # 10iterに1度、lossを表示
                            t_iter_finish = time.time()
                            duration = t_iter_finish - t_iter_start
                            print('イテレーション {} || Loss: {:.4f} || 10iter: {:.4f} sec.'.format(
                                iteration, loss.item(), duration))
                            t_iter_start = time.time()

                        epoch_train_loss += loss.item()
                        iteration += 1

                    # 検証時
                    else:
                        epoch_val_loss += loss.item()

        # epochのphaseごとのlossと正解率
        t_epoch_finish = time.time()
        print('-------------')
        print('epoch {} || Epoch_TRAIN_Loss:{:.4f} ||Epoch_VAL_Loss:{:.4f}'.format(
            epoch+1, epoch_train_loss, epoch_val_loss))
        print('timer:  {:.4f} sec.'.format(t_epoch_finish - t_epoch_start))
        t_epoch_start = time.time()

        # ログを保存
        log_epoch = {'epoch': epoch+1,
                     'train_loss': epoch_train_loss, 'val_loss': epoch_val_loss}
        logs.append(log_epoch)
        df = pd.DataFrame(logs)
        df.to_csv("log_output.csv")

        epoch_train_loss = 0.0  # epochの損失和
        epoch_val_loss = 0.0  # epochの損失和

        # ネットワークを保存する
        if ((epoch+1) % 10 == 0):
            torch.save(net.state_dict(), 'weights/ssd300_' +
                       str(epoch+1) + '.pth')
            
            
# 学習・検証を実行する
num_epochs= 10
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs=num_epochs)