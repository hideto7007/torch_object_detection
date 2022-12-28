

class Config:
    def __init__(self):
        self.num_epochs = 30
        self.path = "./data/VOCdevkit/VOC2012/"
        self.img_flg = False
        self.cuda = 0
        self.voc_classes = ['aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor']
        # ミニバッチのサイズを指定
        self.batch_size = 32
        # GPU設定判別
        if self.cuda:
            self.size = 2242
        else:
            self.size = 100
        self.mean = (104, 117, 123)
        self.input_size = 300
        self.std = (0.229, 0.224, 0.225)
        # vgg16, resnet, efficientNet
        self.model = "vgg16"
        self.fine_flag = 1
        self.out_feature = 2
        self.print_debug = False
        # SSD300の設定
        self.ssd_cfg = {
                        'num_classes': 21,  # 背景クラスを含めた合計クラス数
                        'input_size': 300,  # 画像の入力サイズ
                        'bbox_aspect_num': [4, 6, 6, 6, 4, 4],  # 出力するDBoxのアスペクト比の種類
                        'feature_maps': [38, 19, 10, 5, 3, 1],  # 各sourceの画像サイズ
                        'steps': [8, 16, 32, 64, 100, 300],  # DBOXの大きさを決める
                        'min_sizes': [30, 60, 111, 162, 213, 264],  # DBOXの大きさを決める
                        'max_sizes': [60, 111, 162, 213, 264, 315],  # DBOXの大きさを決める
                        'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
                    }