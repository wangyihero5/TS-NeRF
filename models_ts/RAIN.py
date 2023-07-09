import torch.nn as nn
import torch

from models_ts.function import calc_mean_std, weights_init_kaiming
import clip
from models_ts.DirectionLoss import CLIPLoss
from ZSSGAN.utils.text_templates import imagenet_templates

# 使用nn创建的会自动放入parameter
vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(inplace=True),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(inplace=True),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=False), # hero: 原版为ceil_mode=True
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(inplace=True),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(inplace=True),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=False),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(inplace=True),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=False),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=False),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(inplace=True)  # relu5-4
)

fc_encoder = nn.Sequential(
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 1024)
)
device = "cuda" if torch.cuda.is_available() else "cpu"
class Net(nn.Module):
    def __init__(self, encoder, fc_encoder):
        super(Net, self).__init__()
        enc_layers = list(encoder.children())
        self.enc_1 = nn.Sequential(*enc_layers[:4])  # input -> relu1_1
        self.enc_2 = nn.Sequential(*enc_layers[4:11])  # relu1_1 -> relu2_1
        self.enc_3 = nn.Sequential(*enc_layers[11:18])  # relu2_1 -> relu3_1
        self.enc_4 = nn.Sequential(*enc_layers[18:31])  # relu3_1 -> relu4_1
        self.fc_encoder = fc_encoder

        # self.clip_loss = DirectionLoss.CLIPLoss(device) # 计算方向clip loss
        # fix the encoder
        for name in ['enc_1', 'enc_2', 'enc_3', 'enc_4']:
            for param in getattr(self, name).parameters():
                param.requires_grad = False
                
        self.fc_encoder.apply(weights_init_kaiming)

    # extract relu1_1, relu2_1, relu3_1, relu4_1 from input image
    def encode_with_intermediate(self, input):
        results = [input]
        for i in range(4):
            func = getattr(self, 'enc_{:d}'.format(i + 1))
            results.append(func(results[-1]))
        return results[1:]

    
    # extract relu4_1 from input image 提取第4-1
    def get_content_feat(self, input):
        for i in range(4):
            # getattr解释的很抽象 告诉我这个函数的作用相当于是object.name，可以获得函数名（如：Sequence）+（input）-->就可以使用Sequence进行forward
            input = getattr(self, 'enc_{:d}'.format(i + 1))(input)
        return input

    # 输入stylePic得到性相应的feature、方差
    def get_style_feat(self, input):
        style_feats = self.encode_with_intermediate(input)
        out_mean = []
        out_std = []
        out_mean_std = []
        for style_feat in style_feats:
            style_feat_mean, style_feat_std, style_feat_mean_std = self.calc_feat_mean_std(style_feat)
            out_mean.append(style_feat_mean)
            out_std.append(style_feat_std)
            out_mean_std.append(style_feat_mean_std)
        return style_feats, torch.cat(out_mean_std, dim=-1)

    
    def calc_feat_mean_std(self, input, eps=1e-5):
        # eps is a small value added to the variance to avoid divide-by-zero.
        size = input.size()
        assert (len(size) == 4)
        N, C = size[:2]
        feat_var = input.view(N, C, -1).var(dim=2) + eps
        feat_std = feat_var.sqrt().view(N, C)
        feat_mean = input.view(N, C, -1).mean(dim=2).view(N, C)
        return feat_mean, feat_std, torch.cat([feat_mean, feat_std], dim = 1)
    
    # 输入style图，使用encode产生相应的styleLatentCode
    def get_hyper_input(self, style):
        style_feats = self.encode_with_intermediate(style)  # 获得style的4个中间特征
        _, _, style_feat_mean_std = self.calc_feat_mean_std(style_feats[-1]) # 计算最后一个特征的均方差

        intermediate = self.fc_encoder(style_feat_mean_std) # style_feat_mean_std（1，1024）
        intermediate_mean = intermediate[:, :512]

        return intermediate_mean


    #  输入text，使用encode产生相应的styleLatentCode
    #  hero15 需要修改
    def get_hyperText_input(self, text,target_class="sketch"):
        # style_feats = self.encode_with_intermediate(style)  # 获得style的4个中间特征
        # _, _, style_feat_mean_std = self.calc_feat_mean_std(style_feats[-1]) # 计算最后一个特征的均方差
        # labels = ["dog", "cat", "bird", "person", "mushroom", "cup"]
        # text = [f"A photo of {label}" for label in labels]

        # # text encoding
        clipDir=CLIPLoss(device)
        source_features = clipDir.get_text_features(target_class, templates=imagenet_templates)
        # 将文本模版的text向量，求均值
        source_features=torch.mean(source_features, dim=0, keepdim=True).float()

        '''
        model = clipTools.get_clip_model()
        text = "A photo of sketch"
        text = clip.tokenize(text).to(device)
        with torch.no_grad():
            # 现在可以通过从CLIP模型中调用“encode_text”方法来提取文本特征
            text_mean = model.encode_text(text)  # (1,512) 感觉处理后也是text的一种分布，可以替代img的styleEAV分布-->后续hypernet可以再fineturn
        # Style VAE 相比目前存在的模型，能够以更广泛的样式生成更准确的图像描述
        # https://www.e-learn.cn/topic/1761980
        # intermediate = self.fc_encoder(text)  # style_feat_mean_std（1，1024）
        # intermediate_mean = intermediate[:, :512]

        return text_mean.float()  # shape需要（1，512）
        '''
        return source_features
