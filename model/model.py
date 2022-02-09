from . import resnet
from .domain_specific_module import BatchNormDomain
from utils import utils
from . import utils as model_utils
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function

backbones = [resnet]

# +
class ReverseLayerF(Function):

    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha

        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha

        return output, None

class FC_BN_ReLU_Domain(nn.Module):
    def __init__(self, in_dim, out_dim, num_domains_bn):
        super(FC_BN_ReLU_Domain, self).__init__()
        self.fc = nn.Linear(in_dim, out_dim)
        self.bn = BatchNormDomain(out_dim, num_domains_bn, nn.BatchNorm1d)
        self.relu = nn.ReLU(inplace=True)
        self.bn_domain = 0

    def set_bn_domain(self, domain=0):
        assert(domain < self.num_domains_bn), "The domain id exceeds the range."
        self.bn_domain = domain
        self.bn.set_domain(self.bn_domain)

    def forward(self, x):
        x = self.fc(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


# -

class DANet(nn.Module):
    def __init__(self, num_classes, feature_extractor='resnet101',
                 fx_pretrained=True, fc_hidden_dims=[], frozen=[],
                 num_domains_bn=2, dropout_ratio=(0.5,), temp=0.05):
        super(DANet, self).__init__()
        if feature_extractor == 'LeNet':
            self.feature_extractor = nn.Sequential(
                nn.Conv2d(3, 20, kernel_size=5),
                nn.MaxPool2d(2),
                nn.ReLU(),
                nn.Conv2d(20, 50, kernel_size=5),
                nn.Dropout2d(p=0.5),
                nn.MaxPool2d(2),
                nn.ReLU(),
            )
            feat_dim = 50*5*5
        else:
            self.feature_extractor = utils.find_class_by_name(
                   feature_extractor, backbones)(pretrained=fx_pretrained,
                   frozen=frozen, num_domains=num_domains_bn)
            feat_dim = self.feature_extractor.out_dim
            

        self.bn_domain = 0
        self.num_domains_bn = num_domains_bn
        self.temp = temp

        self.in_dim = feat_dim

        self.FC = nn.ModuleDict()
        self.dropout = nn.ModuleDict()
        self.num_hidden_layer = len(fc_hidden_dims)

        in_dim = feat_dim
        for k in range(self.num_hidden_layer):
            cur_dropout_ratio = dropout_ratio[k] if k < len(dropout_ratio) \
                      else 0.0
            self.dropout[str(k)] = nn.Dropout(p=cur_dropout_ratio)
            out_dim = fc_hidden_dims[k]
            self.FC[str(k)] = FC_BN_ReLU_Domain(in_dim, out_dim,
                  num_domains_bn)
            in_dim = out_dim

        cur_dropout_ratio = dropout_ratio[self.num_hidden_layer] \
                  if self.num_hidden_layer < len(dropout_ratio) else 0.0

        #self.dropout['logits'] = nn.Dropout(p=cur_dropout_ratio)
        #self.FC['logits'] = nn.Linear(in_dim, num_classes)
        
        
        if feature_extractor == 'LeNet':
            self.FC['fc1'] = nn.Linear(in_dim, 500)
            self.FC['logits'] = nn.Sequential(
                    nn.ReLU(),
                    nn.Dropout(p=0.5),
                    nn.Linear(500, num_classes)
                    )
        else:
            self.FC['fc1'] = nn.Linear(in_dim, 512)
            self.FC['logits'] = nn.Linear(512, num_classes, bias=False)
        for key in self.FC:
            for m in self.FC[key].modules():
                if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.kaiming_normal_(m.weight)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)

    def set_bn_domain(self, domain=0):
        assert(domain < self.num_domains_bn), \
               "The domain id exceeds the range."
        self.bn_domain = domain
        for m in self.modules():
            if isinstance(m, BatchNormDomain):
                m.set_domain(domain)

    def forward(self, x, reverse=False, eta=0.1):
        feat = self.feature_extractor(x).view(-1, self.in_dim)

        to_select = {}
        to_select['feat'] = feat

        x = feat
        x = self.FC['fc1'](x)
        if reverse:
            x = ReverseLayerF.apply(x, eta)
        x = F.normalize(x)
        x = self.FC['logits'](x) / self.temp
        to_select['logits'] = x
        

        #for key in self.FC:
        #    x = self.dropout[key](x)
        #    x = self.FC[key](x)
        #    to_select[key] = x

        to_select['probs'] = F.softmax(x, dim=1)

        return to_select

# +
def danet(num_classes, feature_extractor, fx_pretrained=True,
          frozen=[], dropout_ratio=0.5, state_dict=None,
          fc_hidden_dims=[], num_domains_bn=1, temp=0.05, **kwargs):

    model = DANet(feature_extractor=feature_extractor,
                num_classes=num_classes, frozen=frozen,
                fx_pretrained=fx_pretrained,
                dropout_ratio=dropout_ratio,
                fc_hidden_dims=fc_hidden_dims,
                num_domains_bn=num_domains_bn, temp=temp, **kwargs)

    if state_dict is not None:
        model_utils.init_weights(model, state_dict, num_domains_bn, False)

    return model

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.discriminator = nn.Sequential(
            nn.Linear(in_features=2048, out_features=256),
            nn.ReLU(True),
            nn.Linear(in_features=256, out_features=2)
        )

    def forward(self, input_feature, alpha):
        reversed_input = ReverseLayerF.apply(input_feature, alpha)
        x = self.discriminator(reversed_input)
        # return F.softmax(x, dim=1)
        return x
class Discriminator_WGAN(nn.Module):
    ''' Domain Discriminator '''

    def __init__(self, encoded_dim=2048, num_domains_bn=1, fc_hidden_dims=[]):
        super(Discriminator_WGAN, self).__init__()
        self.encoded_dim = encoded_dim
        self.bn_domain = 0
        self.num_domains_bn = num_domains_bn
        self.num_hidden_layer = len(fc_hidden_dims)
        self.FC = nn.ModuleDict()


        in_dim = self.encoded_dim
        for k in range(self.num_hidden_layer):
            out_dim = fc_hidden_dims[k]
            self.FC[str(k)] = FC_BN_ReLU_Domain(in_dim, out_dim,
                  num_domains_bn)
            in_dim = out_dim

        self.FC['logits'] = nn.Linear(in_dim, 1)


    def forward(self, x):
        for key in self.FC:
            x = self.FC[key](x)
        return x

    def set_bn_domain(self, domain=0):
        assert(domain < self.num_domains_bn), \
               "The domain id exceeds the range."
        self.bn_domain = domain
        for m in self.modules():
            if isinstance(m, BatchNormDomain):
                m.set_domain(domain)
                
            

class Predictor_deep(nn.Module):
    def __init__(self, num_class=64, inc=4096, temp=0.05):
        super(Predictor_deep, self).__init__()
        self.fc1 = nn.Linear(inc, 512)
        self.fc2 = nn.Linear(512, num_class, bias=False)
        self.num_class = num_class
        self.temp = temp

    def forward(self, x, reverse=False, eta=0.1):
        x = self.fc1(x)
        if reverse:
            x = grad_reverse(x, eta)
        x = F.normalize(x)
        x_out = self.fc2(x) / self.temp
        return x_out

