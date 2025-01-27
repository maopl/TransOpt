import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from transopt.benchmark.TriLevelOpt.models import WideResNet




class QFunc(torch.nn.Module):
    '''Control variate for RELAX'''

    def __init__(self, num_latents, hidden_size=100):
        super(QFunc, self).__init__()
        self.h1 = torch.nn.Linear(num_latents, hidden_size)
        self.nonlin = torch.nn.Tanh()
        self.out = torch.nn.Linear(hidden_size, 1)

    def forward(self, p, w):
        # the multiplication by 2 and subtraction is from toy.py...
        # it doesn't change the bias of the estimator, I guess
        # print(p, w)
        z = torch.cat([p, w.unsqueeze(dim=-1)], dim=-1)
        z = z.reshape(-1)
        # print(z)
        z = self.h1(z * 2. - 1.)
        # print(z)
        z = self.nonlin(z)
        # print(z)
        z = self.out(z)
        # print(z)
        return z


class DifferentiableAugment(nn.Module):
    def __init__(self, sub_policy):
        super(DifferentiableAugment, self).__init__()
        self.sub_policy = sub_policy

    def forward(self, origin_images, probability_b, magnitude):
        images = origin_images
        adds = 0
        for i in range(len(self.sub_policy)):
            if probability_b[i].item() != 0.0:
                images = images - magnitude[i]
                adds = adds + magnitude[i]
        images = images.detach() + adds
        return images

    def parameters(self):
        pass


class MixedAugment(nn.Module):
    def __init__(self, sub_policies):
        super(MixedAugment, self).__init__()
        self.sub_policies = sub_policies
        self._compile(sub_policies)

    def _compile(self, sub_polices):
        self._ops = nn.ModuleList()
        self._nums = len(sub_polices)
        for sub_policy in sub_polices:
            ops = DifferentiableAugment(sub_policy)
            self._ops.append(ops)

    def forward(self, origin_images, probabilities_b, magnitudes, weights_b):
        return self._ops[weights_b.item()](origin_images, probabilities_b[weights_b.item()], magnitudes[weights_b.item()])

    

class Network(nn.Module):
    def __init__(
            self,
            model_name,
            input_shape,
            num_classes,
            model_size,
            dropout_rate,
            sub_policies,
            temperature):
        super(Network, self).__init__()
        self.model_name = model_name
        self.input_shape = input_shape
        self.num_classes = num_classes
        self.model_size = model_size
        self.dropout_rate = dropout_rate
        self.sub_policies = sub_policies
        self.use_cuda = True if torch.cuda.is_available() else False
        self.temperature = torch.tensor(temperature)
        if self.use_cuda:
            self.temperature = self.temperature.cuda()

        self.mix_augment = MixedAugment(sub_policies)

        self.model = self.create_model()

        self._initialize_augment_parameters()
        self.augmenting = True

    def set_augmenting(self, value):
        assert value in [False, True]
        self.augmenting = value

    def create_model(self):
        return WideResNet(
            self.input_shape,
            self.num_classes,
            model_size=self.model_size,
            dropout_rate=self.dropout_rate)


    def update_temperature(self, value):
        self.temperature.data.sub_(self.temperature.data - value)

    def augment_parameters(self):
        return self._augment_parameters

    def genotype(self):
        def _parse():
            index = torch.argsort(self.ops_weights)
            probabilities = self.probabilities.clamp(0, 1)
            magnitudes = self.magnitudes.clamp(0, 1)
            ops_weights = torch.nn.functional.softmax(self.ops_weights, dim=-1)
            gene = []
            for idx in reversed(index):
                gene += [tuple([(self.sub_policies[idx][k],
                          probabilities[idx][k].data.detach().item(),
                          magnitudes[idx][k].data.detach().item(),
                          ops_weights[idx].data.detach().item()) for k in range(len(self.sub_policies[idx]))])]
            return gene

        return _parse()

    def sample(self):
        EPS = 1e-6
        num_sub_policies = len(self.sub_policies)
        num_ops = len(self.sub_policies[0])
        probabilities_logits = torch.log(self.probabilities.clamp(0.0+EPS, 1.0-EPS)) - torch.log1p(-self.probabilities.clamp(0.0+EPS, 1.0-EPS))
        probabilities_u = torch.rand(num_sub_policies, num_ops).cuda()
        probabilities_v = torch.rand(num_sub_policies, num_ops).cuda()
        probabilities_u = probabilities_u.clamp(EPS, 1.0)
        probabilities_v = probabilities_v.clamp(EPS, 1.0)
        probabilities_z = probabilities_logits + torch.log(probabilities_u) - torch.log1p(-probabilities_u)
        probabilities_b = probabilities_z.gt(0.0).type_as(probabilities_z)
        def _get_probabilities_z_tilde(logits, b, v):
            theta = torch.sigmoid(logits)
            v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
            z_tilde = logits + torch.log(v_prime) - torch.log1p(-v_prime)
            return z_tilde
        probabilities_z_tilde = _get_probabilities_z_tilde(probabilities_logits, probabilities_b, probabilities_v)
        self.probabilities_logits = probabilities_logits
        self.probabilities_b = probabilities_b
        self.probabilities_sig_z = torch.sigmoid(probabilities_z/self.temperature)
        self.probabilities_sig_z_tilde = torch.sigmoid(probabilities_z_tilde/self.temperature)

        ops_weights_p = torch.nn.functional.softmax(self.ops_weights, dim=-1)
        ops_weights_logits = torch.log(ops_weights_p)
        ops_weights_u = torch.rand(num_sub_policies).cuda()
        ops_weights_v = torch.rand(num_sub_policies).cuda()
        ops_weights_u = ops_weights_u.clamp(EPS, 1.0)
        ops_weights_v = ops_weights_v.clamp(EPS, 1.0)
        ops_weights_z = ops_weights_logits - torch.log(-torch.log(ops_weights_u))
        ops_weights_b = torch.argmax(ops_weights_z, dim=-1)
        def _get_ops_weights_z_tilde(logits, b, v):
            theta = torch.exp(logits)
            z_tilde = -torch.log(-torch.log(v)/theta-torch.log(v[b]))
            z_tilde = z_tilde.scatter(dim=-1, index=b, src=-torch.log(-torch.log(v[b])))
            # v_prime = v * (b - 1.) * (theta - 1.) + b * (v * theta + 1. - theta)
            # z_tilde = logits + torch.log(v_prime) - torch.log1p(-v_prime)
            return z_tilde
        ops_weights_z_tilde = _get_ops_weights_z_tilde(ops_weights_logits, ops_weights_b, ops_weights_v)
        self.ops_weights_logits = ops_weights_logits
        self.ops_weights_b = ops_weights_b
        self.ops_weights_softmax_z = torch.nn.functional.softmax(ops_weights_z/self.temperature, dim=-1)
        self.ops_weights_softmax_z_tilde = torch.nn.functional.softmax(ops_weights_z_tilde/self.temperature, dim=-1)
        # print(probabilities_z)
        # print(ops_weights_z)
        # print(probabilities_z_tilde)
        # print(ops_weights_z_tilde)
        # ops_weights_dist = torch.distributions.RelaxedOneHotCategorical(
        #     self.temperature, logits=self.ops_weights)
        #     # self.temperature, torch.nn.functional.softmax(self.ops_weights, dim=-1))
        # sample_ops_weights = ops_weights_dist.rsample()
        # sample_ops_weights = sample_ops_weights.clamp(0.0, 1.0)
        # self.sample_ops_weights_index = torch.max(sample_ops_weights, dim=-1, keepdim=True)[1]
        # one_h = torch.zeros_like(sample_ops_weights).scatter_(-1, self.sample_ops_weights_index, 1.0)
        # self.sample_ops_weights = one_h - sample_ops_weights.detach() + sample_ops_weights
        # print(sample_probabilities)
        # print(self.sample_probabilities_index)
        # print(sample_ops_weights)
        # print(self.sample_ops_weights_index)
        # print(self.sample_ops_weights)

    def forward_train(self, origin_images, probabilities_b, magnitudes, ops_weights_b):
        mix_image = self.mix_augment.forward(
            origin_images, probabilities_b, magnitudes, ops_weights_b)
        output = self.model(mix_image)
        return output

    def forward_test(self, images):
        return self.model(images)

    def forward(self, origin_images, hparams):
        probabilities_b, magnitudes, ops_weights_b  = hparams
        return self.forward_train(origin_images, probabilities_b, magnitudes, ops_weights_b)



    
class DataParameters(nn.Module):
    def __init__(self, sub_policies):
        super(DataParameters, self).__init__()
        self.sub_policies = sub_policies
        num_sub_policies = len(self.sub_policies)
        num_ops = len(self.sub_policies[0])
        with torch.no_grad():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            self.probabilities = Variable(0.5*torch.ones(num_sub_policies, num_ops, device=device), requires_grad=True)
            self.ops_weights = Variable(1e-3*torch.ones(num_sub_policies, device=device), requires_grad=True)
            self.q_func = [QFunc(num_sub_policies*(num_ops+1)).to(device)]
            self.magnitudes = Variable(0.5*torch.ones(num_sub_policies, num_ops, device=device), requires_grad=True)

        self._augment_parameters = [
            self.probabilities,
            self.ops_weights,
            self.magnitudes,
        ]
        self._augment_parameters += [*self.q_func[0].parameters()]

    def parameters(self):
        return self._augment_parameters

    def forward(self):
        return self._augment_parameters
