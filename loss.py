import torch
from torch import nn
from torch.nn import functional as F
from models.gan import NLayerDiscriminator


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


class AdversarialLoss(nn.Module):
    def __init__(self):
        super(AdversarialLoss, self).__init__()
        self.discriminator = NLayerDiscriminator(
            input_nc=3,
            n_layers=3,
            use_actnorm=False,
            ndf=64
        ).apply(weights_init)

    def adopt_weight(self, weight, global_step, threshold=0, value=0.):
        if global_step < threshold:
            weight = value
        return weight

    def hinge_d_loss(self, logits_real, logits_fake):
        loss_real = torch.mean(F.relu(1. - logits_real))
        loss_fake = torch.mean(F.relu(1. + logits_fake))
        d_loss = 0.5 * (loss_real + loss_fake)
        return d_loss

    def calculate_adaptive_weight(self, nll_loss, g_loss, last_layer=None):
        if last_layer is not None:
            nll_grads = torch.autograd.grad(nll_loss, last_layer, retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, last_layer, retain_graph=True)[0]
        else:
            nll_grads = torch.autograd.grad(nll_loss, self.last_layer[0], retain_graph=True)[0]
            g_grads = torch.autograd.grad(g_loss, self.last_layer[0], retain_graph=True)[0]

        d_weight = torch.norm(nll_grads) / (torch.norm(g_grads) + 1e-4)
        d_weight = torch.clamp(d_weight, 0.0, 1e4).detach()
        d_weight = d_weight * self.discriminator_weight
        return d_weight

    def forward(self, fake, real, optimizer_idx):
        # fake/real: batch(6) x 3 x w x w
        # now the GAN part
        if optimizer_idx == 0:
            # generator update
            logits_fake = self.discriminator(fake)
            g_loss = -torch.mean(logits_fake)
            return g_loss

        if optimizer_idx == 1:
            # second pass for discriminator update
            logits_real = self.discriminator(real.detach())
            logits_fake = self.discriminator(fake.contiguous().detach())

            d_loss = self.hinge_d_loss(logits_real, logits_fake)

            return d_loss