import torch.nn as nn
import torch
import math


class ConditioningAugmention(nn.Module):
    def __init__(self, input_dim, emb_dim, device):
        super(ConditioningAugmention, self).__init__()
        self.device = device
        self.input_dim = input_dim
        self.emb_dim = emb_dim

        # TODO: Implement [FC + Activation] x1 with nn.Sequential()

        self.layer = nn.Sequential(
            nn.Linear(input_dim, emb_dim),
            nn.ReLU()
        )


    def forward(self, x):

        x = self.layer(x)

        mu = x[:, :self.emb_dim // 2]
        log_sigma = x[:, self.emb_dim // 2:]
        z = torch.randn(mu.size(0), self.emb_dim//2, device=self.device)
        condition = torch.cat([mu, torch.exp(log_sigma) * z], dim=1)

        return condition, mu, log_sigma

class ImageExtractor(nn.Module):
    def __init__(self, in_chans):
        super(ImageExtractor, self).__init__()
        self.in_chans = in_chans
        self.out_chans = 3

        self.image_net = nn.Sequential(
            nn.ConvTranspose2d(in_chans, self.out_chans, kernel_size=3, stride=1, padding=1),
            nn.Tanh()
        )

    def forward(self, x):

        out = self.image_net(x)

        ################# Problem 2-(b). #################
        return out


class Generator_type_1(nn.Module):
    def __init__(self, in_chans, input_dim):
        super(Generator_type_1, self).__init__()
        self.in_chans = in_chans
        self.input_dim = input_dim

        self.mapping = self._mapping_network()
        self.upsample_layer = self._upsample_network()
        self.image_net = self._image_net()

    def _image_net(self):
        return ImageExtractor(self.in_chans // 16)

    def _mapping_network(self):
        return nn.Sequential(
            nn.Linear(self.input_dim, self.in_chans * 4 * 4),
            nn.BatchNorm1d(self.in_chans * 4 * 4),
            nn.LeakyReLU(),
        )


    def _upsample_network(self):

        return nn.Sequential(
            nn.ConvTranspose2d(self.in_chans, self.in_chans // 2, kernel_size=4, stride=2, padding=1),  # ConvTranspose2d layer
            nn.BatchNorm2d(self.in_chans // 2),  # Batch normalization
            nn.ReLU(),  # ReLU activation function

            nn.ConvTranspose2d(self.in_chans//2, self.in_chans // 4, kernel_size=4, stride=2, padding=1),  # ConvTranspose2d layer
            nn.BatchNorm2d(self.in_chans // 4),  # Batch normalization
            nn.ReLU(),  # ReLU activation function

            nn.ConvTranspose2d(self.in_chans // 4, self.in_chans // 8, kernel_size=4, stride=2, padding=1),  # ConvTranspose2d layer
            nn.BatchNorm2d(self.in_chans // 8),  # Batch normalization
            nn.ReLU(),  # ReLU activation function

            nn.ConvTranspose2d(self.in_chans // 8, self.in_chans // 16, kernel_size=4, stride=2, padding=1),  # ConvTranspose2d layer
            nn.BatchNorm2d(self.in_chans // 16),  # Batch normalization
            nn.ReLU()  # ReLU activation function
        )


    def forward(self, condition, noise):
      
        concat_input = torch.cat([condition, noise], dim=1)
        mapping_input = self.mapping(concat_input)
        mapped_input = mapping_input.view(-1, 1024, 4, 4)

        # Use self.upsample_layer to extract out
        out = self.upsample_layer(mapped_input)

        # Use self.image_net to extract out_image
        out_image = self.image_net(out)

        return out, out_image


class Generator_type_2(nn.Module):
    def __init__(self, in_chans, condition_dim, num_res_layer, device):
        super(Generator_type_2, self).__init__()
        self.device = device

        self.in_chans = in_chans
        self.condition_dim = condition_dim
        self.num_res_layer = num_res_layer

        self.joining_layer = self._joint_conv()
        self.res_layer = nn.ModuleList(
            [self._res_layer() for _ in range(self.num_res_layer)])
        self.upsample_layer = self._upsample_network()
        self.image_net = self._image_net()

    def _image_net(self):
        return ImageExtractor(self.in_chans // 2)

    def _upsample_network(self):

        upsample_network= nn.Sequential(nn.ConvTranspose2d(self.in_chans, self.in_chans//2, kernel_size=4, stride=2, padding=1),
                             nn.BatchNorm2d(self.in_chans//2),
                             nn.ReLU())
        return upsample_network

    def _joint_conv(self):
        
        joining_layer=nn.Sequential(nn.Conv2d(self.condition_dim,self.in_chans, kernel_size=3, stride=1, padding=1),
                                    nn.BatchNorm2d(self.in_chans),
                                    nn.ReLU())

        return joining_layer

    def _res_layer(self):
        return ResModule(self.in_chans)

    def forward(self, condition, prev_output):

        condition = condition.reshape(prev_output.shape[0], 1, 128 //prev_output.shape[3], prev_output.shape[3]) 
        condition = condition.repeat(1, prev_output.shape[1], prev_output.shape[3]//(128 //prev_output.shape[3]), 1)  

        combined_input=torch.cat([prev_output,condition], dim=1)
        out = self.joining_layer(combined_input)

        for res_layer in self.res_layer:
            out = res_layer(out)

        out = self.upsample_layer(out)
        out_image = self.image_net(out)

        return out, out_image


class ResModule(nn.Module):
    def __init__(self, in_chans):
        super(ResModule, self).__init__()
        self.in_chans = in_chans

        self.layer = nn.Sequential(nn.Conv2d(self.in_chans, self.in_chans, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(self.in_chans),
                                   nn.ReLU(),
                                   nn.Conv2d(self.in_chans, self.in_chans, kernel_size=3, padding=1),
                                   nn.BatchNorm2d(self.in_chans)
                                   )

    def forward(self, x):
        
        res_out =self.layer(x)+x

        return res_out


class Generator(nn.Module):
    def __init__(self, text_embedding_dim, projection_dim, noise_input_dim, in_chans, out_chans, num_stage, device):
        super(Generator, self).__init__()
        self.device = device

        self.text_embedding_dim = text_embedding_dim
        self.condition_dim = projection_dim
        self.noise_dim = noise_input_dim
        self.input_dim = self.condition_dim + self.noise_dim
        self.in_chans = in_chans # Ng
        self.out_chans = out_chans

        self.num_stage = num_stage
        self.num_res_layer_type2 = 2  # NOTE: you can change this

        self.condition_aug = self._conditioning_augmentation()
        self.g_layer = nn.ModuleList(
            [self._stage_generator(i) for i in range(self.num_stage)])

    def _conditioning_augmentation(self):

        return ConditioningAugmention(self.text_embedding_dim, self.condition_dim, self.device)

    def _stage_generator(self, i):
        
        if i == 0:
            return Generator_type_1(self.in_chans, self.input_dim)
        else:
            return Generator_type_2(2*self.in_chans//32, self.condition_dim//i, self.num_res_layer_type2, self.device)
            

    def forward(self, text_embedding, noise):

        C_txt, mu, log_sigma = self.condition_aug(text_embedding)

        fake_images = []
        for i in range(self.num_stage):
            generator = self.g_layer[i]
            if i == 0:
                prev_out, fake_image = generator.forward(C_txt, noise)
            else:
                prev_out, fake_image = generator.forward(C_txt, prev_out)
            fake_images.append(fake_image)
        return fake_images, mu, log_sigma
        

class UncondDiscriminator(nn.Module):
    def __init__(self, in_chans, out_chans):
        super(UncondDiscriminator, self).__init__()
        self.in_chans = in_chans
        self.out_chans = out_chans

        self.uncond_layer = nn.Sequential(
            nn.Conv2d(8 * in_chans, 1, kernel_size=4, stride=4, padding=0),
            nn.Flatten()
        )


    def forward(self, x):
        
        uncond_out = self.uncond_layer(x)        

        return uncond_out


class CondDiscriminator(nn.Module):
    def __init__(self, in_chans, condition_dim, out_chans):
        super(CondDiscriminator, self).__init__()
        self.in_chans = in_chans
        self.condition_dim = condition_dim
        self.out_chans = out_chans

        self.cond_layer = nn.Sequential(nn.Conv2d(self.in_chans+self.condition_dim, 8 * self.in_chans, kernel_size=4, stride=1, padding=0),
                                        nn.BatchNorm2d(8 * self.in_chans),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Conv2d(8 * self.in_chans, 1, kernel_size=4, stride=4, padding=0),
                                        nn.Flatten()
                                        )

    def forward(self, x, c):

        changed_c=c.view(-1, self.condition_dim,1,1).repeat(1,1,x.shape[2],x.shape[3])
        combined_input=torch.cat([x,changed_c], dim=1)
        cond_out = self.condlayer(combined_input)

        return cond_out


class AlignCondDiscriminator(nn.Module):
    def __init__(self, in_chans, condition_dim, text_embedding_dim):
        super(AlignCondDiscriminator, self).__init__()
        self.in_chans = in_chans
        self.condition_dim = condition_dim
        self.text_embedding_dim = text_embedding_dim

        self.align_layer = nn.Sequential(nn.Conv2d(self.in_chans+self.condition_dim, 8 * self.in_chans, kernel_size=4, stride=1, padding=0),
                                        nn.BatchNorm2d(8 * self.in_chans),
                                        nn.LeakyReLU(0.2, inplace=True),
                                        nn.Conv2d(8 * self.in_chans, text_embedding_dim, kernel_size=4, stride=4, padding=0),
                                        nn.Flatten()
                                        )

    def forward(self, x, c):

        reshaped_c=c.view(-1, self.condition_dim, 1,1).repeat(1,1,x.shape[2],x.shape[3])
        combined_input=torch.cat([x, reshaped_c], dim=1)
        align_out = self.align_layer(combined_input)
        align_out=align_out.squeeze()

        return align_out


class Discriminator(nn.Module):
    def __init__(self, projection_dim, img_chans, in_chans, out_chans, text_embedding_dim, curr_stage, device):
        super(Discriminator, self).__init__()
        self.condition_dim = projection_dim
        self.img_chans = img_chans
        self.in_chans = in_chans
        self.out_chans = out_chans
        self.text_embedding_dim = text_embedding_dim
        self.curr_stage = curr_stage
        self.device = device

        
        self.global_layer = self._global_discriminator()
        self.prior_layer = self._prior_layer()
        self.prior_layer2 = self._prior_layer2()
        self.uncond_discriminator = self._uncond_discriminator()
        self.cond_discriminator = self._cond_discriminator()
        self.align_cond_discriminator = self._align_cond_discriminator()

    def _global_discriminator(self):

        layers=nn.Sequential(nn.Conv2d(self.img_chans, self.in_chans, kernel_size=4, stride=2, padding=1),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Conv2d(self.in_chans, self.in_chans * 2, kernel_size=4, stride=2, padding=1),
                            nn.BatchNorm2d(self.in_chans*2),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Conv2d(self.in_chans * 2, self.in_chans * 4, kernel_size=4, stride=2, padding=1),
                            nn.BatchNorm2d(self.in_chans * 4),
                            nn.LeakyReLU(0.2, inplace=True),
                            nn.Conv2d(self.in_chans * 4, self.in_chans * 8, kernel_size=4, stride=2, padding=1),
                            nn.BatchNorm2d(self.in_chans * 8),
                            nn.LeakyReLU(0.2, inplace=True)
                            )
        return layers


    def _prior_layer(self):

        layers=[]
        H = 128
        k=int(math.log2(H/64))

        for i in range(k):
            in_channels = self.in_chans * (2 ** (i + 3))  # 전역 레이어의 마지막 채널 수
            layers.extend([
                nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1),  # Double size
                nn.BatchNorm2d(in_channels * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ])

        for i in range(k):
            in_channels = self.in_chans * (2 ** (k - i + 3))  # 마지막 채널 수부터 시작하여 반대로 증가
            layers.extend([
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),  # Halve size
                nn.BatchNorm2d(in_channels // 2),
                nn.LeakyReLU(0.2, inplace=True)
            ])

        return nn.Sequential(*layers)

    def _prior_layer2(self):

        layers=[]
        H = 256
        k=int(math.log2(H/64))

        for i in range(k):
            in_channels = self.in_chans * (2 ** (i + 3))  # 전역 레이어의 마지막 채널 수
            layers.extend([
                nn.Conv2d(in_channels, in_channels * 2, kernel_size=3, stride=2, padding=1),  # Double size
                nn.BatchNorm2d(in_channels * 2),
                nn.LeakyReLU(0.2, inplace=True)
            ])

        for i in range(k):
            in_channels = self.in_chans * (2 ** (k - i + 3))  # 마지막 채널 수부터 시작하여 반대로 증가
            layers.extend([
                nn.Conv2d(in_channels, in_channels // 2, kernel_size=3, padding=1),  # Halve size
                nn.BatchNorm2d(in_channels // 2),
                nn.LeakyReLU(0.2, inplace=True)
            ])

        return nn.Sequential(*layers)
    

    def _uncond_discriminator(self):
        return UncondDiscriminator(self.in_chans, self.out_chans)

    def _cond_discriminator(self):
        return CondDiscriminator(self.in_chans, self.condition_dim, self.out_chans)

    def _align_cond_discriminator(self):
        return AlignCondDiscriminator(self.in_chans, self.condition_dim, self.text_embedding_dim)

    def forward(self,
                img,
                condition=None,  # for conditional loss (mu)
                ):

        global_out = self.global_layer(img)

        if global_out.shape[3] == 4: 
          prior_out = global_out
        elif global_out.shape[3] == 8:
          prior_out = self.prior_layer(global_out)
        else:
          prior_out = self.prior_layer2(global_out)


        if condition is None:
            out = self.uncond_discriminator(prior_out)
            align_out = None
        else:
            cond_out = self.cond_discriminator(prior_out, condition)
            align_out = self.align_cond_discriminator(prior_out, condition)
            out = cond_out + align_out

        out = nn.Sigmoid()(out)

        return out, align_out


def weight_init(layer):
    if (isinstance(layer, nn.Conv2d) or isinstance(layer, nn.ConvTranspose2d)):
        nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)

    elif isinstance(layer, nn.BatchNorm2d):
        nn.init.normal_(layer.weight.data, mean=1.0, std=0.02)
        nn.init.constant_(layer.bias.data, val=0)

    elif isinstance(layer, nn.Linear):
        nn.init.normal_(layer.weight.data, mean=0.0, std=0.02)
        if layer.bias is not None:
            nn.init.constant_(layer.bias.data, val=0.0)
