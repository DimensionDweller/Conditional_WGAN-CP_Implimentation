class Generator(nn.Module):
    """
    This class represents the Generator in the Generative Adversarial Network (GAN). 
    The generator's purpose is to generate images that are indistinguishable from real images.
    
    Attributes
    ----------
    z_dim : int
        Dimension of the latent space (input noise vector).
    num_classes : int
        Number of image classes.
    embedding : torch.nn.Embedding
        Embedding layer used to map the labels to continuous embeddings.
    gen : torch.nn.Sequential
        The main generator network, which is a series of convolutional transpose layers.
    """
    def __init__(self, z_dim=100, num_classes=3, channels_img=3, features_g=64):
        super(Generator, self).__init__()
        self.z_dim = z_dim
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, num_classes)  # Add an embedding layer
        self.gen = nn.Sequential(
            # Each block upsamples the spatial resolution of the input
            self._block(z_dim + num_classes, features_g * 16, 4, 1, 0),  # 4x4
            self._block(features_g * 16, features_g * 8, 4, 2, 1),  # 8x8
            self._block(features_g * 8, features_g * 4, 4, 2, 1),  # 16x16
            self._block(features_g * 4, features_g * 2, 4, 2, 1),  # 32x32
            nn.ConvTranspose2d(features_g * 2, channels_img, kernel_size=4, stride=2, padding=1),  # 64x64
            nn.Tanh(),
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        A private method that returns a sequential block for the generator.
        """
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels, affine=False),
            nn.LeakyReLU(0.2),
           # nn.Dropout(0.5),
        )

    def forward(self, z, labels):
        """
        Forward pass of the generator.
        
        Parameters
        ----------
        z : torch.Tensor
            A noise tensor with shape (batch_size, z_dim).
        labels : torch.Tensor
            A tensor containing the labels of the images.

        Returns
        -------
        out : torch.Tensor
            A batch of generated images.
        """
        labels = self.embedding(labels).unsqueeze(2).unsqueeze(3)  # Use the embedding layer
        x = torch.cat([z, labels], dim=1)
        return self.gen(x)


class Discriminator(nn.Module):
    """
    This class represents the Discriminator (or the Critic in the case of WGAN-GP) in the GAN. 
    The discriminator's purpose is to differentiate between real and generated images.
    
    Attributes
    ----------
    num_classes : int
        Number of image classes.
    embedding : torch.nn.Embedding
        Embedding layer used to map the labels to continuous embeddings.
    disc : torch.nn.Sequential
        The main discriminator network, which is a series of convolutional layers.
    """
    def __init__(self, num_classes=3, channels_img=3, features_d=64):
        super(Discriminator, self).__init__()
        self.num_classes = num_classes
        self.embedding = nn.Embedding(num_classes, num_classes)  # Add an embedding layer
        self.disc = nn.Sequential(
            # Each block reduces the spatial resolution of the input
            nn.Conv2d(channels_img + num_classes, features_d, kernel_size=4, stride=2, padding=1),  # 32x32
            nn.LeakyReLU(0.2),
            self._block(features_d, features_d * 2, 4, 2, 1),  # 16x16
            self._block(features_d * 2, features_d * 4, 4, 2, 1),  # 8x8
            self._block(features_d * 4, features_d * 8, 4, 2, 1),  # 4x4
            nn.Conv2d(features_d * 8, num_classes + 1, kernel_size=4, stride=2, padding=0),  # 1x1
        )

    def _block(self, in_channels, out_channels, kernel_size, stride, padding):
        """
        A private method that returns a sequential block for the discriminator.
        """
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=False),
            nn.BatchNorm2d(out_channels, affine=True),
            nn.LeakyReLU(0.2),
            nn.Dropout2d(0.5),
        )

    def forward(self, x, labels):
        """
        Forward pass of the discriminator.
        
        Parameters
        ----------
        x : torch.Tensor
            A tensor containing a batch of images.
        labels : torch.Tensor
            A tensor containing the labels of the images.

        Returns
        -------
        real_fake : torch.Tensor
            A batch of outputs representing whether the images are real or fake.
        classes : torch.Tensor
            A batch of outputs representing the classes of the images.
        """
        labels = self.embedding(labels).view(labels.shape[0], self.num_classes, 1, 1)  # Use the embedding layer
        x = torch.cat([x, labels * torch.ones([x.shape[0], self.num_classes, x.shape[2], x.shape[3]], device=x.device)], dim=1)
        out = self.disc(x)
        real_fake, classes = out[:, -1], out[:, :-1]  # Separate the outputs
        return real_fake, classes
