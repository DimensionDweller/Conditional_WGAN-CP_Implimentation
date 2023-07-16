# Conditional Wasserstein GAN with Gradient Penalty (WGAN-GP) for Animal Face Generation

This project presents a Conditional Wasserstein Generative Adversarial Network with Gradient Penalty (WGAN-GP), trained on an animal faces dataset for 50 epochs. The model can generate novel images of different animal faces, demonstrating the impressive capability of WGANs for generating diverse and high-quality images.

## Table of Contents

- [Background](#background)
- [Project Description](#project-description)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Usage](#usage)
- [Future Work](#future-work)

## Background

Generative Adversarial Networks (GANs) are powerful generative models that can learn to generate new samples from the same distribution as the training data. Wasserstein GANs with Gradient Penalty (WGAN-GP) are a type of GAN that uses the Wasserstein distance and a gradient penalty to stabilize the training process and generate higher-quality images. This project uses a WGAN-GP to generate novel animal face images. 

![image](https://github.com/YourUsername/WGAN-GP_AnimalFaces/assets/XXXXXX/XXXXXX)

## Project Description

The model is trained on an animal faces dataset, which contains images of various animal faces. The training was performed for 50 epochs.

This project includes an interesting feature:
- A conditional model architecture that allows generation of images of specific classes of animals by conditioning the generator and discriminator on the class labels. 

## Model Architecture

The architecture of the conditional WGAN-GP consists of a Generator and a Discriminator (or Critic), both of which are conditioned on the class labels of the images.

### Generator
The Generator takes a noise vector and a class label as input, and generates an image of the corresponding class. The class label is embedded and concatenated with the noise vector to condition the generation process.

### Discriminator
The Discriminator takes an image and a class label as input, and outputs a single scalar value that indicates whether the image is real or fake. The class label is embedded and concatenated with the image to condition the discrimination process.

The generator and discriminator architectures consist of a series of convolutional and transposed convolutional layers, respectively, with batch normalization and leaky ReLU activation functions.

## Loss Function

The loss function used for training the WGAN-GP is composed of three terms:

1. **Wasserstein loss**: This is the difference between the average discriminator outputs for real and fake images. It encourages the discriminator to distinguish between real and fake images, and the generator to generate realistic images. Mathematically, the Wasserstein loss can be represented as follows:

   $$
   L_{\text{Wasserstein}} = E_{x \sim P_{\text{real}}} [D(x)] - E_{z \sim P_z} [D(G(z))]
   $$

   where \(E\) is the expectation, \(x\) are the real images, \(P_{\text{real}}\) is the data distribution, \(z\) are the noise samples, \(P_z\) is the noise distribution, \(D\) is the discriminator, and \(G\) is the generator.

2. **Gradient penalty**: This is a penalty term that encourages the gradients of the discriminator outputs with respect to the images to have a norm of 1. This helps to enforce the Lipschitz constraint and stabilize the training process. The gradient penalty is defined as:

   $$
   L_{\text{GP}} = E_{\hat{x} \sim P_{\hat{x}}} \left[ \left( ||\nabla_{\hat{x}} D(\hat{x})||_2 - 1 \right)^2 \right]
   $$

   where $\(\hat{x}\)$ are the interpolated samples between real and generated images, \(P_{\hat{x}}\) is the interpolated distribution, and \(\nabla_{\hat{x}}\) is the gradient with respect to \(\hat{x}\).

3. **Auxiliary classifier loss**: This is the cross-entropy loss between the predicted and true class labels. It encourages the discriminator to correctly classify the images, and the generator to generate images of the correct class. The auxiliary classifier loss is given by:

   $$
   L_{\text{AC}} = E_{x \sim P_{\text{real}}, y \sim P_y} [-y \log(D_y(x)) - (1-y) \log(1-D_y(x))]
   $$

   where \(y\) are the class labels, \(P_y\) is the class distribution, and \(D_y\) is the discriminator's classification output.

The total loss for the generator and the discriminator is a weighted sum of these three terms:

$$
L_{\text{Generator}} = -L_{\text{Wasserstein}} + \lambda L_{\text{AC}}
$$

$$
L_{\text{Discriminator}} = L_{\text{Wasserstein}} + \lambda_{\text{GP}} L_{\text{GP}} + \lambda L_{\text{AC}}
$$

where \(\lambda\) and \(\lambda_{\text{GP}}\) are the weighting factors for the auxiliary classifier loss and the gradient penalty, respectively.

Please note that for rendering in GitHub, the LaTex needs to be surrounded by single dollar signs for inline equations and double dollar signs for standalone equations. The updated equations are given below:

1. **Wasserstein loss**: 

   $$L_{\text{Wasserstein}} = E_{x \sim P_{\text{real}}} [D(x)] - E_{z \sim P_z} [D(G(z))]$$

2. **Gradient penalty**: 

   $$L_{\text{GP}} = E_{\hat{x} \sim P_{\hat{x}}} \left[ \left( ||\nabla_{\hat{x}} D(\hat{x})||_2 - 1 \right)^2 \right]$$

3. **Auxiliary classifier loss**: 

   $$L_{\text{AC}} = E_{x \sim P_{\text{real}}, y \sim P_y} [-y \log(D_y(x)) - (1-y) \log(1-D_y(x))]$$

The total loss for the generator and the discriminator is a weighted sum of these three terms:

$$L_{\text{Generator}} = -L_{\text{Wasserstein}} + \lambda L_{\text{AC}}$$

$$L_{\text{Discriminator}} = L_{\text{Wasserstein}} + \lambda_{\text{GP}} L_{\text{GP}} + \lambda L_{\text{AC}}$$


This architecture was chosen because the WGAN-GP algorithm has been shown to be effective for training GANs, and the conditional model architecture allows the generation of images of specific classes.

## Results

![image](https://github.com/YourUsername/WGAN-GP_AnimalFaces/assets/XXXXXX/XXXXXX)

## Usage

Firstly, clone the repository to your local machine using the following command:

```bash
git clone https://github.com/YourUsername/WGAN-GP_AnimalFaces.git
```

Navigate to the directory of the project:

```bash
cd WGAN-GP_AnimalFaces
```

Install the necessary Python packages:

```bash
pip install -r requirements.txt
```

Run the following command to start training the model:

```bash
python main.py
```

You can adjust the hyperparameters of the model by modifying the `main.py` file.

Please note that you may need a machine with a GPU to train the model in a reasonable amount of time. The code is set up to use a GPU if one is available, and will otherwise fall back to using a CPU.

## Future Work and Conclusion

Generative Adversarial Networks (GANs) are powerful tools for generating new data, and the Wasserstein GAN with Gradient Penalty (WGAN-GP) algorithm provides a stable and effective approach for training GANs. The use of a conditional model architecture in this project allows the generation of images of specific classes, providing control over the type of images generated.

In future work, it would be interesting to explore other conditioning methods, such as the Projection Discriminator or the Auxiliary Classifier GAN (AC-GAN). Furthermore, other types of data, such as text or audio, could also be generated using a similar approach.

To conclude, while GANs can be challenging to train, the WGAN-GP algorithm and the use of conditioning provide effective solutions to these challenges. With these tools, GANs can be used to generate diverse and high-quality images.
