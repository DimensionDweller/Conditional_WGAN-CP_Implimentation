# function to unnormalize images
def unnormalize(img):
    img = img / 2 + 0.5 # unnormalize
    npimg = img.numpy()
    return np.transpose(npimg, (1, 2, 0))


# function to plot images
def plot_images(images):
    fig, axes = plt.subplots(1, len(images), figsize=(len(images), 3))
    for i, img in enumerate(images):
        img = unnormalize(img)
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.show()
