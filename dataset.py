class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = sorted([class_name for class_name in os.listdir(root_dir) if class_name != ".ipynb_checkpoints"])
        self.class_to_idx = {class_name: idx for idx, class_name in enumerate(self.classes)}
        self.samples = self.make_dataset()

    def make_dataset(self):
        samples = []
        for class_name in self.classes:
            class_dir = os.path.join(self.root_dir, class_name)
            if os.path.isdir(class_dir):
                for file_name in os.listdir(class_dir):
                    file_path = os.path.join(class_dir, file_name)
                    if os.path.isfile(file_path):
                        samples.append((file_path, self.class_to_idx[class_name]))
        return samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        image_path, label = self.samples[index]
        image = Image.open(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, label
