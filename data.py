import torchvision.transforms as transforms
import torchvision.datasets as dset

SCALE_FACTOR = 4
HR_SIZE = 256


def create_train_transforms(scale_factor, hr_size, random_crop_size):
    return (
        transforms.Compose(
            [
                transforms.Resize(hr_size),
                transforms.RandomCrop(random_crop_size),
                transforms.ToTensor(),
            ]
        ),
        transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize(random_crop_size // scale_factor),
                transforms.ToTensor(),
            ]
        ),
    )


def create_test_transforms(scale_factor, hr_size):
    return (
        transforms.Compose([transforms.Resize(hr_size), transforms.ToTensor()]),
        transforms.Compose(
            [transforms.Resize(hr_size // scale_factor), transforms.ToTensor()]
        ),
    )


def create_vis_transforms(scale_factor, hr_size):
    return (
        transforms.Compose([transforms.Resize(hr_size), transforms.ToTensor()]),
        transforms.Compose(
            [transforms.Resize(hr_size // scale_factor), transforms.ToTensor()]
        ),
        transforms.Compose(
            [
                transforms.Resize(hr_size // scale_factor),
                transforms.Resize(hr_size),
                transforms.ToTensor(),
            ]
        ),
    )


class TrainDataset(dset.ImageFolder):
    def __init__(self, root, scale_factor, hr_size, random_crop_size):
        hr_transform, lr_transform = create_train_transforms(
            scale_factor, hr_size, random_crop_size
        )
        super(TrainDataset, self).__init__(root, transform=hr_transform)
        self.lr_transform = lr_transform
        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        try:
            sample = self.loader(path)
        except:
            print(f"Faulty image: {path}")
            raise
        if self.transform is not None:
            hr_sample = self.transform(sample)
        if self.lr_transform is not None:
            lr_sample = self.lr_transform(hr_sample)
        return hr_sample, lr_sample


class TestDataset(dset.ImageFolder):
    def __init__(self, root, scale_factor, hr_size):
        hr_transform, lr_transform = create_test_transforms(scale_factor, hr_size)
        super(TestDataset, self).__init__(root, transform=hr_transform)
        self.lr_transform = lr_transform
        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        try:
            sample = self.loader(path)
        except:
            print(f"Faulty image: {path}")
            raise
        if self.transform is not None:
            hr_sample = self.transform(sample)
        if self.lr_transform is not None:
            lr_sample = self.lr_transform(sample)
        return hr_sample, lr_sample


class VisDataset(dset.ImageFolder):
    def __init__(self, root, scale_factor, hr_size):
        hr_transform, lr_transform, simple_sr_transform = create_vis_transforms(
            scale_factor, hr_size
        )
        super(VisDataset, self).__init__(root, transform=hr_transform)
        self.lr_transform = lr_transform
        self.simple_sr_transform = simple_sr_transform
        self.imgs = self.samples

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, _ = self.samples[index]
        try:
            sample = self.loader(path)
        except:
            print(f"Faulty image: {path}")
            raise
        if self.transform is not None:
            hr_sample = self.transform(sample)
        if self.lr_transform is not None:
            lr_sample = self.lr_transform(sample)
        if self.simple_sr_transform is not None:
            simple_sr_sample = self.simple_sr_transform(sample)
        return hr_sample, lr_sample, simple_sr_sample
