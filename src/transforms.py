from torchvision import transforms


def build_transforms(image_size: int, train: bool) -> transforms.Compose:
    normalize = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    if train:
        return transforms.Compose(
            [
                transforms.Resize(int(image_size * 1.1)),
                transforms.RandomResizedCrop(image_size, scale=(0.85, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.1),
                transforms.ToTensor(),
                normalize,
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 1.1)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalize,
        ]
    )
