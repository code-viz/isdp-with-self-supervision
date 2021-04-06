from torchvision import transforms
from PIL import Image


def normalize(prep_dict):
    if prep_dict['name'] == 'source':
        return transforms.Compose([
            transforms.Resize(prep_dict['resize_size']),
            transforms.CenterCrop(prep_dict['crop_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    if prep_dict['name'] == 'target':
        return transforms.Compose([
            transforms.Resize(prep_dict['resize_size']),
            transforms.CenterCrop(prep_dict['crop_size']),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])
