from torchvision import transforms as T
import torch 

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD  = [0.229, 0.224, 0.225]

def em_seg_transforms(img_size=(308,308)):
    return T.Compose([
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

def denorm_imagenet(x: torch.Tensor) -> torch.Tensor:
    """
    x: (B, C, H, W) normalized with ImageNet stats.
    Returns a de-normalized tensor in [approximately 0..1] range (clamp later).
    """
    if x.dim() != 4:
        raise ValueError(f"Expected (B,C,H,W), got {tuple(x.shape)}")
    if x.size(1) != 3:
        # nothing to do for non-RGB; return as-is
        return x
    mean = x.new_tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std  = x.new_tensor(IMAGENET_STD).view(1, 3, 1, 1)
    return x * std + mean
