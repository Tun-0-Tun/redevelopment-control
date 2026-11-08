import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision


def device():
    if torch.cuda.is_available(): return "cuda"
    if torch.backends.mps.is_available(): return "mps"
    return "cpu"


class UNet(nn.Module):
    """ResNet18-энкодер + UNet-декодер. На входе 3 канала, на выходе n классов (0 = фон)."""
    def __init__(s, n):
        super().__init__()
        r = torchvision.models.resnet18(weights=None)  # веса приходят из нашего чекпойнта, интернет не нужен
        s.l0 = nn.Sequential(r.conv1, r.bn1, r.relu)
        s.pool, s.l1, s.l2, s.l3, s.l4 = r.maxpool, r.layer1, r.layer2, r.layer3, r.layer4
        blk = lambda i, o: nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o), nn.ReLU(True))
        s.d4, s.d3, s.d2, s.d1, s.d0 = blk(768, 256), blk(384, 128), blk(192, 64), blk(128, 64), blk(67, 32)
        s.out = nn.Conv2d(32, n, 1)

    def forward(s, x):
        up = lambda a, b: torch.cat([F.interpolate(a, size=b.shape[2:], mode="bilinear", align_corners=False), b], 1)
        e0 = s.l0(x); e1 = s.l1(s.pool(e0)); e2 = s.l2(e1); e3 = s.l3(e2); e4 = s.l4(e3)
        d = s.d4(up(e4, e3)); d = s.d3(up(d, e2)); d = s.d2(up(d, e1)); d = s.d1(up(d, e0))
        return s.out(s.d0(up(d, x)))


_cache = {}


def load(name, n):
    """Грузит redevelopment_control/weights/<name>.pt один раз на процесс."""
    if name not in _cache:
        from pathlib import Path
        sd = torch.load(Path(__file__).parent / "weights" / f"{name}.pt", map_location="cpu")
        net = UNet(n)
        net.load_state_dict({k: v.float() if v.is_floating_point() else v for k, v in sd.items()})
        _cache[name] = net.to(device()).eval()
    return _cache[name]
