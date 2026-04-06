## Dana Tran
## CS 5330 Project 1
## 04/06/26

import sys
from pathlib import Path
import torch
import torchvision.transforms as T
from PIL import Image
from task1_network import MyNetwork

def load_model(weights_path: str="mnist_cnn.pt"):
    model = MyNetwork()
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def image_to_batch(path: Path, invert: bool):
    img = Image.open(path).convert("L")
    resize = T.Resize((28, 28), interpolation=T.InterpolationMode.LANCZOS)
    to_tensor = T.ToTensor()
    t = to_tensor(resize(img))
    if invert:
        t = 1.0 - t
    t = T.Normalize((0.1307,), (0.3081,))(t)
    return t.unsqueeze(0)


def main(argv):
    if len(argv) < 2:
        print("Usage: python task1_handwritten.py <folder> [--no-invert]")
        print("  folder: images of digits (e.g. 0.png .. 9.png)")
        return 1
    folder = Path(argv[1])
    invert = True
    if "--no-invert" in argv:
        invert = False
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
    paths = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in exts
    )
    if not paths:
        print(f"No images found in{folder}")
        return 1
    model = load_model()
    print(f"invert={invert}(black-on-white photos usually need invert=True)\n")
    for p in paths:
        batch = image_to_batch(p, invert=invert)
        with torch.no_grad():
            out = model(batch)
        pred = int(out.argmax(dim=1).item())
        vals = ", ".join(f"{v:6.2f}"for v in out[0].tolist())
        print(f"{p.name}: pred={pred}  log_softmax=[{vals}]")
    return 0


if __name__== "__main__":
    raise SystemExit(main(sys.argv))
