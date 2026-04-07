## Dana Tran
## CS 5330 Project 5
## 04/06/26

import sys

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torchvision import transforms

from task1_network import MyNetwork


#take a 28x28 grayscale crop and convert it to the exact tensor format the MNIST network expects
def preprocess_roi(gray_roi_28):
    """gray_roi_28: uint8 [28,28], digit should be dark-on-light or adjust invert."""
    t = torch.from_numpy(gray_roi_28).float().unsqueeze(0).unsqueeze(0) / 255.0
    t = 1.0 - t
    t = transforms.Normalize((0.1307,), (0.3081,))(t)
    
    return t

#open the webcam, grab frames, run the center crop through the network, and overlay the prediction
def main(argv):
    device = torch.device("cpu")
    model = MyNetwork()
    model.load_state_dict(torch.load("mnist_cnn.pt", map_location=device))
    model.eval()

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("could not open camera")
        return 1

    print("press q to quit")
    while True:
        ok, frame = cap.read()
        if not ok:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        crop = gray[y0 : y0 + side, x0 : x0 + side]
        small = cv2.resize(crop, (28, 28), interpolation=cv2.INTER_AREA)

        x = preprocess_roi(small).to(device)
        with torch.no_grad():
            out = model(x)
        pred = int(out.argmax(dim=1).item())

        cv2.putText(
            frame,
            f"pred: {pred}",
            (10, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.2,
            (0, 255, 0),
            2,
        )
        cv2.imshow("digit cam (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return 0

if __name__=="__main__":
    raise SystemExit(main(sys.argv))