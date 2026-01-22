import os
import torch
import numpy as np
import cv2
from tqdm import tqdm
from natsort import natsorted
from torchvision import transforms
from torchvision.models.optical_flow import raft_large

# -----------------------------
# Config
# -----------------------------
ROOT_PATH = "/egr/research-sprintai/baliahsa/projects/DeepfakeBench/DeepfakeDatasets/rgb/FaceForensics++/original_sequences"
# SUBFOLDERS = [
#     "DeepFakeDetection/c23",
#     "FaceSwap/c23",
#     "Face2Face/c23",
#     "NeuralTextures/c23",
#     "FaceShifter/c23",
#     "Deepfakes/c23"
# ]
SUBFOLDERS = [
    "actors/c23",
    "youtube/c23"
]

DEVICE = "cuda"
IMG_RESIZE = 256  # Resize frames for faster flow
FP16 = True       # Use mixed precision
PATCH_FLOW = False # If True, downsample flow to CLIP patch size

to_tensor = transforms.ToTensor()

def read_image(path, resize=IMG_RESIZE):
    """Read an image and convert to tensor [C,H,W]"""
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if resize is not None:
        img = cv2.resize(img, (resize, resize))
    return to_tensor(img)  # float32 [0,1]

# -----------------------------
# Load RAFT-Large
# -----------------------------
model = raft_large(pretrained=True).to(DEVICE).eval()

# -----------------------------
# Main loop
# -----------------------------
for subfolder in SUBFOLDERS:
    root = os.path.join(ROOT_PATH, subfolder)
    frames_dir = os.path.join(root, "frames")
    flow_dir = os.path.join(root, "flow")
    os.makedirs(flow_dir, exist_ok=True)

    video_names = natsorted(os.listdir(frames_dir))

    for video in tqdm(video_names, desc=subfolder):
        frame_dir = os.path.join(frames_dir, video)
        if not os.path.isdir(frame_dir):
            continue

        video_flow_dir = os.path.join(flow_dir, video)
        os.makedirs(video_flow_dir, exist_ok=True)

        frames = natsorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])
        if len(frames) < 2:
            continue

        # read first frame
        prev_img = read_image(os.path.join(frame_dir, frames[0])).unsqueeze(0).to(DEVICE)  # [1,3,H,W]

        for i in range(1, len(frames)):
            curr_img = read_image(os.path.join(frame_dir, frames[i])).unsqueeze(0).to(DEVICE)

            # Compute flow
            with torch.no_grad():
                if FP16:
                    with torch.cuda.amp.autocast():
                        flow = model(prev_img, curr_img)
                else:
                    flow = model(prev_img, curr_img)

            # extract tensor if it's returned as a list
            if isinstance(flow, list):
                flow = flow[0]

            # convert to [H,W,2] numpy
            flow_np = flow.squeeze(0).permute(1, 2, 0).cpu().numpy()
            save_name = frames[i].replace(".png", ".npy")
            np.save(os.path.join(video_flow_dir, save_name), flow_np)

            # shift previous image
            prev_img = curr_img