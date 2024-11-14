import torch
import cv2
import numpy as np

# Load MiDaS model
#midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small")
#midas_transforms = torch.hub.load("intel-isl/MiDaS", "transforms").small_transform

#device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#midas.to(device)
#midas.eval()

def estimate_depth(image):
    # Preprocess the image and predict the depth map
    input_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #input_image = midas_transforms(input_image).to(device)

    with torch.no_grad():
        prediction = midas(input_image)

        # Resize prediction to original image size
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=image.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    depth_map = prediction.cpu().numpy()
    return depth_map

# Function to estimate distance using the bounding box height
def estimate_distance(focal_length, real_height, pixel_height):
    if pixel_height == 0:  # To avoid division by zero
        return float('inf')
    return (focal_length * real_height) / pixel_height

def pixel_to_world(x_pixel, y_pixel, focal_length, distance, frame_width, frame_height):
    scale_x = (distance / focal_length) * frame_width
    scale_y = (distance / focal_length) * frame_height


    world_x = (x_pixel - (frame_width / 2)) * scale_x
    world_y = (y_pixel - (frame_height / 2)) * scale_y

    return world_x, world_y
