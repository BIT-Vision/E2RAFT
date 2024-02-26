import cv2
import numpy as np
from FlowToImage import flow_to_image


def generate_flow_circle(center, height, width):
    x0, y0 = center
    if x0 >= height or y0 >= width:
        raise AttributeError('ERROR')
    flow = np.zeros((height, width, 2), dtype=np.float32)

    grid_x = np.tile(np.expand_dims(np.arange(width), 0), [height, 1])
    grid_y = np.tile(np.expand_dims(np.arange(height), 1), [1, width])

    grid_x0 = np.tile(np.array([x0]), [height, width])
    grid_y0 = np.tile(np.array([y0]), [height, width])

    flow[:,:,0] = grid_x - grid_x0
    flow[:,:,1] = grid_y0 - grid_y

    return flow


if __name__ == "__main__":
    center = [128, 128]
    flow = generate_flow_circle(center, height=257, width=257)

    flow_rgb = flow_to_image(flow)
    flow_rgb = cv2.cvtColor(flow_rgb.transpose(1,2,0), cv2.COLOR_RGB2BGR)

    cv2.imwrite('ColorWheel.png', flow_rgb)
