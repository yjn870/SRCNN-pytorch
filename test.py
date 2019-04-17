import argparse
import os

import torch
import torch.backends.cudnn as cudnn
import numpy as np
import PIL.Image as pil_image

from models import SRCNN
from utils import convert_rgb_to_y, calc_psnr


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights-file', type=str, required=True)
    parser.add_argument('--image-file', type=str, required=True)
    parser.add_argument('--scale', type=int, default=3)
    args = parser.parse_args()

    cudnn.benchmark = True
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    model = SRCNN().to(device)

    state_dict = model.state_dict()
    for n, p in torch.load(args.weights_file, map_location=lambda storage, loc: storage).items():
        if n in state_dict.keys():
            state_dict[n].copy_(p)
        else:
            raise KeyError(n)

    model.eval()

    image = pil_image.open(args.image_file).convert('RGB')

    image_width = (image.width // args.scale) * args.scale
    image_height = (image.height // args.scale) * args.scale
    image = image.resize((image_width, image_height), resample=pil_image.BICUBIC)
    image = image.resize((image.width // args.scale, image.height // args.scale), resample=pil_image.BICUBIC)
    image = image.resize((image.width * args.scale, image.height * args.scale), resample=pil_image.BICUBIC)

    _, cb, cr = image.convert('YCbCr').split()

    image.save(args.image_file.replace('.', '_bicubic_x{}.'.format(args.scale)))

    image = np.array(image).astype(np.float32)
    image = convert_rgb_to_y(image)
    image /= 255.
    image = torch.from_numpy(image).to(device)
    image = image.unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        preds = model(image).clamp(0.0, 1.0)

    psnr = calc_psnr(image, preds)
    print('PSNR: {:.2f}'.format(psnr))

    preds = preds.mul(255.0).byte().cpu().numpy().squeeze(0).squeeze(0)

    y = pil_image.fromarray(preds)
    output = pil_image.merge('YCbCr', (y, cb, cr)).convert('RGB')
    output.save(args.image_file.replace('.', '_srcnn_x{}.'.format(args.scale)))
