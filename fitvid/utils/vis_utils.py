import numpy as np
import torch
import cv2
import matplotlib.pyplot as plt
from fitvid.utils.depth_utils import depth_to_rgb_im

color_map = {
    0: [0, 0, 0],         # Black
    1: [128, 0, 0],       # Maroon
    2: [0, 128, 0],       # Green
    3: [128, 128, 0],     # Olive
    4: [0, 0, 128],       # Navy
    5: [128, 0, 128],     # Purple
    6: [0, 128, 128],     # Teal
    7: [128, 128, 128],   # Gray
    8: [192, 192, 192],   # Silver
    9: [255, 0, 0],       # Red
    10: [0, 255, 0],      # Lime
    11: [255, 255, 0],    # Yellow
    12: [0, 0, 255],      # Blue
    13: [255, 0, 255],    # Fuchsia
    14: [0, 255, 255],    # Aqua
    15: [255, 255, 255],  # White
    16: [128, 128, 64],   # Yellow Green
    17: [192, 0, 64],     # Scarlet
    18: [64, 128, 192],   # Light Blue
    19: [192, 64, 128]    # Pink
}


def build_visualization(
    gt,
    pred,
    gt_depth=None,
    gt_depth_pred=None,
    pred_depth=None,
    rgb_loss=None,
    depth_loss_weight=None,
    depth_loss=None,
    name="Depth",
):
    tlen = gt.shape[1]

    if gt_depth is None:  # for depth only or no depth prediction case
        cmap = plt.get_cmap("jet_r")
        if gt.shape[-3] == 1:
            gt = np.moveaxis(depth_to_rgb_im(gt.detach().cpu().numpy(), cmap), 4, 2)
            pred = np.moveaxis(depth_to_rgb_im(pred.detach().cpu().numpy(), cmap), 4, 2)
        elif gt.shape[-3] == 20:
            # Convert from one-hot encoding (H x W x 20) to seg img (H x W x 1) 
            gt = gt.detach().cpu().numpy()
            pred = pred.detach().cpu().numpy()
            bs, seq_len, h, w = gt.shape[0], gt.shape[1], gt.shape[-1], gt.shape[-2]
            
            gt = np.transpose(gt, (0, 1, 3, 4, 2))
            pred = np.transpose(pred, (0, 1, 3, 4, 2))
            print("gt.shape, pred.shape: ", gt.shape, pred.shape)
            gt_seg_img = np.zeros((bs, seq_len, h, w), dtype=int)
            pred_seg_img = np.zeros((bs, seq_len, h, w), dtype=int)

            gt_seg_img = np.argmax(gt, axis=-1)
            logSoftmax = torch.nn.LogSoftmax(dim=-1)
            pred_torch = torch.from_numpy(pred)
            out = logSoftmax(pred_torch).numpy()
            pred_seg_img = np.argmax(out, axis=-1)

            # for b in range(bs):
            #     for s in range(seq_len):
            #         for i in range(h):
            #             for j in range(w):
            #                 # gt_label = np.where(gt[b, s, i, j] == 1.0)
            #                 # # if len(gt_label[0]) == 0:
            #                 # #     print("gt[b, s, i, j]: ",gt[b, s, i, j])
            #                 # #     print("wowwwwwwwwwwwwwwwwwwwwwwwwww")
            #                 # #     continue
            #                 # # print("gt_label: ", gt_label)
            #                 # gt_label = gt_label[0][0]
            #                 # gt_seg_img[b, s, i, j] = gt_label 

            #                 pred_torch = torch.from_numpy(pred[b, s, i, j])
            #                 # print("pred_torch: ", pred_torch)
            #                 logSoftmax = torch.nn.LogSoftmax(dim=0)
            #                 out = logSoftmax(pred_torch).numpy()
            #                 # print("out: ", out)
            #                 # pred_label = np.where(pred[b, s, i, j] == 1.0)[0][0] 
            #                 max_index = np.argmax(out)
            #                 pred_seg_img[b, s, i, j] = max_index 
            # # obs[k] = np.transpose(one_hot_encoded_image, (0, 3, 1, 2))
            gt = gt_seg_img
            pred = pred_seg_img
            print("gt, pred: ", gt.shape, pred.shape)

            # convert segmentation images to rgb images for visualization
            gt_rgb = np.zeros((gt.shape[0], gt.shape[1], gt.shape[2], gt.shape[3], 3))
            pred_rgb = np.zeros((gt.shape[0], gt.shape[1], gt.shape[2], gt.shape[3], 3))

            for class_id, color in color_map.items():
                gt_rgb[gt == class_id] = color
                pred_rgb[pred == class_id] = color
            
            gt_rgb = np.transpose(gt_rgb, (0, 1, 4, 2, 3))
            pred_rgb = np.transpose(pred_rgb, (0, 1, 4, 2, 3))

            # fig, ax = plt.subplots(1,2)
            # ax[0].imshow(gt[0][0])
            # ax[1].imshow(gt_rgb[0][0])
            # plt.show()

        else:
            gt = gt.detach().cpu().numpy() * 255
            pred = pred.detach().cpu().numpy() * 255

        # rgb_loss_im = generate_sample_metric_imgs(rgb_loss.detach().cpu().numpy(), tlen)
        # print("rgb_loss_im: ", rgb_loss_im.shape)
        # image_rows = [gt_rgb, pred_rgb, rgb_loss_im]
        image_rows = [gt_rgb, pred_rgb]
        image_rows = np.concatenate(image_rows, axis=-1)  # create a horizontal row
        image_rows = np.concatenate(image_rows, axis=-2)  # create B rows
        # headers = ["GT", "Pred", "RGB Loss"]
        headers = ["GT", "Pred"]
        text_headers = [generate_text_square(h) for h in headers]
        text_headers = np.concatenate(text_headers, axis=-1)
        text_headers = np.tile(text_headers[None], (tlen, 1, 1, 1))
        image_rows = np.concatenate((text_headers, image_rows), axis=-2)
        return image_rows

    else:
        if name.lower() == "depth":
            # convert depth images to RGB visualizations
            cmap = plt.get_cmap("jet_r")
            gt_depth = np.moveaxis(
                depth_to_rgb_im(gt_depth.detach().cpu().numpy(), cmap), 4, 2
            )
            gt_depth_pred = np.moveaxis(
                depth_to_rgb_im(gt_depth_pred.detach().cpu().numpy(), cmap), 4, 2
            )
            pred_depth = np.moveaxis(
                depth_to_rgb_im(pred_depth.detach().cpu().numpy(), cmap), 4, 2
            )

            # create images for numerical values
            total_loss = (
                depth_loss_weight * depth_loss + (1 - depth_loss_weight) * rgb_loss
            )
            rgb_loss_im = generate_sample_metric_imgs(
                rgb_loss.detach().cpu().numpy(), tlen
            )
            depth_loss_im = generate_sample_metric_imgs(
                depth_loss.detach().cpu().numpy(), tlen
            )
            total_loss_im = generate_sample_metric_imgs(
                total_loss.detach().cpu().numpy(), tlen
            )

            # each shape is [B, T, 3, H, W]
            image_rows = [
                gt.detach().cpu().numpy() * 255,
                pred.detach().cpu().numpy() * 255,
                pred_depth,
                gt_depth,
                gt_depth_pred,
                rgb_loss_im,
                depth_loss_im,
                total_loss_im,
            ]
            image_rows = np.concatenate(image_rows, axis=-1)  # create a horizontal row
            image_rows = np.concatenate(image_rows, axis=-2)  # create B rows

            headers = [
                "GT",
                "Pred",
                f"Pred {name}",
                f"GT {name}",
                f"GT {name[0]} Pred",
                "RGB Loss",
                f"{name[0]} Loss",
                "Total L",
            ]
            text_headers = [generate_text_square(h) for h in headers]
            text_headers = np.concatenate(text_headers, axis=-1)
            # duplicate text headers through time
            text_headers = np.tile(text_headers[None], (tlen, 1, 1, 1))
            image_rows = np.concatenate((text_headers, image_rows), axis=-2)
            return image_rows
        elif name.lower() == "normal":
            # create images for numerical values
            total_loss = (
                depth_loss_weight * depth_loss + (1 - depth_loss_weight) * rgb_loss
            )
            rgb_loss_im = generate_sample_metric_imgs(
                rgb_loss.detach().cpu().numpy(), tlen
            )
            depth_loss_im = generate_sample_metric_imgs(
                depth_loss.detach().cpu().numpy(), tlen
            )
            total_loss_im = generate_sample_metric_imgs(
                total_loss.detach().cpu().numpy(), tlen
            )

            # each shape is [B, T, 3, H, W]
            image_rows = [
                gt.detach().cpu().numpy() * 255,
                pred.detach().cpu().numpy() * 255,
                pred_depth.cpu().detach().numpy() * 255,
                gt_depth.cpu().detach().numpy() * 255,
                gt_depth_pred.cpu().detach().numpy() * 255,
                rgb_loss_im,
                depth_loss_im,
                total_loss_im,
            ]
            image_rows = np.concatenate(image_rows, axis=-1)  # create a horizontal row
            image_rows = np.concatenate(image_rows, axis=-2)  # create B rows

            headers = [
                "GT",
                "Pred",
                f"Pred {name}",
                f"GT {name}",
                f"GT {name[0]} Pred",
                "RGB Loss",
                f"{name[0]} Loss",
                "Total L",
            ]
            text_headers = [generate_text_square(h) for h in headers]
            text_headers = np.concatenate(text_headers, axis=-1)
            # duplicate text headers through time
            text_headers = np.tile(text_headers[None], (tlen, 1, 1, 1))
            image_rows = np.concatenate((text_headers, image_rows), axis=-2)
            return image_rows


def generate_sample_metric_imgs(l, tlen, size=(64, 64)):
    # output should be in shape [B, T, 3, *size]
    squares = []
    for num in l:
        text = "{:.3e}".format(num)  # 3 decimal points in scientific notation
        squares.append(generate_text_square(text, size))
    squares = np.stack(squares)
    squares = np.tile(squares[:, None], (1, tlen, 1, 1, 1))
    return squares


def generate_text_square(text, size=(64, 64), fontscale=2.5):
    img = np.ones(shape=(512, 512, 3), dtype=np.int16)
    cv2.putText(
        img=img,
        text=text,
        org=(50, 250),
        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
        fontScale=fontscale,
        color=(255, 255, 255),
        thickness=3,
    )
    img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)
    img = np.moveaxis(img, 2, 0)
    return img


def save_moviepy_gif(obs_list, name, fps=5):
    from moviepy.editor import ImageSequenceClip

    clip = ImageSequenceClip(obs_list, fps=fps)
    if name[:-4] != ".gif":
        clip.write_gif(f"{name}.gif", fps=fps)
    else:
        clip.write_gif(name, fps=fps)
