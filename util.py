from PIL import Image
import numpy as np


def load_img(img_path):
    try:
        img = Image.open(img_path)
    except IOError:
        img = None
    return img


def crop_img(img, width=224, height=224):
    size = np.shape(img)
    shape = (height, width, 3)
    if len(size) == 2:
        img = np.tile(img, (1, 1, 3))
    ans = np.zeros(shape)
    if size[0] > height and size[1] > width:
        ans[...] = img[(size[0]-height)/2:(size[0]+height)/2, (size[1]-width)/2:(size[1]+width)/2, ...]
    elif size[0] < height and size[1] < width:
        ans[(height-size[0])/2:(height+size[0])/2, (width-size[1])/2:(width+size[1])/2, ...] = img
    elif size[0] < height:
        ans[(height-size[0])/2:(height+size[0])/2, ...] = img[:, (size[1]-width)/2:(size[1]+width)/2, ...]
    else:
        ans[:, (width-size[1])/2:(width+size[1])/2, ...] = img[(size[0]-height)/2:(size[0]+height)/2, ...]
    return ans


def img_preprocess(img, width=224, height=224, crop=True):
    if crop:  # for image caption
        resized_img = crop_img(np.array(img.resize((256, 256)), dtype=np.float32))
    else:  # for neural style
        resized_img = np.array(img.resize((width, height), Image.BICUBIC), dtype=np.float32)
        if len(resized_img.shape) == 2:
            resized_img = np.tile(resized_img[..., np.newaxis], (1, 1, 3))
    avg_img = np.ones_like(resized_img)
    avg_img[..., 0] *= 123.68
    avg_img[..., 1] *= 116.779
    avg_img[..., 2] *= 103.939
    return (resized_img-avg_img)[np.newaxis, ...]


def img_deprocess(img):
    img = img[0, ...]
    avg_img = np.ones_like(img)
    avg_img[..., 0] *= 123.68
    avg_img[..., 1] *= 116.779
    avg_img[..., 2] *= 103.939
    return Image.fromarray(np.clip(img+avg_img, 0, 255).astype(np.uint8))


def get_batch(paths, st, batch_size=4, width=512, height=512):
    ans = np.empty(shape=(batch_size, height, width, 3))
    for k, j in enumerate(range(st, st+batch_size)):
        ans[k, ...] = img_preprocess(load_img(paths[j]), width, height, crop=False)[0]
    return ans
