import cv2 as cv
import numpy as np
import torch

def is_tensor(data):
    return isinstance(data, torch.Tensor)


def empty(shape, dtype=torch.float32, device='cpu'):
    return torch.empty(shape, dtype=dtype, device=device)


def get_device(device=None):
    if device is not None:
        return torch.device(device)

    if torch.cuda.is_available():
        return torch.device('cuda')

    if torch.backends.mps.is_available():
        return torch.device('mps')

    return torch.device('cpu')


def numpy_to_tensor(data, device='cpu', dtype=None):
    array = np.ascontiguousarray(data)
    tensor = torch.from_numpy(array)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor.to(device)


def tensor_to_numpy(data, dtype=None):
    if not is_tensor(data):
        return data

    array = data.detach().cpu().numpy()
    if dtype is not None:
        array = array.astype(dtype, copy=False)
    return array


def to_numpy_bgr(data, input_type='numpy'):
    if input_type == 'numpy':
        if data.ndim == 2:
            return cv.cvtColor(data, cv.COLOR_GRAY2BGR)

        return cv.cvtColor(data, cv.COLOR_RGB2BGR)
    else:
        return image_to_cv(data)


def image_to_tensor(data, device='cpu'):
    if data.ndim == 2:
        img_rgb = cv.cvtColor(data, cv.COLOR_GRAY2RGB)
    elif data.ndim == 3 and data.shape[2] == 3:
        img_rgb = cv.cvtColor(data, cv.COLOR_BGR2RGB)
    else:
        img_rgb = data

    tensor = torch.from_numpy(np.ascontiguousarray(img_rgb)).to(torch.float32)

    if tensor.ndim == 3 and tensor.shape[-1] in (1, 3):
        tensor = tensor.permute(2, 0, 1)

    return (tensor / 255.0).to(device)


def image_to_cv(data):
    img_numpy = data.detach().cpu().numpy()
    if img_numpy.ndim == 2:
        img_numpy = img_numpy[:, :, np.newaxis]
    if img_numpy.shape[0] in [1, 3]:
        img_numpy = np.transpose(img_numpy, (1, 2, 0))
    if img_numpy.max() <= 1.0:
        img_numpy = img_numpy * 255
    img_numpy = img_numpy.astype(np.uint8)
    if img_numpy.shape[2] == 3:
        img_opencv = cv.cvtColor(img_numpy, cv.COLOR_RGB2BGR)
    else:
        img_opencv = img_numpy

    return img_opencv


def features_to_tensor(data, device='cpu'):
    keypoints = data.get('keypoints')
    descriptors = data.get('descriptors')
    scores = data.get('scores')

    if keypoints is not None:
        keypoints = cv.KeyPoint_convert(keypoints)
        keypoints = np.asarray(keypoints, dtype=np.float32)
        if keypoints.size == 0:
            keypoints = np.empty((0, 2), dtype=np.float32)

    if descriptors is None:
        descriptors = np.empty((len(keypoints), 0), dtype=np.float32)
    else:
        descriptors = np.asarray(descriptors)

    if scores is None:
        scores = np.empty((len(keypoints),), dtype=np.float32)
    else:
        scores = np.asarray(scores)

    result = {
        'keypoints': numpy_to_tensor(keypoints, device=device),
        'descriptors': numpy_to_tensor(descriptors, device=device),
        'scores': numpy_to_tensor(scores, device=device)
    }

    if 'width' in data:
        result['width'] = data['width']
    if 'height' in data:
        result['height'] = data['height']

    return result


def features_to_cv(data):
    keypoints = data.get('keypoints')
    descriptors = data.get('descriptors')
    scores = data.get('scores')

    if keypoints is not None:
        keypoints_np = keypoints.detach().cpu().numpy()

        if keypoints_np.ndim == 3:
            keypoints_np = keypoints_np.reshape(-1, 2)

        keypoints_np = keypoints_np.astype(np.float32)
        keypoints = cv.KeyPoint_convert(keypoints_np)
        keypoints = np.array(keypoints, dtype=object)

    if descriptors is not None:
        descriptors = descriptors.detach().cpu().numpy()

    if scores is not None:
        scores = scores.detach().cpu().numpy()

    result = {'keypoints': keypoints, 'descriptors': descriptors, 'scores': scores}

    if 'width' in data:
        result['width'] = data['width']
    if 'height' in data:
        result['height'] = data['height']

    return result


def matches_to_tensor(data, device='cpu'):
    matches = data.get('matches')

    if not matches:
        matches_np = np.empty((0, 2), dtype=np.int64)
    else:
        matches_np = np.asarray([[match.queryIdx, match.trainIdx] for match in matches],
                                dtype=np.int64).reshape(-1, 2)
    return {'matches': numpy_to_tensor(matches_np, device=device, dtype=torch.long)}


def matches_to_cv(data):
    matches = data.get('matches')
    scores = data.get('scores')

    if matches is not None:
        matches_np = matches.detach().cpu().numpy()
        dmatches = []
        for query_idx, train_idx in matches_np:
            dmatch = cv.DMatch()
            dmatch.queryIdx = int(query_idx)
            dmatch.trainIdx = int(train_idx)
            dmatches.append(dmatch)

        matches = np.asarray(dmatches)

    if scores is not None:
        scores = scores.detach().cpu().numpy()
        scores = np.asarray(scores)

    output = {'matches': matches, 'scores': scores}

    if 'keypoints0' in data:
        output['keypoints0'] = features_to_cv({'keypoints': data['keypoints0']}).get('keypoints')
    if 'keypoints1' in data:
        output['keypoints1'] = features_to_cv({'keypoints': data['keypoints1']}).get('keypoints')
    return output
