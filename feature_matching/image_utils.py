from screeninfo import get_monitors
from converter import Converter
import cv2 as cv


def to_numpy_bgr(data, input_type='numpy'):
    if input_type == 'numpy':
        if data.ndim == 2:
            return cv.cvtColor(data, cv.COLOR_GRAY2BGR)

        return cv.cvtColor(data, cv.COLOR_RGB2BGR)
    else:
        converter = Converter.create('image')
        return converter.convert(data, 'tensor', 'opencv')


def read_image(path, input_type='numpy'):
    if path is None:
        raise ValueError('Empty path to the image')
    if not path.exists():
        raise ValueError('Incorrect path to the image')
    image = cv.imread(str(path))
    if image is None:
        raise ValueError(f'Failed to read image from {path}')

    if input_type == 'numpy':
        return image
    else:
        converter = Converter.create('image')
        return converter.convert(image, 'opencv', 'tensor')


def save_image(img, save_path, input_type='numpy'):
    if img is None:
        raise ValueError('Empty image')
    if save_path is None:
        raise ValueError('Empty path to save')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    if input_type == 'numpy':
        success = cv.imwrite(str(save_path), img)
    else:
        converter = Converter.create('image')
        img_np = converter.convert(img, 'tensor', 'opencv')

        success = cv.imwrite(str(save_path), img_np)

    return success


def show_image(img, title='Result', input_type='numpy'):
    if img is None:
        raise ValueError('Empty image to show')

    if input_type == 'numpy':
        img_to_show = img
    else:
        converter = Converter.create('image')
        img_to_show = converter.convert(img, 'tensor', 'opencv')

    img_height, img_width = img_to_show.shape[:2]

    monitors = get_monitors()
    win_width = min(img_width, monitors[0].width)
    win_height = min(img_height, monitors[0].height)

    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.resizeWindow(title, win_width, win_height)
    cv.imshow(title, img_to_show)
    cv.waitKey(0)
    cv.destroyAllWindows()
