from screeninfo import get_monitors
import cv2 as cv


def read_image(path):
    if path is None:
        raise ValueError('Empty path to the image')
    if not path.exists():
        raise ValueError('Incorrect path to the image')
    image = cv.imread(str(path))
    if image is None:
        raise ValueError(f'Failed to read image from {path}')

    return image


def save_image(img, save_path):
    if img is None:
        raise ValueError('Empty image')
    if save_path is None:
        raise ValueError('Empty path to save')
    save_path.parent.mkdir(parents=True, exist_ok=True)
    success = cv.imwrite(str(save_path), img)

    return success


def show_image(img, title='Result'):
    if img is None:
        raise ValueError('Empty image to show')

    img_to_show = img
    img_height, img_width = img_to_show.shape[:2]

    monitors = get_monitors()
    win_width = min(img_width, monitors[0].width)
    win_height = min(img_height, monitors[0].height)

    cv.namedWindow(title, cv.WINDOW_NORMAL)
    cv.resizeWindow(title, win_width, win_height)
    cv.imshow(title, img_to_show)
    cv.waitKey(0)
    cv.destroyAllWindows()
