import scipy
import numpy as np
import pydicom
from glob import glob
from pydicom.pixel_data_handlers.util import apply_voi_lut


class DataLoader:
    def __init__(self, img_res=(128, 128)):
        self.img_res = img_res

    def load_data(self, domain, batch_size=1, is_testing=False):
        data_type = "train_%s" % domain if not is_testing else "test_%s" % domain
        path = glob('./data/%s/%s/*' % (data_type))

        batch_images = np.random.choice(path, size=batch_size)

        imgs = []
        for i, img_path in enumerate(batch_images):
            if domain == 'kVCT':
                img = self.read_dicom(img_path)
            else:
                img = np.load(img_path)
            img = scipy.misc.imresize(img, self.img_res)  # resizing 부분
            img = 2*(img-np.min(img)) / (np.max(img) - np.min(img)) - 1  # [-1, 1] normalization
            imgs = np.append(img, i)

        imgs = np.array(imgs)

        return imgs

    def load_batch(self, batch_size=1, is_testing=False):
        data_type = "train" if not is_testing else "val"
        path_kVCT = glob('./data/%s_kVCT/*' % (self.data_type))
        path_pCT = glob('./data/%s_pCT/*' % (self.data_type))

        self.n_batches = int(min(len(path_kVCT), len(path_pCT)) / batch_size)
        total_samples = self.n_batches * batch_size

        # Sample n_batches * batch_size from each path list so that model sees all samples from both domains
        path_kVCT = np.random.choice(path_kVCT, size=total_samples, replace=False)
        path_pCT = np.random.choice(path_pCT, size=total_samples, replace=False)

        for i in range(self.n_batches - 1):
            batch_kVCT = path_kVCT[i * batch_size:(i + 1) * batch_size]
            batch_pCT = path_pCT[i * batch_size:(i + 1) * batch_size]
            imgs_kVCT, imgs_pCT = [], []
            for img_kVCT, img_pCT in zip(batch_kVCT, batch_pCT):
                img_kVCT = self.read_dicom(img_kVCT)
                img_pCT = self.np.load(img_pCT)

                img_kVCT = scipy.misc.imresize(img_kVCT, self.img_res)
                img_pCT = scipy.misc.imresize(img_pCT, self.img_res)

                img_kVCT = 2 * (img_kVCT - np.min(img_kVCT)) / (np.max(img_kVCT) - np.min(img_kVCT)) - 1
                img_pCT = 2 * (img_pCT - np.min(img_pCT)) / (np.max(img_pCT) - np.min(img_pCT)) - 1

                imgs_kVCT = np.append(img_kVCT, i)
                imgs_pCT = np.append(img_pCT, i)

            yield imgs_kVCT, imgs_pCT

    # def load_img(self, path):
    #     img = self.imread(path)
    #     img = scipy.misc.imresize(img, self.img_res)
    #     img = img / 127.5 - 1.
    #     return img[np.newaxis, :, :, :]

    def read_dicom(self, path, voi_lut=True):
        dicom = pydicom.read_file(path)
        # VOI LUT (if available by DICOM device) is used to transform raw DICOM data to "human-friendly" view
        return apply_voi_lut(dicom.pixel_array, dicom) if voi_lut else dicom.pixel_array
