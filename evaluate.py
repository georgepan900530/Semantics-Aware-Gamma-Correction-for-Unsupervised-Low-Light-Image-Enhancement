import cv2
import os
import numpy as np
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
from skimage.metrics import structural_similarity as ssim
import argparse
import math
from scipy.special import gamma
from scipy.ndimage.filters import gaussian_filter
import scipy.io
import torch
import lpips

parser = argparse.ArgumentParser()
parser.add_argument("--ref_path", required=True, type=str, help="path to the directory of reference images")
parser.add_argument("--output_path", required=True, type=str, help="path to the directory of enhanced images")
parser.add_argument("--niqe_path", type=str, default = "niqe/mvg_params.mat", help="path to niqe model parameters")
opt = parser.parse_args()

alpha_p = np.arange(0.2, 10, 0.001)
alpha_r_p = scipy.special.gamma(2.0 / alpha_p) ** 2 / (scipy.special.gamma(1.0 / alpha_p) * scipy.special.gamma(3. / alpha_p))
loss_fn = lpips.LPIPS(net='alex', spatial = True)
loss_fn.cuda()


def estimate_aggd_params(x):
    x_left = x[x < 0]
    x_right = x[x >= 0]
    stddev_left = 0
    stddev_right = 0
    if len(x_left) > 0:
        stddev_left = math.sqrt((np.sum(x_left ** 2) / (x_left.size)))
    if len(x_right) > 0:
        stddev_right = math.sqrt((np.sum(x_right ** 2) / (x_right.size)))

    if stddev_right == 0:
        return 1, 0, 0
    r_hat = np.sum(np.abs(x)) ** 2 / (x.size * np.sum(x ** 2))
    y_hat = stddev_left / stddev_right  # gamma hat
    R_hat = r_hat * (y_hat ** 3 + 1) * (y_hat + 1) / ((y_hat ** 2 + 1) ** 2)

    pos = np.argmin((alpha_r_p - R_hat) ** 2)
    alpha = alpha_p[pos]
    beta_left = stddev_left * math.sqrt(gamma(1.0 / alpha) / gamma(3.0 / alpha))
    beta_right = stddev_right * math.sqrt(gamma(1.0 / alpha) / gamma(3.0 / alpha))
    return alpha, beta_left, beta_right


def compute_nss_features(img_norm):
    features = []
    alpha, beta_left, beta_right = estimate_aggd_params(img_norm)
    features.extend([alpha, (beta_left + beta_right) / 2])

    for x_shift, y_shift in ((0, 1), (1, 0), (1, 1), (1, -1)):
        img_pair_products = img_norm * np.roll(np.roll(img_norm, y_shift, axis=0), x_shift, axis=1)
        alpha, beta_left, beta_right = estimate_aggd_params(img_pair_products)
        eta = (beta_right - beta_left) * (gamma(2.0 / alpha) / gamma(1.0 / alpha))
        features.extend([alpha, eta, beta_left, beta_right])

    return features


def norm(img, sigma=7 / 6):
    mu = gaussian_filter(img, sigma, mode='nearest', truncate=2.2)
    sigma = np.sqrt(np.abs(gaussian_filter(img * img, sigma, mode='nearest', truncate=2.2) - mu * mu))
    img_norm = (img - mu) / (sigma + 1)
    return img_norm


def niqe(image):
    if image.ndim == 3:
        img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        img = image
    
    model_mat = scipy.io.loadmat(opt.niqe_path)
    model_mu = model_mat['mean']
    model_cov = model_mat['cov']

    features = None
    h, w = img.shape
    quantized_h = max(h // 96, 1) * 96
    quantized_w = max(w // 96, 1) * 96

    quantized_img = img[:quantized_h, :quantized_w]
    img_scaled = quantized_img
    for scale in [1, 2]:
        if scale != 1:
            img_scaled = cv2.resize(quantized_img, None, fx=1 / scale, fy=1 / scale, interpolation=cv2.INTER_AREA)
        img_norm = norm(img_scaled.astype(float))
        scale_features = []
        block_size = 96 // scale
        for block_col in range(img_norm.shape[0] // block_size):
            for block_row in range(img_norm.shape[1] // block_size):
                block_features = compute_nss_features(
                    img_norm[block_col * block_size:(block_col + 1) * block_size, block_row * block_size:(block_row + 1) * block_size])
                scale_features.append(block_features)

        if features is None:
            features = np.vstack(scale_features)
        else:
            features = np.hstack([features, np.vstack(scale_features)])

    features_mu = np.mean(features, axis=0)
    features_cov = np.cov(features.T)

    pseudoinv_of_avg_cov = np.linalg.pinv((model_cov + features_cov) / 2)
    niqe_quality = math.sqrt((model_mu - features_mu).dot(pseudoinv_of_avg_cov.dot((model_mu - features_mu).T)))

    return niqe_quality


def populate_list(images_path):
	cwd = os.getcwd()
	path = os.path.join(cwd, str(images_path))
	image_list = os.listdir(path)
	image_list.sort()
	return image_list


def calculate_ssim(img1, img2):
    img1 = cv2.cvtColor(img1,cv2.COLOR_BGR2GRAY)
    img2 = cv2.cvtColor(img2,cv2.COLOR_BGR2GRAY)
    sim = ssim(img1, img2)
    return sim
    

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))


def Lpips(img1, img2):
    ref = img1[:,:,::-1]
    output = img2
    output = cv2.resize(img2, (ref.shape[1], ref.shape[0]), interpolation=cv2.INTER_AREA)
    output = output[:,:,::-1]
    ref = im2tensor(ref)
    output = im2tensor(output)
    ref = ref.cuda()
    output = output.cuda()
    d = loss_fn.forward(ref, output)
    return d.mean().cpu().detach().numpy()


def get_maximum(img2, histsize):
    maximum2 = 0
    for i in range(3):
        range_v = 180 if i == 0 else 256  # 180 for H, 256 for S,V
        hist2 = cv2.calcHist([cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)], [i], None, [range_v / histsize], [0, range_v])

        if np.max(hist2, 0)[0] > maximum2:
            maximum2 = np.max(hist2, 0)[0]

    return maximum2


def get_comp_weight(img1, img2, histsize):
    weight = []
    for i in range(3):
        range_v = 180 if i == 0 else 256  # 180 for H, 256 for S,V
        hist1 = cv2.calcHist([cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)], [i], None, [range_v / histsize], [0, range_v])
        hist2 = cv2.calcHist([cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)], [i], None, [range_v / histsize], [0, range_v])

        max1 = np.max(hist1, 0)[0]
        max2 = np.max(hist2, 0)[0]
        weight.append((max1 + max2) / 2)

    weight /= sum(weight)

    return weight


def make_diff(vector, order):
    for _ in range(order):
        vector = vector[0:-1] - vector[1:]

    return vector


def get_integral(vector):
    vector = np.abs(vector)
    integral = np.sum(vector) - 0.5 * (vector[0] + vector[-1])

    return integral


def peak_positioning(vector, Pcount):
    # Pcount should be a positive integer

    cur_peak = 0
    peak = []
    sort_v = np.argsort(-vector)
    for i in sort_v:
        if i == 0:
            if vector[i] >= vector[i + 1]:
                peak.append((i, vector[i]))
                cur_peak += 1
                if cur_peak == Pcount:
                    return peak
        elif i == len(sort_v) - 1:
            if vector[i] >= vector[i - 1]:
                peak.append((i, vector[i]))
                cur_peak += 1
                if cur_peak == Pcount:
                    return peak
        else:
            if vector[i] >= vector[i - 1] and vector[i] >= vector[i + 1]:
                peak.append((i, vector[i]))
                cur_peak += 1
                if cur_peak == Pcount:
                    return peak

    return peak


def get_peak_weight(peak1, peak2):
    weight = []
    for i in range(len(peak1)):
        weight.append((peak1[i][1] + peak2[i][1]) / 2)
    weight /= sum(weight)

    return weight


def peak_aligning(peaks1, peaks2, threshold=0.25):
    if len(peaks1) != len(peaks2):
        raise Exception('The number of peak1 and peak2 should be the same!')
    for i in range(len(peaks1) - 1):
        for j in range(i + 1, len(peaks1)):
            if math.fabs(peaks1[i][0] - peaks2[i][0]) < math.fabs(peaks1[i][0] - peaks2[j][0]) and math.fabs(
                    peaks1[j][0] - peaks2[j][0]) < math.fabs(
                peaks1[j][0] - peaks2[i][0]):  # The x-coordinate must satisfy the conditions
                continue
            difference = 0.5 * math.fabs(peaks1[i][1] - peaks1[j][1]) + 0.5 * math.fabs(peaks2[i][1] - peaks2[j][1])
            range_p = 0.25 * (peaks1[i][1] + peaks1[j][1] + peaks2[i][1] + peaks2[j][1])
            if difference / range_p <= threshold:
                peaks1[i], peaks1[j] = peaks1[j], peaks1[i]

    return peaks1, peaks2


if __name__ == '__main__':
    ref_path = opt.ref_path
    output_path = opt.output_path
    reference_list = populate_list(ref_path)
    output_list = populate_list(output_path)
    
    MSE_list = []
    PSNR_list = []
    SSIM_list = []
    NIQE_list = []
    LPIPS_list = []

    for idx in range(len(reference_list)):
        img1 = cv2.imread(os.path.join(os.getcwd(), ref_path, reference_list[idx]))
        img2 = cv2.imread(os.path.join(os.getcwd(), output_path, output_list[idx]))
        img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)

        MSE = mean_squared_error(img1, img2)
        PSNR = peak_signal_noise_ratio(img1, img2)
        SSIM = calculate_ssim(img1, img2)
        NIQE = niqe(img2)
        LPIPS = Lpips(img1, img2)

        print('MSE: ', MSE)
        print('PSNR: ', PSNR)
        print('SSIM: ', SSIM)
        print('NIQE: ', NIQE)
        print('LPIPS: ', LPIPS)
        MSE_list.append(MSE)
        PSNR_list.append(PSNR)
        SSIM_list.append(SSIM)
        NIQE_list.append(NIQE)
        LPIPS_list.append(LPIPS)

    print('\nFinal Averaged Score:\n')
    print('MSE = ', np.array(MSE_list).mean() / (255*255))
    print('PSNR = ', np.array(PSNR_list).mean())
    print('SSIM = ', np.array(SSIM_list).mean())
    print('LPIPS = ', np.array(LPIPS_list).mean())
    print('NIQE = ', np.array(NIQE_list).mean())
