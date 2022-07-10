import cv2
import os
import numpy as np
from skimage.metrics import mean_squared_error
from skimage.metrics import peak_signal_noise_ratio
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
parser.add_argument("--niqe_path", required=True, type=str, help="path to niqe model parameters")
parser.add_argument("--test_low", required=False, type=bool, default=False ,help="if test on low-light image: True")
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

def ssim(img1, img2):
	C1 = (0.01 * 255) ** 2
	C2 = (0.03 * 255) ** 2
	img1 = img1.astype(np.float64)
	img2 = img2.astype(np.float64)
	kernel = cv2.getGaussianKernel(11, 1.5)
	window = np.outer(kernel, kernel.transpose())
	mu1 = cv2.filter2D(img1, -1, window)[5:-5, 5:-5]  # valid
	mu2 = cv2.filter2D(img2, -1, window)[5:-5, 5:-5]
	mu1_sq = mu1 ** 2
	mu2_sq = mu2 ** 2
	mu1_mu2 = mu1 * mu2
	sigma1_sq = cv2.filter2D(img1 ** 2, -1, window)[5:-5, 5:-5] - mu1_sq
	sigma2_sq = cv2.filter2D(img2 ** 2, -1, window)[5:-5, 5:-5] - mu2_sq
	sigma12 = cv2.filter2D(img1 * img2, -1, window)[5:-5, 5:-5] - mu1_mu2
	ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) *
															(sigma1_sq + sigma2_sq + C2))
	return ssim_map.mean()


def calculate_ssim(img1, img2):
	'''calculate SSIM
	the same outputs as MATLAB's
	img1, img2: [0, 255]
	'''
	if not img1.shape == img2.shape:
		raise ValueError('Input images must have the same dimensions.')
	if img1.ndim == 2:
		return ssim(img1, img2)
	elif img1.ndim == 3:
		if img1.shape[2] == 3:
			ssims = []
			for i in range(3):
				ssims.append(ssim(img1, img2))
			return np.array(ssims).mean()
		elif img1.shape[2] == 1:
			return ssim(np.squeeze(img1), np.squeeze(img2))
	else:
		raise ValueError('Wrong input image dimensions.')

def im2tensor(image, imtype=np.uint8, cent=1., factor=255./2.):
    return torch.Tensor((image / factor - cent)[:, :, :, np.newaxis].transpose((3, 2, 0, 1)))
def Lpips(img1, img2):
    #ref = lpips.im2tensor(lpips.load_image(img1))
    #output = lpips.im2tensor(lpips.load_image(img2))
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


def CSE(img1, img2, histsize=4, Pcount=2, Kdiff=1, Ithreshold=0.25):
    '''
    function: given two images, return CSE value.
    input:  range:[0,255]   type:uint8    format:[h,w,c]   BGR(Note: not RGB)
    output: a python value, i.e., color-sensitive error (CSE)
    '''

    maximum2 = get_maximum(img2, histsize)
    comp_weight = get_comp_weight(img1, img2, histsize)
    error = 0
    for i in range(3):
        range_v = 180 if i == 0 else 256  # 180 for H, 256 for S,V
        hist1 = cv2.calcHist([cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)], [i], None, [range_v / histsize], [0, range_v])
        hist2 = cv2.calcHist([cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)], [i], None, [range_v / histsize], [0, range_v])

        if np.max(hist2, 0)[0] < maximum2 * Ithreshold:
            error += (1 / range_v) * comp_weight[i] * np.sqrt(np.mean((hist1 - hist2) ** 2))
            continue

        peak1 = peak_positioning(hist1.squeeze(-1), Pcount)
        peak2 = peak_positioning(hist2.squeeze(-1), Pcount)
        peak1, peak2 = peak_aligning(peak1, peak2, 0.3)
        peak_weight = get_peak_weight(peak1, peak2)

        distance = 0
        for j in range(Pcount):
            distance += math.fabs(peak1[j][0] - peak2[j][0]) * peak_weight[j]

        diff1 = make_diff(hist1, Kdiff)
        diff2 = make_diff(hist2, Kdiff)
        integral = np.abs(get_integral(diff1) - get_integral(diff2))[0]

        error += (1 / range_v) * comp_weight[i] * integral * math.exp(math.sqrt(distance))

    return error
    


if __name__ == '__main__':
	#ref_path = 'data/train_data/LOL/high/'
	#output_path = 'data/test_output/low/'
	#ref_path = 'data/mm20data/test/test_H/'
	#output_path = "/home/YHr10942/pfc/EnlightenGAN/EnlightenGAN-master/Results/"
  ref_path = opt.ref_path
  output_path = opt.output_path
  reference_list = populate_list(ref_path)
  output_list = populate_list(output_path)
  low = opt.test_low
  print(len(reference_list), len(output_list))
  print(reference_list[0], output_list[0])
  MSE_list = []
  PSNR_list = []
  SSIM_list = []
  NIQE_list = []
  LPIPS_list = []
  CSE_list = []
  for idx in range(len(reference_list)):
    img1 = cv2.imread(os.path.join(os.getcwd(), ref_path, reference_list[idx]))
    img2 = cv2.imread(os.path.join(os.getcwd(), output_path, output_list[idx]))
    img2 = cv2.resize(img2, (img1.shape[1], img1.shape[0]), interpolation=cv2.INTER_AREA)
    #print(img2.max())
    # print(output_list[idx])
    MSE = mean_squared_error(img1, img2)
    PSNR = peak_signal_noise_ratio(img1, img2)
    SSIM = calculate_ssim(img1, img2)
    NIQE = niqe(img2)
    LPIPS = Lpips(img1, img2)
    # cse = CSE(img1, img2)
    cse = 0
    print('MSE: ', MSE)
    print('PSNR: ', PSNR)
    print('SSIM: ', SSIM)
    print('NIQE: ', NIQE)
    print('LPIPS: ', LPIPS)
    print('CSE: ', cse)
    MSE_list.append(MSE)
    PSNR_list.append(PSNR)
    SSIM_list.append(SSIM)
    NIQE_list.append(NIQE)
    LPIPS_list.append(LPIPS)
    CSE_list.append(cse)
  sortedPSNR = np.argsort(np.array(PSNR_list))
  sortedSSIM = np.argsort(np.array(SSIM_list))
  for i in range(10):
    print(output_list[sortedPSNR[-i]])

  print('\n\n')
  
  for i in range(10):
    print(output_list[sortedSSIM[-i]])
	
  print('MSE = ', np.array(MSE_list).mean() / (255*255))
  print('PSNR = ', np.array(PSNR_list).mean())
  print('SSIM = ', np.array(SSIM_list).mean())
  print('NIQE = ', np.array(NIQE_list).mean())
  print('LPIPS = ', np.array(LPIPS_list).mean())
  print('CSE = ', np.array(CSE_list).mean())
  if low == True:
      print('CSE(ratio) = ', np.array(CSE_list).mean()/20292.465560863002)
  else:
      print('CSE(ratio) = ', np.array(CSE_list).mean()/1450.6354374043135)