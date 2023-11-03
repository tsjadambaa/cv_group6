import os
import time

import matplotlib.pyplot as plt
import torch
import numpy as np
from tqdm.auto import tqdm

from dataset import lowlight_dataset
from torch.utils.data import DataLoader

from torchmetrics.image import PeakSignalNoiseRatio as PSNR
from torchmetrics.image import StructuralSimilarityIndexMeasure as SSIM
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity as LPIPS
import model
import utils


def get_metrics(enh_img: torch.Tensor, gt_img: torch.Tensor, device: str = 'gpu'):
    '''
        Функция для измерения метрик. Расчитывает среднее значение по сэмплу
        Arguments:
            :enh_img: изображения после обработки
            :gt_img: ground-truth изображения
            :device: cpu/cuda
    '''
    batch_size = enh_img.shape[0]
    enh_img = enh_img.to(device)
    gt_img = gt_img.to(device)
    # PSNR
    psnr = PSNR(data_range=batch_size).to(device)
    psnr_score = psnr(enh_img, gt_img).item()
    # SSIM
    ms_ssim = SSIM(data_range=batch_size).to(device)
    ssim_score = ms_ssim(enh_img, gt_img).item()
    # LILPS
    lpips = LPIPS('alex').to(device)
    lpips_score = lpips(enh_img, gt_img).item()
    del psnr, ms_ssim, lpips
    return psnr_score, ssim_score, lpips_score


def evaluate(model_name, dataloader, device='cpu'):
    '''Оценка метрик решения'''

    psnr_list = np.array([])
    ssim_list = np.array([])
    lpips_list = np.array([])
    times = np.array([])
    print(device)
    print(model_name)

    DCE_net = model.enhance_net_nopool().cuda()
    DCE_net.load_state_dict(torch.load(model_name))
    DCE_net.training = False

    pbar = tqdm(total=len(dataloader))
    with torch.no_grad():
        for lowlight_img, gt_img in dataloader:
            # Улучшение качества изображения
            start = time.time()
            img_lowlight = lowlight_img.cuda()
            _, enhanced_image, _ = DCE_net(img_lowlight)
            times = np.append(times, time.time() - start)
            ################################

            psnr_score, ssim_score, lpips_score = get_metrics(enhanced_image, gt_img, device=device)

            psnr_list = np.append(psnr_list, psnr_score)
            ssim_list = np.append(ssim_list, ssim_score)
            lpips_list = np.append(lpips_list, lpips_score)
            del lowlight_img, enhanced_image, gt_img
            pbar.update(1)

    print(f'PSNR: {np.ma.masked_invalid(psnr_list).mean()}\nSSIM: {ssim_list.mean()}\nLPIPS: {lpips_list.mean()}')
    print(f"Среднее время на изображение: {times.mean()}")


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def img_show(*images):
    fig, axs = plt.subplots(1, len(images), figsize=(5, 6))
    for ax, img in zip(axs, images):
        ax.imshow(img)
    plt.show()


def train(loader, snapshots_folder, lr, weight_decay, epochs):
    DCE_net = model.enhance_net_nopool().cuda()

    DCE_net.apply(weights_init)

    L_color = utils.L_color()
    L_spa = utils.L_spa()

    L_exp = utils.L_exp(16, 0.6)
    L_TV = utils.L_TV()

    optimizer = torch.optim.Adam(DCE_net.parameters(), lr=lr, weight_decay=weight_decay)

    DCE_net.train()
    last_model_name = ""

    for epoch in range(epochs):
        for iteration, img in enumerate(loader):
            img_lowlight, img_highlight = img
            img_lowlight = img_lowlight.cuda()
            enhanced_image_1, enhanced_image, A = DCE_net(img_lowlight)
            # img_show(img_lowlight.cpu()[0, :, :, :].permute(1, 2, 0),
            #          enhanced_image.cpu()[0, :, :, :].permute(1, 2, 0).detach().numpy(),
            #          enhanced_image_1.cpu()[0, :, :, :].permute(1, 2, 0).detach().numpy())

            Loss_TV = 200 * L_TV(A)
            loss_spa = torch.mean(L_spa(enhanced_image, img_lowlight))
            loss_col = 5 * torch.mean(L_color(enhanced_image))
            loss_exp = 10 * torch.mean(L_exp(enhanced_image))

            # best_loss
            loss = Loss_TV + loss_spa + loss_col + loss_exp

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm(DCE_net.parameters(), 0.1)
            optimizer.step()

            if ((iteration + 1) % 10) == 0:
                print("Loss at iteration", iteration + 1, ":", loss.item())

            if ((iteration + 1) % 10) == 0:
                last_model_name = snapshots_folder + "Epoch" + str(epoch) + '.pth'
                torch.save(DCE_net.state_dict(), last_model_name)
    return last_model_name


if __name__ == "__main__":
    os.environ['CUDA_VISIBLE_DEVICES'] = "0"
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Device: %s" % device)

    snapshots_folder = "snapshots_folder/"
    if not os.path.exists(snapshots_folder):
        os.mkdir(snapshots_folder)

    batch_size = 14
    lr = 0.0001
    weight_decay = 0.0001
    epochs = 200
    print(F"Batch size: {batch_size}, lr: {lr}, weight_decay: {weight_decay}, epochs: {epochs}")

    img_names = os.listdir('dataset/low')
    dataset = lowlight_dataset(img_names, 'dataset')
    generator = torch.Generator().manual_seed(42)
    test_ds, valid_ds = torch.utils.data.random_split(dataset, (0.8, 0.2), generator=generator)
    test_dl = DataLoader(test_ds.dataset, batch_size=batch_size, num_workers=4)
    valid_dl = DataLoader(valid_ds.dataset, batch_size=batch_size, num_workers=4)
    model_name = train(test_dl, snapshots_folder=snapshots_folder, lr=lr, weight_decay=weight_decay, epochs=epochs)
    model_name = snapshots_folder + "Epoch37.pth"
    evaluate(model_name, valid_dl, device=device)
