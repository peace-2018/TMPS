import argparse
import os
from glob import glob
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from tqdm import tqdm
from TMPS_seg_arch_flash import TMPS_arch
from dataset import Dataset
import torchvision.transforms as transforms_vision


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default=None,
                        help='model name')

    args = parser.parse_args()

    return args


def main():
    # args = parse_args()

    with open('./TMPS_arch_result/models/config.yml', 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    print('-'*20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-'*20)

    cudnn.benchmark = True

    print("=> creating model %s" % config['TMPS_arch'])
    model = TMPS_arch()
    model = model.cuda()

    # Data loading code
    test_img_ids = glob(os.path.join(config['test_dataset'], 'Frame', '*' + config['test_img_ext']))
    test_img_ids = [os.path.splitext(os.path.basename(p))[0] for p in test_img_ids]

    model.load_state_dict(torch.load('./TMPS_arch_result/models/model.pth'))
    model.eval()
    test_transform = transforms_vision.Compose([])
    test_dataset = Dataset(
        size_h=config['input_h'],
        size_w=config['input_w'],
        img_ids=test_img_ids,
        img_dir=os.path.join(config['test_dataset'], 'Frame'),  # valid_dataset
        mask_dir=os.path.join(config['test_dataset'], 'GT'),
        img_ext=config['test_img_ext'],
        mask_ext=config['test_mask_ext'],
        num_classes=config['num_classes'],
        transform=test_transform)

    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('./test_outputs', 'ETIS-LaribPolypDB'), exist_ok=True)

    with torch.no_grad():

        for input, target, meta in tqdm(test_loader, total=len(test_loader)):
            input = input.cuda()
            model = model.cuda()
            output = model(input)
            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):

                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join(os.path.join('./test_outputs', 'ETIS-LaribPolypDB', meta['img_id'][i] + '.png')),
                                (output[i, c] * 255).astype('uint8'))


    torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
