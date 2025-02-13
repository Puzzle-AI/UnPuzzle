"""
Embedding slide_feature dataset     Script  ver： Dec 5th 17:00

"""
import sys
from pathlib import Path

# For convenience
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir))
sys.path.append(str(this_file_dir.parent))
sys.path.append(str(this_file_dir.parent.parent))  # Go up 3 levels

import time
import torch.nn as nn
from tqdm import tqdm

from DataPipe.h5tools import hdf5_save_a_slide_embedding_dataset
from DataPipe.dataset_framework import *
from ModelBase.Get_WSI_model import build_WSI_prob_embedding_model
from Utils.tools import setup_seed


def building_prob_dataset(model, dataloader, dataset_size, h5_file_path, device=torch.device("cpu")):
    model.eval()  # Set model to evaluate mode
    epoch_time = time.time()
    failed_sample_count = 0
    # data loop
    for data_iter_step, sample in tqdm(enumerate(dataloader), desc="Embedding Slide level features for",
                                       unit="batch (of {} slides)".format(dataloader.batch_size),
                                       total=dataset_size // dataloader.batch_size):
        '''
        sample = {'image_features': data_dict['image_features'],
                  'image_features_lens': data_dict['image_features_lens'],
                  'pad_mask': data_dict['pad_mask'],
                  'coords_yx': data_dict['coords_yx'],
                  'slide_id': slide_id,
                  'task_name_list': self.task_name_list,
                  'task_description_list': task_description_list}
        '''
        # jump the batch if it cannot correct by WSI_collate_fn in dataloader
        if sample is None:
            failed_sample_count += dataloader.batch_size
            continue
        else:
            # take data and task_description_list from sample
            image_features, coords_yx, task_description_list, slide_ids = sample
            # image_features is a tensor of [B,N,D],  coords_yx is tensor of [B,N,2]
            image_features = image_features.to(device)
            coords_yx = coords_yx.to(device)
            # task_description_list [task, batch_size] batch-stacked tensors, element of long-int or float

        # count failed samples in dataloader (should be 0, normally)
        # default B - B = 0, we don't have the last batch issue 'drop last batch in training code'
        failed_sample_count += dataloader.batch_size - len(slide_ids)

        with torch.cuda.amp.autocast():
            with torch.no_grad():
                slide_features = model(image_features, coords_yx)
                for slide_idx in range(len(slide_ids)):
                    slide_feature = slide_features[slide_idx].cpu().numpy()
                    slide_id = slide_ids[slide_idx]
                    # Notice we will save the output features in whatever the model returns (without batch dim)
                    # print('slide_id', slide_id)
                    # print('slide_feature', slide_feature)
                    hdf5_save_a_slide_embedding_dataset(h5_file_path, slide_id, slide_feature,
                                                        section_type='slide_features')

    # total samples (remove dataloader-failed samples)
    valid_samples = dataset_size - failed_sample_count

    time_elapsed = time.time() - epoch_time
    print('Embedding complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('total samples from embedded tile dataset:', dataset_size)
    print('total valid_samples in the embedded slide dataset:', valid_samples)


def main(args):
    slide_embedding_folder = args.slide_embedding_folder or \
                             os.path.join(args.tile_embedding_path, args.task_setting_folder_name)
    if not os.path.exists(slide_embedding_folder):
        os.mkdir(slide_embedding_folder)
    # output path
    h5_file_path = os.path.join(slide_embedding_folder, 'slide_embedding.h5')

    # instantiate the dataset
    dataset = SlideDataset(args.tile_embedding_path, args.task_description_csv,
                           task_setting_folder_name=args.task_setting_folder_name,
                           split_name=None, slide_id_key=args.slide_id_key, split_target_key=None,
                           task_type='embedding', max_tiles=args.max_tiles)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size,
                                             collate_fn=MTL_collate_fn,
                                             shuffle=False, num_workers=args.num_workers)
    # info
    dataset_size = len(dataset)

    # GPU idx start with0, -1 to use multiple GPU
    if args.gpu_idx == -1:  # use all cards
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            gpu_use = args.gpu_idx
        else:
            print('we dont have more GPU idx here, try to use gpu_idx=0')
            try:
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # setting k for: only card idx k is sighted for this code
                gpu_use = 0
            except:
                print("GPU distributing ERRO occur use CPU instead")
                gpu_use = 'cpu'

    else:
        # Decide which device we want to run on
        try:
            # setting k for: only card idx k is sighted for this code
            os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu_idx)
            gpu_use = args.gpu_idx
        except:
            print('we dont have that GPU idx here, try to use gpu_idx=0')
            try:
                # setting 0 for: only card idx 0 is sighted for this code
                os.environ['CUDA_VISIBLE_DEVICES'] = '0'
                gpu_use = 0
            except:
                print("GPU distributing ERRO occur use CPU instead")
                gpu_use = 'cpu'
    # device environment
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # build model
    model, default_ROI_feature_dim = build_WSI_prob_embedding_model(model_name=args.model_name,
                                                                    local_weight_path=args.local_weight_path,
                                                                    ROI_feature_dim=args.ROI_feature_dim)
    assert args.ROI_feature_dim is None or default_ROI_feature_dim == args.ROI_feature_dim

    model = torch.compile(model)
    print('GPU:', gpu_use)
    if gpu_use == -1:
        model = nn.DataParallel(model)
    model.to(device)

    building_prob_dataset(model, dataloader, dataset_size, h5_file_path, device=device)


def get_args_parser():
    parser = argparse.ArgumentParser(description='building slide_level prob_dataset')

    # Environment parameters
    parser.add_argument('--gpu_idx', default=-1, type=int,
                        help='use a single GPU with its index, -1 to use multiple GPU')

    # PATH
    parser.add_argument('--slide_embedding_folder', default=None,
                        type=str, help='path to the embedded slide dataset, '
                                       'default go to task_setting folder of embedded tile dataset')
    parser.add_argument('--tile_embedding_path',
                        default='/data/hdd_1/BigModel/TCGA-LUAD-LUSC/Tile_embeddings/gigapath/',
                        type=str, help='path to the embedded tile dataset')
    parser.add_argument('--local_weight_path', default=None, type=str, help='local weight path')

    # Task settings and configurations for dataloaders
    # labels
    parser.add_argument('--task_description_csv', default=None, type=str,
                        help='label csv file path, default none will go to task_description.csv in task_setting_folder')

    parser.add_argument('--task_setting_folder_name', default='task-settings', type=str,
                        help='task-settings folder name')

    parser.add_argument('--slide_id_key', default='patient_id', type=str,
                        help='key for mapping the label')

    # slide module settings
    parser.add_argument('--max_tiles', default=None, type=int, help='max tile for loading, '
                                                                    'default will load config or be 10000')

    # Model settings
    parser.add_argument('--model_name', default='gigapath', type=str, help='slide_feature level model name')
    parser.add_argument('--ROI_feature_dim', default=None, type=int,
                        help='feature slide_embed_dim , default will be loading from model config')

    # embedding settings
    parser.add_argument('--batch_size', default=1, type=int, help='batch_size of WSI, default 1')
    parser.add_argument('--num_workers', default=2, type=int, help='dataloader num_workers')

    # helper
    parser.add_argument('--check_minibatch', default=25, type=int, help='check batch_size')

    return parser


if __name__ == '__main__':
    # setting up the random seed
    setup_seed(42)

    parser = get_args_parser()
    args = parser.parse_args()
    main(args)

    '''
    python Build_slide_probing_dataset.py --model_name gigapath --slide_embedding_folder /data/hdd_1/BigModel/TCGA-LUAD-LUSC/Slide_embeddings/gigapath --tile_embedding_path /data/hdd_1/BigModel/TCGA-LUAD-LUSC/Tile_embeddings/gigapath --num_workers 8
    '''
