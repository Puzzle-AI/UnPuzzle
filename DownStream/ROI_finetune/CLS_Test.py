"""
Testing ROI models    Script  ver： Feb 9th 16:30
"""
from __future__ import print_function, division
import os
import sys
from pathlib import Path

# For convenience, import all path to sys
this_file_dir = Path(__file__).resolve().parent
sys.path.append(str(this_file_dir))
sys.path.append(str(this_file_dir.parent))
sys.path.append(str(this_file_dir.parent.parent))
sys.path.append(str(this_file_dir.parent.parent.parent))  # Go up 3 levels
from ModelBase.Get_ROI_model import build_ROI_CLS_model, build_promptmodel
from Utils.data_augmentation import *
from Utils.visual_usage import *
from Utils.check_log_json import CLS_JSON_logger

import argparse
import json
import time
import torch
import torch.nn as nn
import torchvision
from tensorboardX import SummaryWriter


# Test iteration
def test_model(model, test_dataloader, criterion, class_names, test_dataset_size, model_name, test_model_name,
               edge_size,
               check_minibatch=100, device=None, draw_path='../imaging_results', enable_attention_check=None,
               enable_visualize_check=True, writer=None):
    """
    Testing iteration

    :param model: model object
    :param test_dataloader: the test_dataloader obj
    :param criterion: loss func obj
    :param class_names: The name of classes for priting
    :param test_dataset_size: size of datasets

    :param model_name: model idx for the getting trained model
    :param edge_size: image size for the input image
    :param check_minibatch: number of skip over minibatch in calculating the criteria's results etc.

    :param device: cpu/gpu object
    :param draw_path: path folder for output pic
    :param enable_attention_check: use attention_check to show the pics of models' attention areas
    :param enable_visualize_check: use visualize_check to show the pics

    :param writer: attach the records to the tensorboard backend
    """

    # scheduler is an LR scheduler object from torch.optim.lr_scheduler.
    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    since = time.time()

    # initiate the empty json dict
    json_logger = CLS_JSON_logger(class_names, draw_path, test_model_name)

    print('Epoch: Test')
    print('-' * 10)
    epoch = 'Test'
    phase = 'Test'
    index = 0
    model_time = time.time()

    json_logger.init_epoch(epoch=epoch)
    model.eval()  # Set model to evaluate mode

    # criteria, initially empty
    running_loss = 0.0
    log_running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    for inputs, labels in test_dataloader:  # use different dataloader in different phase
        inputs = inputs.to(device)
        # print('inputs[0]',type(inputs[0]))

        labels = labels.to(device)

        # zero the parameter gradients only need in training
        # optimizer.zero_grad()
        with torch.amp.autocast("cuda"):
            with torch.no_grad():
                # forward
                outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)

            # log criteria: update
            log_running_loss += loss.item()
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)

            json_logger.update_confusion_matrix(preds, labels)

        # attach the records to the tensorboard backend
        if writer is not None:
            # ...log the running loss
            writer.add_scalar(phase + ' minibatch loss', float(loss.item()), index)
            writer.add_scalar(phase + ' minibatch ACC',
                              float(torch.sum(preds == labels.data) / inputs.size(0)), index)

        # at the checking time now
        if check_minibatch == 0 or (index % check_minibatch == check_minibatch - 1):
            model_time = time.time() - model_time
            # avoid divide zero
            if check_minibatch != 0:
                check_index = index // check_minibatch + 1
            else:
                check_index = index + 1

            epoch_idx = 'Test'
            print('Epoch:', epoch_idx, '   ', phase, 'index of ' + str(check_minibatch) + ' minibatch:',
                  check_index, '     time used:', model_time)

            print('minibatch AVG loss:', float(log_running_loss) / check_minibatch)

            # how many image u want to check, should SMALLER THAN the batchsize

            if enable_attention_check:
                try:
                    check_SAA(inputs, labels, model, model_name, edge_size, class_names, num_images=1,
                              pic_name='GradCAM_' + str(epoch_idx) + '_I_' + str(index + 1),
                              draw_path=draw_path, writer=writer)
                except:
                    print('model:', model_name, ' with edge_size', edge_size, 'is not supported yet')
            else:
                pass

            if enable_visualize_check:
                visualize_check(inputs, labels, model, class_names, num_images=-1,
                                pic_name='Visual_' + str(epoch_idx) + '_I_' + str(index + 1),
                                draw_path=draw_path, writer=writer)

            model_time = time.time()
            log_running_loss = 0.0

        index += 1

    # log criteria: print
    epoch_loss = running_loss / test_dataset_size
    epoch_acc = running_corrects.double() / test_dataset_size * 100
    print('\nEpoch:  {} \nLoss: {:.4f}  Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

    json_logger.update_epoch_phase(epoch=epoch, phase=phase)

    print('\n')

    time_elapsed = time.time() - since
    print('Testing complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))

    # attach the records to the tensorboard backend
    if writer is not None:
        writer.close()

    json_logger.dump_json()

    return model


def main(args):
    if args.paint:
        # use Agg kernel, not painting in the front-desk
        import matplotlib
        matplotlib.use('Agg')

    gpu_idx = args.gpu_idx  # GPU idx start with0, -1 to use multiple GPU

    enable_notify = args.enable_notify  # False
    enable_tensorboard = args.enable_tensorboard  # False

    enable_attention_check = args.enable_attention_check  # False
    enable_visualize_check = args.enable_visualize_check  # False

    data_augmentation_mode = args.data_augmentation_mode  # 0

    # Prompt
    PromptTuning = args.PromptTuning  # None  "Deep" / "Shallow"
    Prompt_Token_num = args.Prompt_Token_num  # 20
    PromptUnFreeze = args.PromptUnFreeze  # False

    # model info
    model_name = args.model_name  # the model we are going to use. by the format of Model_size_other_info
    tag = '' if args.tag is None else "_" + args.tag
    run_name = 'CLS_' + model_name + tag
    test_model_name = run_name + '_Test'
    # PATH info
    runs_path = args.runs_path  # root path of saving the multiple experiments runs
    model_path = args.save_model_path or args.runs_path  # root path to saving models, if none, will go to draw root
    data_path = args.data_path  # path to the dataset

    # we use the run name as a warp folder for both Train and Test
    assert os.path.exists(os.path.join(runs_path, run_name))
    runs_path = os.path.join(runs_path, run_name, 'Test')
    os.makedirs(runs_path, exist_ok=True)
    auto_save_model_path = os.path.join(model_path, run_name, run_name + '.pth')
    # load trained model by its task-based saving name
    save_model_path = args.model_path_by_hand or auto_save_model_path  # if not specified with args.model_path_by_hand
    # Pre_Trained model basic for prompt turned model's test
    Pre_Trained_model_path = args.Pre_Trained_model_path  # None for auto-loading backbone for prompt

    # choose the test dataset
    if os.path.exists(os.path.join(data_path, 'test')):
        test_data_path = os.path.join(data_path, 'test')
    elif os.path.exists(os.path.join(data_path, 'Test')):
        test_data_path = os.path.join(data_path, 'Test')
    else:
        raise ValueError('test_data_path not available')
    # dataset info
    num_classes = args.num_classes  # default 0 for auto-fit
    edge_size = args.edge_size  # 1000 224 384

    # validating tasks_to_run
    batch_size = args.batch_size  # 10
    criterion = nn.CrossEntropyLoss()

    # Data Augmentation is not used in validating or testing
    data_transforms = data_augmentation(data_augmentation_mode, edge_size=edge_size)

    # test tasks_to_run is the same as the validate dataset's tasks_to_run
    test_datasets = torchvision.datasets.ImageFolder(test_data_path, data_transforms['Val'])
    test_dataset_size = len(test_datasets)
    # skip minibatch none to draw 20 figs
    check_minibatch = args.check_minibatch if args.check_minibatch is not None else test_dataset_size // (
            20 * batch_size)
    check_minibatch = check_minibatch if check_minibatch > 0 else 1

    test_dataloader = torch.utils.data.DataLoader(test_datasets, batch_size=batch_size, shuffle=False, num_workers=1)

    class_names = [d.name for d in os.scandir(test_data_path) if d.is_dir()]
    class_names.sort()

    if num_classes == 0:
        print("class_names:", class_names)
        num_classes = len(class_names)
    else:
        if len(class_names) == num_classes:
            print("class_names:", class_names)
        else:
            print('classfication number of the model mismatch the dataset requirement of:', len(class_names))
            return -1

    if enable_notify:
        import notifyemail as notify

        notify.Reboost(mail_host='smtp.163.com', mail_user='tum9598@163.com', mail_pass='xxxx',
                       default_reciving_list=['tum9598@163.com'],  # change here if u want to use notify
                       log_root_path='log', max_log_cnt=5)

        if enable_tensorboard:
            notify.add_text('testing model_name: ' + str(model_name) + '.  update to the tensorboard')
        else:
            notify.add_text('testing model_name: ' + str(model_name) + '.  not update to the tensorboard')
        notify.add_text('class_names =' + str(class_names))
        notify.add_text('edge_size =' + str(edge_size))
        notify.add_text('batch_size =' + str(batch_size))
        notify.send_log()

    # get model
    pretrained_backbone = False  # model is trained already, pretrained backbone weight is useless here

    if PromptTuning is None:
        model = build_ROI_CLS_model(model_name=model_name, num_classes=num_classes, edge_size=edge_size,
                                    pretrained_backbone=pretrained_backbone)
    else:
        if Pre_Trained_model_path is not None and os.path.exists(Pre_Trained_model_path):
            base_state_dict = torch.load(Pre_Trained_model_path)
        else:
            base_state_dict = 'timm'
            print('base_state_dict of timm')

        print('Test the PromptTuning of ', model_name)
        print('Prompt VPT type:', PromptTuning)
        model = build_promptmodel(num_classes, edge_size, model_name, Prompt_Token_num=Prompt_Token_num,
                                  VPT_type=PromptTuning, base_state_dict=base_state_dict)

    try:
        if PromptTuning is None:
            model.load_state_dict(torch.load(save_model_path))
        else:
            if PromptUnFreeze:
                model.load_state_dict(torch.load(save_model_path))
            else:
                model.load_prompt(torch.load(save_model_path))

        print("model loaded :", model_name)

    except Exception as e:
        try:
            model = nn.DataParallel(model)

            if PromptTuning is None:
                model.load_state_dict(torch.load(save_model_path))
            else:
                if PromptUnFreeze:
                    model.load_state_dict(torch.load(save_model_path))
                else:
                    model.load_prompt(torch.load(save_model_path))
            print("DataParallel model loaded")
        except:
            print(f"model loading error: {e}")
            return -1

    device_idx = 'cuda'
    if gpu_idx == -1:
        if torch.cuda.device_count() > 1:
            print("Use", torch.cuda.device_count(), "GPUs!")
            # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
            model = nn.DataParallel(model)
    else:
        device_idx = "cuda:" + str(gpu_idx).strip()

    device = torch.device(device_idx if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    model.to(device)

    # start tensorboard backend
    if enable_tensorboard:
        writer = SummaryWriter(runs_path)
    else:
        writer = None

    # if u run locally
    # nohup tensorboard --logdir=/home/MSHT/runs --host=0.0.0.0 --port=7777 &
    # tensorboard --logdir=/home/ZTY/runs --host=0.0.0.0 --port=7777

    print("*********************************{}*************************************".format('tasks_to_run'))
    print(args)

    test_model(model, test_dataloader, criterion, class_names, test_dataset_size, model_name=model_name,
               test_model_name=test_model_name, edge_size=edge_size, check_minibatch=check_minibatch,
               device=device, draw_path=runs_path, enable_attention_check=enable_attention_check,
               enable_visualize_check=enable_visualize_check, writer=writer)

    # if args.save_to_mysql:
    #     # 保存结果到数据库
    #     from Experiments.MySQLWrapper import MySQLWrapper
    #     mysql_wrapper = MySQLWrapper()
    #     mysql_wrapper.insert_data(table_name=run_name,
    #                               column_name=('model_name', f'LR1e{}'),
    #                               data=(''))


def get_args_parser():
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')

    # Model Name or index
    parser.add_argument('--model_name', default='ViT', type=str, help='Model Name or index')
    # Model tag (for example k-fold)
    parser.add_argument("--tag", default=None, type=str, help="Model tag (for example 5-fold)")

    # Enviroment parameters
    parser.add_argument('--gpu_idx', default=0, type=int,
                        help='use a single GPU with its index, -1 to use multiple GPU')

    # Path parameters
    parser.add_argument('--data_path', default=r'/data/pancreatic-cancer-project/k5_dataset',
                        help='path to dataset')
    parser.add_argument('--save_model_path', default=None,
                        help='path to save model state-dict')
    parser.add_argument('--runs_path', default=r'/home/pancreatic-cancer-project/runs',
                        help='path to draw and save tensorboard output')
    # model_path_by_hand
    parser.add_argument('--model_path_by_hand', default=None, type=str, help='path to a model state-dict')

    # Help tool parameters
    parser.add_argument('--paint', action='store_false', help='paint in front desk')  # matplotlib.use('Agg')
    parser.add_argument('--enable_notify', action='store_true', help='enable notify to send email')
    # check tool parameters
    parser.add_argument('--enable_tensorboard', action='store_true', help='enable tensorboard to save status')

    parser.add_argument('--enable_attention_check', action='store_true', help='check and save attention map')
    parser.add_argument('--enable_visualize_check', action='store_true', help='check and save pics')

    parser.add_argument('--data_augmentation_mode', default=-1, type=int, help='data_augmentation_mode')

    # PromptTuning
    parser.add_argument('--PromptTuning', default=None, type=str,
                        help='use Prompt Tuning strategy instead of Finetuning')
    # Prompt_Token_num
    parser.add_argument('--Prompt_Token_num', default=20, type=int, help='Prompt_Token_num')
    # PromptUnFreeze
    parser.add_argument('--PromptUnFreeze', action='store_true', help='prompt tuning with all parameaters un-freezed')
    # prompt model basic model path
    parser.add_argument('--Pre_Trained_model_path', default=None, type=str,
                        help='Finetuning a trained model in this dataset')

    # Dataset based parameters
    parser.add_argument('--num_classes', default=0, type=int, help='classification number, default 0 for auto-fit')
    parser.add_argument('--edge_size', default=224, type=int, help='edge size of input image')  # 224 256 384 1000

    # Test tasks_to_run parameters
    parser.add_argument('--batch_size', default=1, type=int, help='testing batch_size default 1')
    # check_minibatch for painting pics
    parser.add_argument('--check_minibatch', default=None, type=int, help='check batch_size')

    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)
