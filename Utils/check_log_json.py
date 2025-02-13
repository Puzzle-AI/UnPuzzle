"""
Organize log and write output excel   Script  ver： Feb 2nd 17:30

Notice:
you can enable_notify to send email
"""

import argparse
import json

import sys
import os

# Add the parent directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from Utils.metrics import *  # Now it works


def calculate_summary(log_dict, class_names, cls_idx):
    tp = log_dict[class_names[cls_idx]]['tp']
    tn = log_dict[class_names[cls_idx]]['tn']
    fp = log_dict[class_names[cls_idx]]['fp']
    fn = log_dict[class_names[cls_idx]]['fn']
    tp_plus_fp = tp + fp
    tp_plus_fn = tp + fn
    fp_plus_tn = fp + tn
    fn_plus_tn = fn + tn

    # precision
    if tp_plus_fp == 0:
        precision = 0
    else:
        precision = float(tp) / tp_plus_fp * 100
    # recall
    if tp_plus_fn == 0:
        recall = 0
    else:
        recall = float(tp) / tp_plus_fn * 100

    # TPR (sensitivity)
    TPR = recall

    # TNR (specificity)
    # FPR
    if fp_plus_tn == 0:
        TNR = 0
        FPR = 0
    else:
        TNR = tn / fp_plus_tn * 100
        FPR = fp / fp_plus_tn * 100

    # NPV
    if fn_plus_tn == 0:
        NPV = 0
    else:
        NPV = tn / fn_plus_tn * 100

    print('{} precision: {:.4f}  recall: {:.4f}'.format(class_names[cls_idx], precision, recall))
    print('{} sensitivity: {:.4f}  specificity: {:.4f}'.format(class_names[cls_idx], TPR, TNR))
    print('{} FPR: {:.4f}  NPV: {:.4f}'.format(class_names[cls_idx], FPR, NPV))
    print('{} TP: {}'.format(class_names[cls_idx], tp))
    print('{} TN: {}'.format(class_names[cls_idx], tn))
    print('{} FP: {}'.format(class_names[cls_idx], fp))
    print('{} FN: {}'.format(class_names[cls_idx], fn))

    # return for tensorboard log writer
    return precision, recall


class CLS_JSON_logger():
    """
    This is the logger class for classification training and testing records

    USAGE:
    1. init the logger at beginning of training
    2. init_epoch at beginning of every epoch
    3. update_confusion_matrix after the (preds, labels) is obtained within a specific phase (Train Val or Test)
    4. update_epoch_phase after all update_confusion_matrix within the same phase of the specific epoch
    5. dump_json at the end of training

    Notice:
    we design this to cover the different stages (phase) of each epoch: 'Train', 'Tal', 'Test'
    the epoch should start with 1, you can use +1 if your epoch is inx from 0
    in the test loop, there is only one epoch = 'Test' and one phase='Test'

    """
    def __init__(self, class_names, draw_path, check_model_name):
        """
        :param class_names: a list of the classification names
        :param draw_path: the runs path of storing the log
        :param check_model_name: the name of model
        """
        # a list of cls names
        self.class_names = class_names
        self.draw_path = draw_path
        self.check_model_name = check_model_name

        # initiate the empty json dict
        self.json_log = {}

    def init_epoch(self, epoch):
        # create empty epoch log
        self.json_log[str(epoch)] = {}
        # initiate the empty matrix_dict
        self.epoch_cls_task_matrix_dict = {}
        for cls_idx in range(len(self.class_names)):
            self.epoch_cls_task_matrix_dict[self.class_names[cls_idx]] = {'tp': 0.0, 'tn': 0.0, 'fp': 0.0, 'fn': 0.0}

    def update_confusion_matrix(self, preds, labels):
        # Compute confusion matrix for each class.
        for cls_idx in range(len(self.class_names)):
            # NOTICE remember to put tensor back to cpu
            tp = np.dot((labels.cpu().data == cls_idx).numpy().astype(int),
                        (preds == cls_idx).cpu().numpy().astype(int))
            tn = np.dot((labels.cpu().data != cls_idx).numpy().astype(int),
                        (preds != cls_idx).cpu().numpy().astype(int))

            fp = np.sum((preds == cls_idx).cpu().numpy()) - tp

            fn = np.sum((labels.cpu().data == cls_idx).numpy()) - tp

            # log_dict[cls_idx] = {'tp': 0, 'tn': 0, 'fp': 0, 'fn': 0}
            self.epoch_cls_task_matrix_dict[self.class_names[cls_idx]]['tp'] += tp
            self.epoch_cls_task_matrix_dict[self.class_names[cls_idx]]['tn'] += tn
            self.epoch_cls_task_matrix_dict[self.class_names[cls_idx]]['fp'] += fp
            self.epoch_cls_task_matrix_dict[self.class_names[cls_idx]]['fn'] += fn

    def update_epoch_phase(self, epoch, phase, tensorboard_writer=None):
        # json log: update
        self.json_log[str(epoch)][phase] = self.epoch_cls_task_matrix_dict

        for cls_idx in range(len(self.class_names)):
            # calculating the confusion matrix
            precision, recall = calculate_summary(self.epoch_cls_task_matrix_dict, self.class_names, cls_idx)
            # attach the records to the tensorboard backend
            if tensorboard_writer is not None:
                # ...log the running loss
                tensorboard_writer.add_scalar(phase + '   ' + self.class_names[cls_idx] + ' precision',
                                              precision, int(epoch))  # already epoch +1
                tensorboard_writer.add_scalar(phase + '   ' + self.class_names[cls_idx] + ' recall',
                                              recall, int(epoch))  # already epoch +1
        # flush matrix_dict
        self.epoch_cls_task_matrix_dict = {}
        for cls_idx in range(len(self.class_names)):
            self.epoch_cls_task_matrix_dict[self.class_names[cls_idx]] = {'tp': 0.0, 'tn': 0.0, 'fp': 0.0, 'fn': 0.0}

        # Offer api for access the current epoch_cls_task_matrix_dict
        return self.json_log[str(epoch)][phase]

    def dump_json(self):
        # save json_log  indent=2 for better view
        json.dump(self.json_log,
                  open(os.path.join(self.draw_path, self.check_model_name + '_log.json'), 'w'),
                  ensure_ascii=False,
                  indent=2)


def find_all_files(root, suffix=None):
    """
    return a list of the files path with a certain suffix
    """
    res = []
    for root, _, files in os.walk(root):
        for f in files:
            if suffix is not None and not f.endswith(suffix):
                continue
            res.append(os.path.join(root, f))
    return res


# Fixme this one is a temporary tool for JSON log decoding
def read_a_json_log(json_path, record_dir):
    """

    """
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    with open(json_path) as f:
        load_dict = json.load(f)
        # print(load_dict)
        epoch_num = len(load_dict)
        try:
            cls_list = [str(cls) for cls in load_dict[str(1)]['Train']]
            test_status = False
        except:
            cls_list = [str(cls) for cls in load_dict['Test']['Test']]
            test_status = True
        else:
            pass
        cls_num = len(cls_list)

        indicator_list = ['Precision', 'Recall', 'Sensitivity', 'Specificity', 'NPV', 'F1_score']
        indicator_num = len(indicator_list)

        blank_num = cls_num * indicator_num
        first_blank_num = blank_num // 2

        empty_str1 = ' ,'  # 对齐Acc
        for i in range(0, first_blank_num):
            empty_str1 += ' ,'

        empty_str2 = ''
        for i in range(0, blank_num):
            empty_str2 += ' ,'

        result_csv_name = os.path.split(json_path)[1].split('.')[0] + '.csv'
        result_indicators = [os.path.split(json_path)[1].split('.')[0], ]  # 第一个位置留给model name

    with open(os.path.join(record_dir, result_csv_name), 'w') as f_log:
        if test_status:
            # 写头文件1
            f_log.write('Phase:,' + empty_str1 + ' Test\n')
            head = 'Epoch:, '
            class_head = 'Acc, '  # 目标 'Acc, '+ 类别* indicator_list
            for cls in cls_list:
                for indicator in indicator_list:
                    class_head += cls + '_' + indicator + ', '

            # 写头文件2
            f_log.write(head + class_head + '\n')  # Test
            f_log.close()

        else:
            # 写头文件1
            f_log.write('Phase:,' + empty_str1 + ' Train' + empty_str2 + ' Val\n')

            head = 'Epoch:, '
            class_head = 'Acc, '  # 目标 'Acc, '+ 类别* indicator_list
            for cls in cls_list:
                for indicator in indicator_list:
                    class_head += cls + '_' + indicator + ', '

            # 写头文件2
            f_log.write(head + class_head + class_head + '\n')  # Train val
            f_log.close()

    # 初始化最佳
    best_val_acc = 0.0

    for epoch in range(1, epoch_num + 1):
        if test_status:
            epoch = 'Test'
        epoch_indicators = [epoch, ]  # 第一个位置留给epoch

        for phase in ['Train', 'Val']:
            if test_status:
                phase = 'Test'

            sum_tp = 0.0

            phase_indicators = [0.0, ]  # 第一个位置留给ACC

            for cls in cls_list:
                log = load_dict[str(epoch)][phase][cls]
                tp = log['tp']
                tn = log['tn']
                fp = log['fp']
                fn = log['fn']

                sum_tp += tp

                Precision = compute_precision(tp, fp)
                Recall = compute_recall(tp, fn)

                Sensitivity = compute_sensitivity(tp, fn)
                Specificity = compute_specificity(tn, fp)

                NPV = compute_NPV(tn, fn)
                F1_score = compute_f1_score(tp, tn, fp, fn)

                cls_indicators = [Precision, Recall, Sensitivity, Specificity, NPV, F1_score]
                phase_indicators.extend(cls_indicators)

            Acc = 100 * (sum_tp / float(tp + tn + fn + fp))  # 直接取最后一个的tp tn fn fp 算总数就行
            phase_indicators[0] = Acc

            epoch_indicators.extend(phase_indicators)

            if Acc >= best_val_acc and phase == 'Val':
                best_val_acc = Acc
                best_epoch_indicators = epoch_indicators

            elif test_status:
                with open(os.path.join(record_dir, result_csv_name), 'a') as f_log:
                    for i in epoch_indicators:
                        f_log.write(str(i) + ', ')
                    f_log.write('\n')
                    f_log.close()
                result_indicators.extend(epoch_indicators)
                return result_indicators  # 结束 返回test的log行
            else:
                pass

        # epoch_indicators
        with open(os.path.join(record_dir, result_csv_name), 'a') as f_log:
            for i in epoch_indicators:
                f_log.write(str(i) + ', ')
            f_log.write('\n')

    with open(os.path.join(record_dir, result_csv_name), 'a') as f_log:
        f_log.write('\n')
        f_log.write('\n')
        # 写头文件1
        f_log.write('Phase:,' + empty_str1 + ' Train' + empty_str2 + ' Val\n')
        # 写头文件2
        f_log.write('Best Epoch:, ' + class_head + class_head + '\n')  # Train val

        try:
            for i in best_epoch_indicators:
                f_log.write(str(i) + ', ')
            f_log.close()
            result_indicators.extend(best_epoch_indicators)
            return result_indicators  # 结束 返回best epoch行
        except:
            print('No best_epoch_indicators')
            return result_indicators  # 结束


def read_all_logs(logs_path, record_dir):
    if not os.path.exists(record_dir):
        os.makedirs(record_dir)

    res = find_all_files(logs_path, suffix='.json')

    result_csv_name = os.path.split(logs_path)[1] + '.csv'

    with open(os.path.join(record_dir, result_csv_name), 'w') as f_log:
        for json_path in res:
            result_indicators = read_a_json_log(json_path, record_dir)  # best_epoch_indicators of a model json log

            for i in result_indicators:
                f_log.write(str(i) + ', ')
            f_log.write('\n')
        f_log.close()

    print('record_dir:', record_dir)


def main(args):
    ONE_LOG = args.ONE_LOG
    runs_root = args.runs_root
    record_dir = args.record_dir

    enable_notify = args.enable_notify  # False

    if ONE_LOG:
        # this will set path just for one file
        read_a_json_log(runs_root, record_dir)
    else:
        read_all_logs(runs_root, record_dir)

    # this is for returning the files with email
    if enable_notify:
        import notifyemail as notify

        notify.Reboost(mail_host='smtp.163.com', mail_user='tum9598@163.com', mail_pass='xxxx',
                       default_reciving_list=[args.reciver],  # change here if u want to use notify
                       log_root_path='log', max_log_cnt=5)

        notify.add_text('  ')

        notify.add_text('PATH: ' + str(runs_root))
        notify.add_text('  ')

        if not ONE_LOG:
            for Experiment_idx in os.listdir(runs_root):
                notify.add_text('Experiment idxs: ' + str(Experiment_idx))
                notify.add_text('  ')

        notify.add_file(record_dir)
        notify.send_log()


def get_args_parser():
    parser = argparse.ArgumentParser(description='Log checker')

    parser.add_argument('--ONE_LOG', action='store_true', help='check only one LOG')

    parser.add_argument('--runs_root', default=r'../../../../Downloads/runs',
                        help='path of the drawn and saved tensorboard output')

    parser.add_argument('--record_dir', default=r'../../../../Downloads/runs/CSV_logs',
                        help='path to save csv log output')

    parser.add_argument('--enable_notify', action='store_true', help='enable notify to send email')
    parser.add_argument('--reciver', default='tum9598@163.com', type=str, help='notify default email')

    return parser


if __name__ == '__main__':
    parser = get_args_parser()
    args = parser.parse_args()
    main(args)