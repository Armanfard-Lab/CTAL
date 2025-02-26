# Hanrong Ye

class PerformanceMeter(object):
    """ A general performance meter which shows performance across one or more tasks """
    def __init__(self, opt, tasks):
        self.database = opt['train_db_name']
        self.tasks = tasks
        self.meters = {t: get_single_task_meter(opt, self.database, t) for t in self.tasks}

    def reset(self):
        for t in self.tasks:
            self.meters[t].reset()

    def update(self, pred, gt):
        for t in self.tasks:
            self.meters[t].update(pred[t], gt[t])

    def get_score(self):
        eval_dict = {}
        for t in self.tasks:
            eval_dict[t] = self.meters[t].get_score()

        return eval_dict

def get_single_task_meter(opt, database, task):
    """ Retrieve a meter to measure the single-task performance """

    # ignore index based on transforms.AddIgnoreRegions
    if task == 'semseg':
        from evaluation.eval_semseg import SemsegMeter
        return SemsegMeter(num_classes=opt.TASKS.NUM_OUTPUT['semseg'])

    elif task == 'normals':
        from evaluation.eval_normals import NormalsMeter
        return NormalsMeter() 

    elif task == 'depth':
        from evaluation.eval_depth import DepthMeter
        # Set effective depth evaluation range. Refer to:
        # https://github.com/sjsu-smart-lab/Self-supervised-Monocular-Trained-Depth-Estimation-using-Self-attention-and-Discrete-Disparity-Volum/blob/3c6f46ab03cfd424b677dfeb0c4a45d6269415a9/evaluate_city_depth.py#L55
        return DepthMeter() 

    else:
        raise NotImplementedError