import sys
import visdom
import numpy as np

if sys.version_info[0] == 2:
    VisdomExceptionBase = Exception
else:
    VisdomExceptionBase = ConnectionError

class Visualizer():
    """
    Visualization for results during the training stage
    """
    def __init__(self, display_id):
        if display_id > 0:
            self.vis = visdom.Visdom()

    def show_current_images(self, image_dict):
        """
        display images
        :param images: images during the training stage  (name, images)
        :return:
        """
        for image_name in image_dict.keys():

            self.vis.image(image_dict[image_name], win=image_name, opts=dict(caption=image_name))

    def plot_current_losses(self, progress_ratio, losses):
        """display the current losses on visdom display: dictionary of error labels and values
        Parameters:
            counter_ratio (float) -- progress (percentage) in the current epoch, between 0 to 1
            losses (OrderedDict)  -- training losses stored in the format of (name, float) pairs
        """
        if not hasattr(self, 'plot_data'):
            self.plot_data = {'X': [], 'Y': [], 'legend': list(losses.keys())}
        self.plot_data['X'].append(progress_ratio)
        self.plot_data['Y'].append(losses["Diceloss"])
        try:
            self.vis.line(
                X=np.array(self.plot_data['X']),
                Y=np.array(self.plot_data['Y']),
                opts={
                    'title': ' loss over time',
                    'legend': self.plot_data['legend'],
                    'xlabel': 'epoch',
                    'ylabel': 'loss'},
                win="loss")
        except VisdomExceptionBase:
            pass
