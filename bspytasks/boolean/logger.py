import torchvision
from torch.utils.tensorboard import SummaryWriter


class Logger():
    def __init__(self, log_dir, comment='DEFAULT_LOGGER'):
        # TODO: LOG HIPERPARAMETERS IN THE COMMENT e.g. "LR_0.1_BATCH_16"
        self.log = SummaryWriter(log_dir, comment=comment)
        self.gate = ''

    def log_train_inputs(self, inputs, targets):
        # self.log.add_graph(net, images)
        pass

    def log_train_predictions(self, predictions):
        pass
        # self.log.add_histogram(
        #     'Predictions', predictions)
        # if i % 1000 == 0:
        #     grid = torchvision.utils.make_grid(inputs)
        #     self.log.add_image('input_images', grid)

    def log_ios_train(self, inputs, targets, predictions, epoch):
        # import matplotlib.pyplot as plt
        # fig = plt.figure()
        # #plt.title(gate_name + ' Veredict:' + str(veredict))
        # plt.plot(predictions.clone().detach().cpu())
        # plt.plot(targets.copy().detach().cpu())
        # plt.ylabel('Current (nA)')
        # plt.xlabel('Time')
        # self.log.add_figure(f'test/' + str(epoch), fig)
        # if save_dir is not None:
        #     plt.savefig(save_dir)
        # if show_plots:
        #     plt.show()
        # plt.close()
        pass

    def log_val_predictions(self, inputs, targets):
        pass

    def log_performance(self, train_losses, val_losses, epoch):
        if val_losses == []:
            self.log.add_scalar('Cost/train/' + self.gate, train_losses[-1], epoch)
        else:
            self.log.add_scalars(
                'Cost/' + self.gate, {'train': train_losses[-1], 'dev': val_losses[-1]}, epoch)

    def log_outputs(self, outputs):
        # self.log.add_histogram(
        #     'Output steering angle histogram', outputs)
        pass

    def close(self):
        self.log.close()
