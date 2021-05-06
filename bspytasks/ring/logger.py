import torchvision
from torch.utils.tensorboard import SummaryWriter
import matplotlib.pyplot as plt
import torch

class Logger:
    def __init__(self, log_dir, comment="DEFAULT_LOGGER"):
        # TODO: LOG HIPERPARAMETERS IN THE COMMENT e.g. "LR_0.1_BATCH_16"
        self.log = SummaryWriter(log_dir, comment=comment)
        self.gate = ""

    def log_val(self, inputs, targets, predictions, model, epoch):
        pass
    def log_train(self, inputs, targets, predictions, model, epoch):
        pass

    def log_debug(self, name, inputs, targets, model): 
        with torch.no_grad():
            model.eval()
            model(inputs)   
        status = model.get_logged_variables()
        for key in status.keys():
            zeros = status[key][targets.squeeze()==0].detach().cpu()
            ones = status[key][targets.squeeze()==1].detach().cpu()
            if len(status[key].shape) > 1:
                
                for i in range(status[key].shape[1]):
                    self.log.add_histogram(name+'_'+key+'_'+str(i)+'/zeros',zeros[:,i])
                    self.log.add_histogram(name+'_'+key+'_'+str(i)+'/ones',ones[:,i])          
            else:
                self.log.add_histogram(name+'_'+key,status[key])

            ordered_data = torch.cat((status[key][targets.squeeze()==0],status[key][targets.squeeze()==1])).detach().cpu()               
            fig = plt.figure()
            
            #self.plot_output(status[key],targets,key+'_'+str(i))
            if key.split('_')[-1] == 'input' and key.split('_')[0] == 'l1':
                plt.scatter(ordered_data[:,0],ordered_data[:,1],c=targets.detach().cpu(), label='DNPU 0')
                plt.scatter(ordered_data[:,2],ordered_data[:,3],c=targets.detach().cpu(), label='DNPU 1')
            else:
                plt.plot(ordered_data,label=name+'_'+key, alpha=0.7, linestyle='-', marker='D')   
            plt.legend()
            plt.close(fig)
            self.log.add_figure(key,fig,close=False)
            
    def log_performance(self, train_losses, val_losses, epoch):
        if val_losses == []:
            self.log.add_scalar("Cost/train/" + self.gate, train_losses[-1], epoch)
        else:
            self.log.add_scalars(
                "Cost/" + self.gate,
                {"train": train_losses[-1], "dev": val_losses[-1]},
                epoch,
            )

    def close(self):
        self.log.close()
