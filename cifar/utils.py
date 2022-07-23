import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib

def histogram(args, loader, metrics, correct_mask, filename, title, x_label,
              min_val=0., max_val=1.0, labels=['correct', 'incorrect']):

    # plot the hist for correctly predicted samples
    n1, bins1, _ = plt.hist(metrics[np.arange(len(loader.dataset))[correct_mask]], label=labels[0],
                            bins=50, 
                            #range=(min_val, max_val), 
                            color='g', alpha=0.5)
    #plt.plot(bins1[:-1], smooth(n1), color='g')
    # plot the hist for incorrectly predicted samples
    n2, bins2, _ = plt.hist(metrics[np.arange(len(loader.dataset))[~correct_mask]], label=labels[1],
                            bins=50, 
                            #range=(min_val, max_val), 
                            color='b', alpha=0.5)
    #plt.plot(bins2[:-1], smooth(n2), color='b')

    plt.xlabel(x_label)
    plt.ylabel('Number of samples')
    plt.title(title)
    #plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.savefig(filename.split('.')[0] + '.pdf')

    # save tikz plot
    tikzplotlib.save(filename.split('.')[0] + '.tex',
                 axis_width='\\figurewidth',
                 axis_height='\\figureheight')
    plt.close()

def density_plot(args, loader, metrics, correct_mask, filename, title, x_label,
                 labels=['correct', 'incorrect']):
    sns.distplot(metrics[np.arange(len(loader.dataset))[correct_mask]], hist=False,
                kde=True, kde_kws = {'linewidth': 2, 'clip': (0, np.log(args.num_classes))}, label=labels[0],
                color='g')
    sns.distplot(metrics[np.arange(len(loader.dataset))[~correct_mask]], hist=False,
                kde=True, kde_kws = {'linewidth': 2, 'clip': (0, np.log(args.num_classes))}, label=labels[1],
                color='b')
    plt.xlabel(x_label)
    plt.ylabel('Density')
    plt.title(title)
    #plt.grid(True)
    #plt.xlim(0, np.log(args.num_classes))
    plt.legend()
    plt.savefig(filename)
    plt.savefig(filename.split('.')[0] + '.pdf')

    # save tikz plot
    tikzplotlib.save(filename.split('.')[0] + '.tex',
                 axis_width='\\figurewidth',
                 axis_height='\\figureheight')
    plt.close()
    
