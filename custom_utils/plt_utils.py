import matplotlib.pyplot as plt
import os
import numpy as np

plt.ioff()

def draw_loss(loss_array, name_array, iter_loss, output_dir, plt_name):
    plt.figure(figsize=(10, 5))
    plt.title(f"{plt_name} During Training")
    for loss,name in zip(loss_array,name_array):
        plt.plot(loss, label=name)
    plt.xlabel("iterations / %d" % (iter_loss))
    plt.ylabel("loss")
    plt.legend()
    plt.savefig(os.path.join(output_dir,f"{plt_name}.png"))