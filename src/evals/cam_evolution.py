#!/usr/bin/env python3

from torchvision.transforms.functional import resize
import matplotlib.animation as animation
import matplotlib.pyplot as plt
from src import const
import random
import torch
import sys


if __name__ == '__main__':
    path = const.DATA_DIR / 'evals' / 'head_to_heads' / sys.argv[1]
    benchmark_batch = torch.load(path / 'benchmark_batch.pt', weights_only=False, map_location='cpu')
    idx = random.randint(0, benchmark_batch.size(0))
    benchmark_batch = benchmark_batch[idx].permute(1, 2, 0)
    print(idx)

    fig = plt.figure(facecolor='white', figsize=(9, 4))
    ax1 = fig.add_subplot(1, 3, 2)
    ax2 = fig.add_subplot(1, 3, 3)

    frames = []
    for i in range(150):
        cam1 = torch.load(path / 'contrastive' / f'cams_{i}.pt', map_location='cpu', weights_only=False)[idx]
        cam2 = torch.load(path / 'default' / f'cams_{i}.pt', map_location='cpu', weights_only=False)[idx]

        cc1 = (cam1 - cam1[0]).abs().sum(0)
        cc2 = (cam2 - cam2[0]).abs().sum(0)

        frames.append([ax1.imshow(benchmark_batch, cmap='jet', alpha=0.5, animated=True),
                       ax1.imshow(resize(cc1[None, None], const.IMAGE_SIZE)[0, 0], cmap='jet', alpha=0.5, animated=True),
                       ax2.imshow(benchmark_batch, cmap='jet', alpha=0.5, animated=True),
                       ax2.imshow(resize(cc2[None, None], const.IMAGE_SIZE)[0, 0], cmap='jet', alpha=0.5, animated=True)])

    ax = fig.add_subplot(1, 3, 1)
    ax.imshow(benchmark_batch)

    ax1.set_xlabel('Interpretability-Constrained \n Cross-Entropy')
    ax2.set_xlabel('Standard Cross-Entropy')

    plt.suptitle('\nProvably-Faithful Attribution Maps')

    ani = animation.ArtistAnimation(fig, frames, interval=100, blit=True)
    ani.save(path / 'improvement_comparison.mp4')

    plt.tight_layout()
    plt.show()
