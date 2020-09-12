import numpy as np


def generator(data, lookback=0, delay=0, min_index=0, max_index=0,
              shuffle=False, batch_size=128, step=6):
    if max_index is None:
        # max allowable index
        # must have a buffer of "delay" steps ahead
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        # shuffle: randomly selects batch_size many indices
        # between min_index+lookback and max_index
        if shuffle:
            rows = np.random.randint(min_index + lookback,
                                     max_index,
                                     size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i, min(i + batch_size, max_index))
            i = i + len(rows)

        samples = np.zeros((len(rows),
                           lookback // step,
                           data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j, row in enumerate(rows):
            indices = range(rows[j] - lookback, rows[j], step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]

        yield samples, targets