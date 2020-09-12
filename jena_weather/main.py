import os
import numpy as np

from generator import generator

# a timestep is 10 minutes
# problem formulation: given data going as far back as "lookback" timesteps and sampled
# every "steps" timesteps, can you predict the temperature "delay" timesteps ahead?

# use a Python generator that yields batches of data from the recent past, along with a
# target temperature in the future.

def main():
    data_dir = os.path.join(os.getcwd(), 'data')
    fname = os.path.join(data_dir, 'jena_climate_2009_2016.csv')
    f = open(fname)
    data = f.read()
    f.close()

    lines = data.split('\n')
    header = lines[0].split(',')
    lines = lines[1:]
    print(header)
    print(len(lines))

    float_data = np.zeros((len(lines), len(header)-1))
    for i, line in enumerate(lines):
        # remove date and time
        values = [float(x) for x in line.split(',')[1:]]
        float_data[i, :] = values

    max_index = 200000

    # data normalization
    mean = float_data[:max_index].mean(axis=0)
    float_data = float_data - mean
    std = float_data[:max_index].mean(axis=0)
    float_data = float_data/std

    lookback = 1440 # 10 days
    delay = 144 # target is 1 day in the future
    step = 6
    batch_size = 128

    val_indices = [200001, 300000]
    test_indices = [300001, None]

    train_gen = generator(float_data,
                          lookback=lookback,
                          delay=delay,
                          min_index=0,
                          max_index=max_index,
                          shuffle=True,
                          step=step,
                          batch_size=batch_size)

    val_gen = generator(float_data,
                        lookback=lookback,
                        delay=delay,
                        min_index=val_indices[0],
                        max_index=val_indices[1],
                        step=step,
                        batch_size=batch_size)

    test_gen = generator(float_data,
                         lookback=lookback,
                         delay=delay,
                         min_index=test_indices[0],
                         max_index=test_indices[1],
                         step=step,
                         batch_size=batch_size)

    # total steps to draw from the generators to see the entire set
    val_steps = (val_indices[1] - val_indices[0] - lookback)
    test_steps = (len(float_data) - test_indices[0] - lookback)


if __name__ == "__main__":
    main()