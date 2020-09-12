import numpy as np
from tqdm import tqdm
from keras.models import Sequential
from keras import layers

def baseline_model(val_steps, val_gen, temp_std):
    """
    Very slow. Average absolute error of 2.57 C.
    """
    batch_maes = []
    for step in tqdm(range(val_steps)):
        samples, targets = next(val_gen)
        preds = samples[:, -1, -1]
        mae = np.mean(np.abs(preds - targets))
        batch_maes.append(mae)
    ave_mae = np.mean(batch_meas)
    print("baseline model error: ", ave_mae*temp_std, "C")

def densely_connected_model(lookback, step, n_features):
    """
    Validation loss values hover around 0.30, pretty much the same as baseline.
    """
    model = Sequential()
    model.add(layers.Flatten(input_shape=(lookback//step, n_features)))
    model.add(layers.Dense(32, activation='relu'))
    model.add(layers.Dense(1))
    return model