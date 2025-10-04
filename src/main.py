from .data import train_val_test_split
from .models import CascadeModel
import numpy as np
if __name__ == "__main__":

    data_folder = "./data"
    x_train, y_train, x_val, y_val, x_test, y_test = train_val_test_split(data_folder,random_seed=1)
    model = CascadeModel(sx=x_train.shape[1], sy=x_train.shape[2], weights_path=["model1_trained.weights.h5", "model2_trained.weights.h5", None])
    model.train(x_train, y_train, x_val, y_val, epochs=50, batch_size=4)
    print("Final Evaluation on Test Set:")
    model.evaluate(x_test, y_test)
