from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, Dropout, concatenate
from tensorflow.keras.optimizers import Adam    
import tensorflow as tf
import numpy as np 

from .metrics import ComputeDice

nbf64 = 8
nbf128 = 16
nbf256 = 32 
nbf512 = 64
nbf1024 = 128


def unet(sx,sy,d):
    inputs = Input((sx, sy, d))
    conv1 = Conv2D(nbf64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(inputs)
    conv1 = Conv2D(nbf64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)
    conv2 = Conv2D(nbf128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool1)
    conv2 = Conv2D(nbf128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)
    conv3 = Conv2D(nbf256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool2)
    conv3 = Conv2D(nbf256, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)
    conv4 = Conv2D(nbf512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool3)
    conv4 = Conv2D(nbf512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv4)
    drop4 = Dropout(0.5)(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(drop4)
    conv5 = Conv2D(nbf1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(pool4)
    conv5 = Conv2D(nbf1024, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv5)
    drop5 = Dropout(0.5)(conv5)
    up6 = Conv2D(nbf512, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(drop5))
    merge6 = concatenate([up6, conv4], axis = 3)
    conv6 = Conv2D(nbf512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge6)
    conv6 = Conv2D(nbf512, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv6)
    up7 = Conv2D(nbf256, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv6))
    merge7 = concatenate([up7, conv3], axis = 3)
    conv7 = Conv2D(nbf256, 3, activation = 'relu',
                     padding = 'same', kernel_initializer = 'he_normal')(merge7)
    conv7 = Conv2D(nbf256, 3, activation = 'relu',
                     padding = 'same', kernel_initializer = 'he_normal')(conv7)
    up8 = Conv2D(nbf128, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv7))
    merge8 = concatenate([up8, conv2], axis = 3)
    conv8 = Conv2D(nbf128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge8)
    conv8 = Conv2D(nbf128, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv8)
    up9 = Conv2D(nbf64, 2, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(UpSampling2D(size = (2,2))(conv8))
    merge9 = concatenate([up9, conv1], axis = 3)
    conv9 = Conv2D(nbf64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(merge9)
    conv9 = Conv2D(nbf64, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv9 = Conv2D(2, 3, activation = 'relu', padding = 'same', kernel_initializer = 'he_normal')(conv9)
    conv10 = Conv2D(1, 1, activation = 'sigmoid')(conv9)
    model = Model(inputs = inputs, outputs = conv10)
    model.compile(optimizer = Adam(learning_rate = 1e-4), loss = 'binary_crossentropy', metrics = ['accuracy'])
#    model.summary()
    return model

def create_model(sx,sy,d):
    model = unet(sx,sy,d)
    return model

def load_model(sx,sy,d,weight_path):
    if weight_path is not None:
        model = unet(sx,sy,d)
        model.load_weights(weight_path)
        return model
    return None

class CascadeModel:
    def __init__(self,sx,sy, weights_path:list = [None, None, None]):
        self.models1 = load_model(sx,sy,2,weights_path[0]) if weights_path[0] is not None else create_model(sx,sy,2)
        self.models2 = load_model(sx,sy,3,weights_path[1]) if weights_path[1] is not None else create_model(sx,sy,3)
        self.models3 = load_model(sx,sy,4,weights_path[2]) if weights_path[2] is not None else create_model(sx,sy,4)
        self.sx = sx
        self.sy = sy
        self.ismodels1loaded = True if weights_path[0] is not None else False
        self.ismodels2loaded = True if weights_path[1] is not None else False
        self.ismodels3loaded = True if weights_path[2] is not None else False

    def train(self, x_train, y_train, x_val, y_val, epochs=10, batch_size=32):

        print("Training Model 1")
        if not self.ismodels1loaded:

                y_train_m1 = (y_train >= 10).astype(np.float32)
                y_val_m1 = (y_val >= 10).astype(np.float32)

                early_stopping1 = tf.keras.callbacks.EarlyStopping(
                    monitor='val_loss', patience=5, restore_best_weights=True
                )
                reduce_lr1 = tf.keras.callbacks.ReduceLROnPlateau(
                    monitor='val_loss', factor=0.5, patience=3, verbose=1
                )

                self.models1.fit(
                    x_train, y_train_m1,
                    validation_data=(x_val, y_val_m1),
                    epochs=epochs,
                    batch_size=batch_size,
                    callbacks=[early_stopping1, reduce_lr1]
                )
                self.models1.save_weights("model1_trained.weights.h5")
        output_train_models1 = self.models1.predict(x_train,verbose=0).squeeze(-1)
        output_val_models1 = self.models1.predict(x_val,verbose=0).squeeze(-1)

        x_train_models2 = np.concatenate([x_train, output_train_models1[..., np.newaxis]], axis=-1)
        x_val_models2 = np.concatenate([x_val, output_val_models1[..., np.newaxis]], axis=-1)

        print("Training Model 2")
        if not self.ismodels2loaded:
            y_train_models2 = np.where(y_train >= 150, 1, 0)
            y_val_models2 = np.where(y_val >= 150, 1, 0)
            early_stopping2 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            self.models2.fit(x_train_models2, y_train_models2, validation_data=(x_val_models2, y_val_models2), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping2])
            self.models2.save_weights("model2_trained.weights.h5")

        output_train_models2 = self.models2.predict(x_train_models2,verbose=0).squeeze(-1)
        output_val_models2 = self.models2.predict(x_val_models2,verbose=0).squeeze(-1)

        x_train_models3 = np.concatenate([x_train, output_train_models1[..., np.newaxis], output_train_models2[..., np.newaxis]], axis=-1)
        x_val_models3 = np.concatenate([x_val, output_val_models1[..., np.newaxis], output_val_models2[..., np.newaxis]], axis=-1)

        print("Training Model 3")
        if not self.ismodels3loaded:
            y_train_models3 = np.where(y_train == 250, 1, 0)
            y_val_models3 = np.where(y_val == 250, 1, 0)
            early_stopping3 = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
            self.models3.fit(x_train_models3, y_train_models3, validation_data=(x_val_models3, y_val_models3), epochs=epochs, batch_size=batch_size, callbacks=[early_stopping3])
            self.models3.save_weights("model3_trained.weights.h5")


        print("Final Evaluation on Training Set:")
        self.evaluate(x_train, y_train)
        print("Final Evaluation on Validation Set:")
        self.evaluate(x_val, y_val)

    
    def predict(self, x):
        output_models1 = self.models1.predict(x,verbose=0).squeeze(-1)
        x_models2 = np.concatenate([x, output_models1[..., np.newaxis]], axis=-1)
        output_models2 = self.models2.predict(x_models2,verbose=0).squeeze(-1)
        x_models3 = np.concatenate([x, output_models1[..., np.newaxis], output_models2[..., np.newaxis]], axis=-1)
        output_models3 = self.models3.predict(x_models3,verbose=0).squeeze(-1)

        y_pred_WM = np.zeros((x.shape[0], self.sx, self.sy), dtype=np.uint8)
        y_pred_GM = np.zeros((x.shape[0], self.sx, self.sy), dtype=np.uint8)
        y_pred_CSF = np.zeros((x.shape[0], self.sx, self.sy), dtype=np.uint8)

        y_pred_WM[output_models3 >= 0.5] = 250
        y_pred_GM[(output_models2 >= 0.5) & (output_models3 < 0.5)] = 150
        y_pred_CSF[(output_models1 >= 0.5) & (output_models2 < 0.5) ] = 10

        y_pred = y_pred_WM + y_pred_GM + y_pred_CSF
        return y_pred
    
    def evaluate(self, x, y):
        y_pred = self.predict(x)
        y = y.squeeze(-1)
        y_WM = (y == 250).astype(np.float32)
        y_GM = (y == 150).astype(np.float32)
        y_CSF = (y == 10).astype(np.float32)

        y_pred_WM = (y_pred == 250).astype(np.float32)
        y_pred_GM = (y_pred == 150).astype(np.float32)
        y_pred_CSF = (y_pred == 10).astype(np.float32)


        print("Model 1 (CSF vs no-CSF):")
        dice = ComputeDice(y_CSF,  y_pred_CSF)
        print(f"  Dice: {dice:.4f}")
        print("Model 2 (GM vs no-GM):")
        dice = ComputeDice(y_GM,  y_pred_GM)
        print(f"  Dice: {dice:.4f}")
        print("Model 3 (WM vs no-WM):")
        dice = ComputeDice(y_WM,  y_pred_WM)
        print(f"  Dice: {dice:.4f}")
        return dice