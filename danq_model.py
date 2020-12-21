import tensorflow.keras.layers as L
import tensorflow.keras.metrics as M
import tensorflow as tf

class DanQ:
    def __init__(self, model_params, outputs, auto_build=False):
        self.model = None
        self.is_trained = False
        self.strategy = tf.distribute.MirroredStrategy()
        self.history_file = outputs['HISTORY']
        self.log_file = outputs['LOG']
        self.results_file = outputs['RESULTS']

        # BATCH_SIZE controls the number of training examples fed through the network per pass.
        #   Reducing this number will decrease training stability but can be beneficial for quicker training cycles.
        #   Increasing this number too significantly can cause a degradation in model quality.
        #   Ideally, hyperparameter tuning is run and a range of 32-150 is used for this to find the optimal batch size.
        #   During my experiments, having a batch size larger than 50 causes instabilities in memory and can cause the program to crash unexpectedly when running full dataset.
        #   In the original DanQ paper, this was set to 100
        self.batch_size = model_params['MINIBATCH_SIZE']

        # EPOCHS controls the numbr of times the training process runs through the entire data set
        #   Reducing this number will decrease the accuracy (i.e. a higher training loss)
        #   Increasing this too much will result in the model overfitting the data, with a corresponding increase in validation loss
        #   Ideally, hyperparemeter tuning is run and a range of 10-30 is used to find the point at which validation loss begins to increase
        #   In the original DanQ paper, this was set to 60, but in their original code base it was set to 20
        self.epochs = model_params['EPOCHS']

        # Loss function applied at each step, DanQ uses 'binary_entropy' as it is appropriate for the binary classification problem we are working
        self.loss_function = model_params['LOSS']

        # Optimizer used at each step, DanQ original used rmsprop, but other optimizers such as adam might be more successful here
        # Note: No learning rate was specified in the original DanQ paper, but this is pretty important to the training process
        self.optimizer = model_params['OPTIMIZER']

        # How many epochs to keep running if expected metrocs don't improve, after this number, the training process will give up and conclude that it's found its best weights
        # Since our implementation doesn't appear to train well to match the original DanQ metrics, we set patience to higher a number to allow for more attempts
        # Interestingly, the metric used to evaluate performance improvement, 'val_loss', doesn't appear to improve much after the first few epochs.
        self.patience = model_params['PATIENCE']

        # How much output to provide, 0 is minimal/none, 1 is standard, 2 is for specific instances only (don't use) 
        self.verbose = model_params['VERBOSE_OUTPUT']
        
        if auto_build:
            self.build_layers()

    def build_layers(self, auto_compile=True, distributed=True):
        # The model needs to be built with a distributed strategy if distributed training is required
        # Right now this code is only set up for distributed training as its not possible to train DanQ in any reasonable amount of time without it
        # TODO: Implement non-distributed training as well
        with self.strategy.scope():
            self.model = tf.keras.Sequential()
            self.model.add(L.Convolution1D(input_shape=(1000,4),
                                filters=320,
                                kernel_size=26,
                                padding="valid", 
                                activation="relu"))
            self.model.add(L.MaxPooling1D(pool_size=13, strides=13))
            self.model.add(L.Dropout(0.2))
            self.model.add(L.Bidirectional(L.LSTM(320, return_sequences=True)))
            self.model.add(L.Dropout(0.5))
            self.model.add(L.Flatten())
            self.model.add(L.Dense(925))
            self.model.add(L.Activation('relu'))
            self.model.add(L.Dense(919))
            self.model.add(L.Activation('sigmoid'))
            if auto_compile:
                self.compile_model()
            return True
        return False

    def compile_model(self):
        # Compile and prepare the model for training/testing
        if self.model is not None:
            self.model.compile(loss=self.loss_function, optimizer=self.optimizer, metrics=['binary_accuracy', M.AUC(), M.Precision(), M.Recall()])
            return True
        return False

    def train(self, train_x, train_y, valid_x, valid_y, output_model_file, distributed=True, save_at_checkpoints=True, early_stopping=True, use_existing=True):
        # Train the network using a multi-GPU environment
        # TODO: Allow for option to run non-distributed
        if self.model is not None:
            # If flagged, try to load a saved model and use that instead for model weights, no need to train again
            if use_existing:
                self.load_saved(output_model_file)
                self.is_trained = True
            else:
                if distributed:
                    with self.strategy.scope():
                        callbacks = []
                        if save_at_checkpoints: 
                            callbacks.append(tf.keras.callbacks.ModelCheckpoint(filepath=output_model_file, verbose=self.verbose, save_best_only=True))
                        if early_stopping:
                            callbacks.append(tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, verbose=self.verbose))
                        
                        # Run training        
                        print("Training model...")
                        history = self.model.fit(train_x, 
                                    train_y, 
                                    batch_size=self.batch_size, 
                                    epochs=self.epochs, 
                                    shuffle=True,
                                    verbose=self.verbose,
                                    validation_data=(valid_x, valid_y),
                                    callbacks=callbacks)
                        print(f"... Done")  
                        with open(self.history_file, "w") as out_file:
                            out_file.write(str(history.history))
                        print(f"Training history saved to {self.history_file}")
                        self.is_trained = True
                        return True
        return False
        

    def test(self, test_x, test_y, distributed=True):
        # Test the network's performance on unseen data
        if not self.is_trained:
            print(f"Model is not trained yet, use train() function to train model or load_model() to load a savd model")
            return False
        if distributed:
            with self.strategy.scope():
                print(f"Evaluating model... ")
                results = self.model.evaluate(test_x, test_y, return_dict=True)
                # results = model.predict(test_data_x) # use this instead to run predictions
                print(f"... Done")

                # Output results to file
                with open(self.results_file, "w") as out_file:
                    out_file.write(str(results))
                    return True
        return False

    def print_model(self):
        # Output model layers, variables, etc.
        if self.model is not None:
            print(self.model.summary())
            return True
        return False

    def load_saved(self, model_file):
        if self.model is not None:
            print(f"Loading saved model...")
            self.model.load_weights(model_file)
            print(f"...DONE")
            return True
        return False

    def save_model(self):
        # Right now we are auto-saving the model when a better metric is achieved, so there is no need to implment this
        # TODO: Enable capability to save different models when hyperparameter tuning
        pass
