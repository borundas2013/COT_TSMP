from CustomLoss import *
import tensorflow as tf
tf.config.run_functions_eagerly(True)

class PPOTrainer:
    def __init__(self, model, lambda_rl=0.1, clip_epsilon=0.2, lr=1e-4):
        self.strategy = tf.distribute.get_strategy()  # Get current strategy instead of creating new one
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.loss_fn = CustomMoleculeLoss(
            group_smarts_list=Constants.GROUP_VOCAB.keys(),
            reverse_vocab=None,
            lambda_rl=lambda_rl,
            clip_epsilon=clip_epsilon
        )

    @tf.function
    def distributed_train_step(self, inputs):
        per_replica_losses = self.strategy.run(self.train_step, args=(inputs,))
        return self.strategy.reduce(tf.distribute.ReduceOp.MEAN, per_replica_losses, axis=None)

    def log_predictions(self, predictions, batch_targets, epoch, batch_count, log_file):
        """Separate function for logging predictions (runs in eager mode)"""
        pred_smiles = self.decode_smiles_batch(predictions)
        input_smiles = self.decode_smiles_input(batch_targets)
        
        pred_groups = [extract_group_smarts(s) for s in pred_smiles]
        input_groups = [extract_group_smarts(s) for s in input_smiles]
        
        log_file.write(f"\nEpoch {epoch + 1}, Batch {batch_count + 1}:\n")
        for i in range(len(pred_smiles)):
            log_file.write(f"Input SMILES: {input_smiles[i]}\n")
            log_file.write(f"Predicted SMILES: {pred_smiles[i]}\n")
            log_file.write(f"Input Groups: {input_groups[i]}\n")
            log_file.write(f"Predicted Groups: {pred_groups[i]}\n")
            log_file.write("-" * 50 + "\n")
        log_file.flush()

    def train(self, dataset, num_epochs, batch_size, callbacks=None):
        dist_dataset = self.strategy.experimental_distribute_dataset(
            dataset.shuffle(buffer_size=1000).batch(batch_size)
        )
        
        log_file = open('training_predictions.log', 'w')
        
        if callbacks:
            for callback in callbacks:
                callback.set_model(self.model)
                callback.on_train_begin()
        
        try:
            for epoch in range(num_epochs):
                if callbacks:
                    for callback in callbacks:
                        callback.on_epoch_begin(epoch)
                
                epoch_logs = {}
                total_loss = 0
                batch_count = 0
                
                for batch in dist_dataset:
                    # Get predictions (not in @tf.function)
                    predictions = self.model(batch[0], training=False)
                    
                    # Log predictions in eager mode
                    self.log_predictions(predictions, batch[1], epoch, batch_count, log_file)
                    
                    # Distributed training step
                    loss = self.distributed_train_step(batch)
                    total_loss += loss
                    batch_count += 1
                
                avg_loss = total_loss / batch_count if batch_count > 0 else float('inf')
                epoch_logs['loss'] = avg_loss
                
                print(f"Epoch {epoch + 1}/{num_epochs} - Loss: {avg_loss:.4f}")
                
                if callbacks:
                    for callback in callbacks:
                        callback.on_epoch_end(epoch, epoch_logs)
            
            if callbacks:
                for callback in callbacks:
                    callback.on_train_end()
                    
        except Exception as e:
            print(f"Training interrupted: {str(e)}")
            if callbacks:
                for callback in callbacks:
                    callback.on_train_end()
            raise
        finally:
            log_file.close()

    def train_step(self, inputs):
        batch_inputs, batch_targets = inputs
        
        with tf.GradientTape() as tape:
            y_pred = self.model(batch_inputs, training=True)
            loss = self.loss_fn(batch_targets, y_pred)
        
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))
        
        return loss

    # Helper methods remain the same
    def decode_smiles_batch(self, tensor):
        pred_indices = tf.argmax(tensor, axis=-1)
        smiles_list = []
        for tokens in pred_indices:
            smiles = decode_smiles(tokens, Constants.TOKENIZER_PATH)
            smiles_list.append(smiles if smiles else "")
        return smiles_list
    
    def decode_smiles_input(self, tensor):
        if len(tensor.shape) > 2:
            tensor = tf.reshape(tensor, [tensor.shape[0], -1])
    
        smiles_list = []
        for tokens in tensor:
            smiles = decode_smiles(tokens, Constants.TOKENIZER_PATH)
            smiles_list.append(smiles if smiles else "")
        return smiles_list