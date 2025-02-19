from CustomLoss import *
class PPOTrainer:
    def __init__(self, model, lambda_rl=0.1, clip_epsilon=0.2, lr=1e-4):
        self.model = model
        self.optimizer = tf.keras.optimizers.Adam(lr)
        self.loss_fn = CustomMoleculeLoss(lambda_rl, clip_epsilon)

    def train(self, dataset, num_epochs, batch_size, callbacks=None):
        dataset = dataset.shuffle(buffer_size=1000).batch(batch_size)
        log_file = open('training_predictions.log', 'w')
        # Initialize callbacks
        if callbacks:
            for callback in callbacks:
                # Set the underlying model for callbacks
                callback.set_model(self.model)  # Use the actual model instead of PPO trainer
                callback.on_train_begin()
                
        
        try:
            for epoch in range(num_epochs):
                if callbacks:
                    for callback in callbacks:
                        callback.on_epoch_begin(epoch)
            
                epoch_logs = {}
                total_loss = 0
                batch_count = 0
                
                for batch_inputs, batch_targets in dataset:
                    # Get predictions before training step
                    predictions = self.model(batch_inputs, training=False)

                    
                    # Decode predictions and inputs
                    pred_smiles = self.decode_smiles_batch(predictions)
                    input_smiles = self.decode_smiles_input(batch_targets)

                    
                    # Extract groups
                    pred_groups = [extract_group_smarts(s) for s in pred_smiles]
                    input_groups = [extract_group_smarts(s) for s in input_smiles]
                    
                    # Log predictions for this batch
                    log_file.write(f"\nEpoch {epoch + 1}, Batch {batch_count + 1}:\n")
                    for i in range(len(pred_smiles)):
                        log_file.write(f"Input SMILES: {input_smiles[i]}\n")
                        log_file.write(f"Predicted SMILES: {pred_smiles[i]}\n")
                        log_file.write(f"Input Groups: {input_groups[i]}\n")
                        log_file.write(f"Predicted Groups: {pred_groups[i]}\n")
                        log_file.write("-" * 50 + "\n")
                    log_file.flush()  # Ensure writing to file
                    
                    # Continue with training
                    loss = self.train_step(batch_inputs, batch_targets)
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

    def decode_smiles_batch(self, tensor):
        """Helper function to decode a batch of predictions to SMILES strings"""
        pred_indices = tf.argmax(tensor, axis=-1)
        smiles_list = []
        for tokens in pred_indices:
            smiles = decode_smiles(tokens, Constants.TOKENIZER_PATH)
            smiles_list.append(smiles if smiles else "")
        return smiles_list
    
    def decode_smiles_input(self, tensor):
        """Helper function to decode input tensor to SMILES strings"""
        # Reshape tensor to combine sequences if needed
        if len(tensor.shape) > 2:
            tensor = tf.reshape(tensor, [tensor.shape[0], -1])
    
        smiles_list = []
        for tokens in tensor:
            smiles = decode_smiles(tokens, Constants.TOKENIZER_PATH)
            smiles_list.append(smiles if smiles else "")
        return smiles_list

    def train_step(self, batch_inputs, batch_targets):
        with tf.GradientTape() as tape:
            # Generate predictions (monomers)
            y_pred = self.model(batch_inputs, training=True)

            # Compute PPO loss
            loss = self.loss_fn(batch_targets, y_pred)

        # Compute gradients and apply updates
        grads = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.model.trainable_variables))

        return loss

