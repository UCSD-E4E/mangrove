read -p 'Model Version: ' version
 
read -p 'Number of Epochs: ' epochs
 
read -p 'Learning Rate: ' learning_rate
 
read -p 'Batch Size: ' batch_size
 
python3 cnn_retrain2.py \
    --image_dir=/input \
    --output_graph=/output/output_graph_v$version.pb \
    --output_lables=/output/output_labels_v$version.pb \
    --how_many_training_steps=$epochs \
    --learning_rate=$learning_rate \
    --train_batch_size=$batch_size \
    --flip_left_right=10 \
    --random_crop=10 \
    --random_scale=10 \
    --random_brightness=10
