# Yolov3_for_1d_data
A Yolo model for 1d detection in TensorFlow. This repo just use one of the scale, so all process just for one scale. (model, gt assignment, loss...). When batch size was set to 16, training  one step just need 0.2s. 
1. Creating your own detection label by calling train.py ground_truth_assignment(). Notice that set the anchor according to your data.  

2. Creating your own data generator by rewriting this functions. 
data_generator(df_gen, batch_size, class_num, data_dir, label_dir)
test_generator(df_gen, class_num, data_dir, label_dir, batch_size)
boxes's length according to your data.
load_data_gt(df_batch, class_num, data_dir, label_dir)
load_data_raw(df_batch, class_num, data_dir, label_dir)
...

3. Just try it. 
