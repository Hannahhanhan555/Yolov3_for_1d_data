import yolo_one_scale_model as config
import pandas as pd
import numpy as np
import scipy.signal as signal
import random
import tensorflow as tf
import soundfile as sf
import matplotlib.pyplot as plt
import time


def data_generator(df_gen, batch_size, class_num, data_dir, label_dir):
    # 得到当前batch对应的df
    batch_num = int(np.ceil(df_gen.shape[0] / batch_size))
    idx = [i for i in range(df_gen.shape[0])]
    while 1:
        random.shuffle(idx)
        df_gen = df_gen.iloc[idx]

        for i in range(batch_num):
            s = i * batch_size
            e = (i + 1) * batch_size
            if e > df_gen.shape[0]:
                e = df_gen.shape[0]
                s = e - batch_size
            df_batch = df_gen.iloc[s:e]

            # 读取数据并返回
            data, label = load_data_raw(df_batch, class_num, data_dir, label_dir)
            # print("label.shape: ", label.shape)

            yield data, label


def test_generator(df_gen, class_num, data_dir, label_dir, batch_size = 1):
    # 得到当前batch对应的df
    batch_num = int(np.ceil(df_gen.shape[0] / batch_size))
    idx = [i for i in range(df_gen.shape[0])]
    while 1:
        random.shuffle(idx)
        df_gen = df_gen.iloc[idx]

        for i in range(batch_num):
            s = i * batch_size
            e = (i + 1) * batch_size
            if e > df_gen.shape[0]:
                e = df_gen.shape[0]
                s = e - batch_size
            df_batch = df_gen.iloc[s:e]

            # 读取数据并返回
            data, label = load_data_gt(df_batch, class_num, data_dir, label_dir)
            record_id = df_batch["path"].values[0].split("\\")[-1].split(".")[0]
            # print("label.shape: ", label.shape)

            yield data, label, record_id


def get_rawdata_se(abnormal_s_absolute, abnormal_e_absolute, cycle_start, n_fft=1024, hop_len=64, label_len=196, fs=4000):
    """
    params:
    abnormal_s, abnormal_e 是异常片段在record中的绝对时间位置
    """
    s_list = []
    e_list = []
    for i in range(len(abnormal_s_absolute)):
        ab_s = (abnormal_s_absolute[i] - cycle_start[i]) * fs
        ab_e = (abnormal_e_absolute[i] - cycle_start[i]) * fs
        s_list.append(ab_s)
        e_list.append(ab_e)
    se = np.array([s_list, e_list])
    se = np.transpose(se)
    return se


def get_label(ground_truth_per_scale, scale, class_num, anchor_box, input_shape):
    """

    :param ground_truth_per_scale: [n, 3]  3 - [x, l, class]
    :param scale:
    :param class_num:
    :param anchor_box:
    :param input_shape:
    :return: label.shape = (output_shape, 3, class_num + 3)
    """
    output_shape = int(np.ceil(input_shape / scale))
    label = np.zeros((output_shape, 3, class_num + 3))

    for (i, ground_truth_per_object) in enumerate(ground_truth_per_scale):

        location = np.floor(ground_truth_per_object[0] / scale)
        tx = (ground_truth_per_object[0] % scale) / scale
        class_onehot = tf.one_hot(ground_truth_per_object[2], depth=class_num)

        for (j, anchor_len) in enumerate(anchor_box):
            tl = np.log(ground_truth_per_object[1] / anchor_len)  # 自然对数
            # 由于anchor_len 没有除scale长度 因此这里不需要再 *scale
            label[int(location), j, :] = np.hstack([tx, tl, 1, class_onehot])

    return label


# 生成dataset前将ground truth转换为需要的标签
def groundtruth_assignment(ground_truth_raw, class_num, input_len):
    """
    just for one sample
    :param
    ground_truth: (n,4) array or tensor, 0,1 - abnormal_start point and end point, 2 - class， 3 - cycle_start
    class_num: class_num
    input_len: 196 / 12000
    :return:
    label_0, label_1,label_2 : 不同尺度上的标签 (output_size, 3, 6)    6 - [tx, tl, conf, one_hot]
    """
    anchor_boxes = tf.constant([[159.30358463, 2212.28297724, 4239.47105541]], dtype=tf.float32)
    scales = [8]

    # 196尺度上
    # se = get_frame_se(ground_truth_raw[:, 0], ground_truth_raw[:, 1], ground_truth_raw[:, 3])
    se = get_rawdata_se(ground_truth_raw[:, 0], ground_truth_raw[:, 1], ground_truth_raw[:, 3])
    ground_truth_raw[:, 0:2] = se
    center = (ground_truth_raw[:, 1] + ground_truth_raw[:, 0]) / 2
    length = ground_truth_raw[:, 1] - ground_truth_raw[:, 0]
    ground_truth = np.transpose(np.vstack([center, length, ground_truth_raw[:, 2]]))

    label = get_label(ground_truth, scales[0], class_num, anchor_boxes[0], input_len)

    return label


def load_data_raw(df_batch, class_num, data_dir, label_dir):
    bs = df_batch.shape[0]
    data = np.zeros((bs, 12000))
    label = np.zeros((bs, 1500, 3, 5))

    fs = 4000
    [b, a]= signal.butter(4, [60 / (fs / 2), (2000 - 0.001) / (fs / 2)], "bandpass")
    for index in range(bs):
        df = df_batch.iloc[index]
        if "steth" in df["path"]:
            record_id = df["path"].split("\\")[-1][0:23]
        else:
            record_id = df["path"].split("\\")[-1][0:22]

        record, fs_raw = sf.read(data_dir + record_id + ".wav")
        s = int(df["cycle_start"] * fs_raw)
        e = int(df["cycle_end"] * fs_raw)
        record_cycle = record[s:e]
        if fs_raw != 4000:
            sig_re = signal.resample(record_cycle, int(record_cycle.shape[0] / fs_raw * 4000))
        else:
            sig_re = record_cycle

        sig_filted = signal.filtfilt(b, a, sig_re)

        if sig_filted.shape[0] < 12000:
            sig_filted = np.pad(sig_filted, (0, 12000 - len(sig_filted)))
        else:
            sig_filted = sig_filted[0:12000]

        data[index] = sig_filted

        label_gt = np.load(df["path"].replace(".npy", "_label_yolo.npy"))
        label_gt[:, 2] -= 1
        label[index] = groundtruth_assignment(label_gt, class_num, 12000)
    data = np.expand_dims(data, axis=-1)
    data = tf.constant(data, tf.float32)
    label = tf.constant(label, tf.float32)
    return data, [label]


def load_data_gt(df_batch, class_num, data_dir, label_dir):
    bs = df_batch.shape[0]
    data = np.zeros((bs, 12000))

    fs = 4000
    [b, a] = signal.butter(4, [60 / (fs / 2), (2000 - 0.001) / (fs / 2)], "bandpass")
    for index in range(bs):
        df = df_batch.iloc[index]
        if "steth" in df["path"]:
            record_id = df["path"].split("\\")[-1][0:23]
        else:
            record_id = df["path"].split("\\")[-1][0:22]

        record, fs_raw = sf.read(data_dir + record_id + ".wav")
        s = int(df["cycle_start"] * fs_raw)
        e = int(df["cycle_end"] * fs_raw)
        record_cycle = record[s:e]
        if fs_raw != 4000:
            sig_re = signal.resample(record_cycle, int(record_cycle.shape[0] / fs_raw * 4000))
        else:
            sig_re = record_cycle
        sig_filted = signal.filtfilt(b, a, sig_re)
        if sig_filted.shape[0] < 12000:
            sig_filted = np.pad(sig_filted, (0, 12000 - len(sig_filted)))
        else:
            sig_filted = sig_filted[0:12000]
        data[index] = sig_filted

        label = np.load(df["path"].replace(".npy", "_label_yolo.npy"))
        label[:, 2] -= 1

    data = np.expand_dims(data, axis=-1)
    data = tf.constant(data, tf.float32)
    label = tf.constant(label, tf.float32)
    return data, label


def iou_from_ground_truth(true, pred):
    true_x = true[:, 0]
    pred_x = pred[:, 0]
    true_l = true[:, 1]
    pred_l = pred[:, 1]
    true_end_point = true_x + true_l / 2
    true_start_point = true_x - true_l / 2
    pred_end_point = pred_x + pred_l / 2
    pred_start_point = pred_x - pred_l / 2
    u = tf.where(true_end_point > pred_end_point, true_end_point, pred_end_point) - tf.where(
        true_start_point < pred_start_point, true_start_point, pred_start_point)
    u = tf.where(u > pred_l + true_l, pred_l + true_l, u)
    i = tf.where(true_end_point < pred_end_point, true_end_point, pred_end_point) - tf.where(
        true_start_point > pred_start_point, true_start_point, pred_start_point)
    i = tf.where(i < 0.0, 0.0, i)
    iou = tf.divide(i, u)
    return iou


def nms(y_pred, conf_threshold = 0.6, iou_threshold = 0.1, prob_threshold=0.7):
    """
    for one sample
    y_pred: [n, 5], [pred_x, pred_length, pred_conf, pred_class, pred_class_prob]
    """
    y_pred = decode_for_inference(y_pred)[0]
    # input shape: [[1, grid1, 15]
    # decode output shape: [1, n0 + n1 + n2, 3, 5]
    # 取0 -> output shape: [grid1 + grid2 + grid3, 3, 5]
    y_pred_sample = tf.reshape(y_pred, [-1, 5])  # [ni * 3, 5] 对应实际尺度上的数据

    # length over 12000, delete
    y_pred_sample_idx = tf.where(y_pred_sample[:, 1] < 12000)
    y_pred_sample = tf.gather(y_pred_sample, tf.squeeze(y_pred_sample_idx))

    # s < 0 or e > 12000 delete
    y_pred_s = y_pred_sample[:, 0] - y_pred_sample[:, 1] / 2
    y_pred_sample_idx = tf.where(y_pred_s >= 0)
    y_pred_sample = tf.gather(y_pred_sample, tf.squeeze(y_pred_sample_idx))

    y_pred_e = y_pred_sample[:, 0] + y_pred_sample[:, 1] / 2
    y_pred_sample_idx = tf.where(y_pred_e <= 12000)
    y_pred_sample = tf.gather(y_pred_sample, tf.squeeze(y_pred_sample_idx))

    # conf threshold
    conf = y_pred_sample[:, 2]
    conf_index = tf.squeeze(tf.where(conf > conf_threshold))
    y_pred_sample = tf.gather(y_pred_sample, conf_index)

    # prob threshold
    prob = y_pred_sample[:, 4]
    prob_index = tf.squeeze(tf.where(prob > prob_threshold))
    y_pred_sample = tf.gather(y_pred_sample, prob_index)

    # conf sort
    conf = y_pred_sample[:, 2]
    max_index = tf.nn.top_k(conf, conf.shape[0])[1]
    y_pred_sample = tf.gather(y_pred_sample, max_index)


    y_save = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)

    # conf排序后，while循环
    index = 0
    while y_pred_sample.shape[0] != 0:
        # 选出conf最大的一个
        y_max = y_pred_sample[0]
        # print("y_max.shape: ", y_max.shape)
        y_save = y_save.write(index, y_max)
        index += 1
        y_pred_sample = y_pred_sample[1:]
        # print("y_pred_sample.shape(up): ", y_pred_sample.shape)
        # 计算iou，删除iou大于阈值的部分
        y_max_tile = tf.tile(tf.expand_dims(y_max, axis=0), [y_pred_sample.shape[0], 1])
        iou = iou_from_ground_truth(y_max_tile, y_pred_sample)  #  input (n, 5), output (n,)
        not_delete_idx = tf.where(iou < iou_threshold)    # (n , 1)
        if tf.shape(not_delete_idx)[0] > 1:
            not_delete_idx = tf.squeeze(not_delete_idx)

        y_pred_sample = tf.gather(y_pred_sample, not_delete_idx)
        if y_pred_sample.shape[0] == 1:
            y_save = y_save.write(index, y_pred_sample[0][0])
            break
    return y_save.stack()


def decode_for_train(y_pred):
    """
    :param :
    y_pred: type list, [y1, y2, y3] 先是尺寸最小的feature map的输出
    :return:
    the tranform output. 3 tensor,
    shape = [batch_size, feature_map_output_size, anchor_num, class_num+3] for each scale
    """
    anchor_num = 3
    class_num = 2

    pred = None
    index = 0
    yi = y_pred[index]
    out_shape = tf.shape(yi)
    batch_size = out_shape[0]
    output_size = out_shape[1]

    yi = tf.reshape(yi, (batch_size, output_size, anchor_num, 3 + class_num))
    # [batchsize, feature_mapsize, anchor_num, 3 + classes]

    t_x = tf.expand_dims(yi[:, :, :, 0], -1)  # 中心位置偏移
    t_length = tf.expand_dims(yi[:, :, :, 1], -1)  # 长度
    conf = tf.expand_dims(yi[:, :, :, 2], -1)  # 是否有物体置信度
    prob = yi[:, :, :, 3:]  # 类别
    pred_x = tf.sigmoid(t_x)
    pred_length = t_length
    pred_conf = tf.sigmoid(conf)
    pred_prob = prob
    temp = tf.concat([pred_x, pred_length, pred_conf, pred_prob], axis=-1)

    if index == 0:
        pred = temp
    else:
        pred = tf.concat([pred, temp], axis=1)

    return [pred]


def decode_for_inference(y_pred):
    """
    :param y_pred: for one sample
    :return:  [n,5]  [pred_x, pred_length, pred_conf, pred_class, pred_class_prob]
    """
    scale = tf.constant([8], dtype=tf.float32)
    anchors = tf.constant([[159.30358463, 2212.28297724, 4239.47105541]], dtype=tf.float32)
    anchor_num = 3
    class_num = 2

    index = 0
    yi = y_pred[index]
    out_shape = tf.shape(yi)
    batch_size = out_shape[0]
    output_size = out_shape[1]

    yi = tf.reshape(yi, (batch_size, output_size, anchor_num, 3 + class_num))
    # [batchsize, feature_mapsize, anchor_num, 3 + classes]

    t_x = tf.expand_dims(yi[:, :, :, 0], -1)  # 网格中心位置
    t_length = tf.expand_dims(yi[:, :, :, 1], -1)  # 长度 [bs, grid, anchor, 1]
    conf = tf.expand_dims(yi[:, :, :, 2], -1)  # 是否有物体置信度
    prob = yi[:, :, :, 3:]  # 类别
    # 生成网格 为得到真实的x坐标
    x = tf.expand_dims(tf.range(output_size, dtype=tf.int32), axis=-1)
    x_grid = tf.tile(x[tf.newaxis, :, tf.newaxis, :], [batch_size, 1, anchor_num, 1])
    x_grid = tf.cast(x_grid, dtype=tf.float32)  # [batch, feature_map_size, 3, 1]

    # 转换结果： x,length 对应原标签中的位置和长度
    pred_x = (tf.sigmoid(t_x) + x_grid) * scale[index]
    pred_length = tf.exp(tf.squeeze(t_length, axis=-1)) * anchors[index]
    pred_length = tf.expand_dims(pred_length, axis=-1)  # * scale[index] anchor里包含着scale信息
    pred_conf = tf.sigmoid(conf)
    pred_prob = tf.nn.softmax(prob)  # 输出结果时需要通过softmax
    pred_class = tf.expand_dims(tf.cast(tf.argmax(pred_prob, axis=-1), tf.float32), axis=-1)
    pred_prob0 = pred_prob[:,:,:, 0]
    pred_prob1 = pred_prob[:,:,:, 1]
    pred_class_prob = tf.expand_dims(tf.where(pred_prob0 > pred_prob1, pred_prob0, pred_prob1), axis=-1)

    pred = tf.concat([pred_x, pred_length, pred_conf, pred_class, pred_class_prob], axis=-1)
    return pred


def model_inference(model, input, label, save_dir, record_id = None):
    """
    :param model:
    :param input: [1, 12000, 1]
    :param label 未经ground truth assignment的label
    :return:
    每次处理一个sample
    """
    pred = model(input)
    # [y0, y1, y2]  [(bs = 1, grid1, 15), (bs = 1, grid2, 15) ,(bs = 1, grid3, 15)]
    # print("model_inference pred0.shape: ", pred[0].shape)
    pred_decode = nms(pred, conf_threshold=0.6, iou_threshold=0.01, prob_threshold=0.7)  # (n,5)
    for index in range(pred_decode.shape[0]):
        print(pred_decode[index])
    draw_image(input, pred_decode, label, record_id, save_dir)

    return pred_decode


def draw_image(record, pred_decode, label, record_id, save_dir):
    fig = plt.figure()
    #     plt.ylim((-1,1))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(record[0])

    s = (label[:, 0] - label[:, -1]) * 4000
    e = (label[:, 1] - label[:, -1]) * 4000
    length = e - s
    class_label = label[:, 2].numpy()

    s_pred = pred_decode[:, 0] - pred_decode[:, 1] / 2
    e_pred = pred_decode[:, 0] + pred_decode[:, 1] / 2
    length_pred = e_pred - s_pred
    pred_conf = pred_decode[:, 2].numpy()
    pred_label = pred_decode[:, 3].numpy()
    pred_prob = pred_decode[:, 4].numpy()
    # print(pred_prob)
    # print(pred_prob[0, int(pred_label[0])])

    for index in range(label.shape[0]):  #
        rect = plt.Rectangle((s[index], -1), length[index], 2, fill=False, edgecolor="blue", linewidth=3)
        ax.add_patch(rect)

    for index in range(pred_decode.shape[0]):  #
        rect = plt.Rectangle((s_pred[index], -1), length_pred[index], 2, fill=False, edgecolor="red")
        plt.text(s_pred[index], -0.5, str(pred_label[index])[0:4])
        plt.text(s_pred[index], -0.75, str(pred_prob[index])[0:4])
        plt.text(s_pred[index], -1, str(pred_conf[index])[0:4])
        ax.add_patch(rect)
    plt.title("label: %s, pred: %s" % (class_label, pred_label))
    plt.plot()
    plt.savefig(save_dir + record_id + ".png")
    plt.close()


# @tf.function
def train_step(x_batch_train, y_true):
    with tf.GradientTape() as tape:
        predictions = model(x_batch_train)
        predictions = decode_for_train(predictions)
        loss = loss_obj(y_true, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    # lr = optimizer.lr
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


if __name__ == "__main__":
    # first config for your train_generator. 
    train_gen = ""
    png_save_dir = ""
    batch_size = 16
    class_num = 2
    sample_num = 100
    epochs = 5
    steps = int(np.ceil(sample_num / batch_size))
    model = config.yolov3(train=True)
    model.build(input_shape=(None, 12000, 1))
    lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=0.0005, decay_steps=2 * steps,
                                                                 decay_rate=0.96)
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    loss_obj = config.Yolo_v3_loss(class_num=2, batch_size=batch_size)


    for epoch in range(epochs):
        step = 0
        loss_batch = 0
        lr = 0
        time_s = time.time()
        for data, label in train_gen:
            loss_value = train_step(data, label)
            loss_batch += loss_value
            step += 1
            if step == steps:
                break
        time_e = time.time()
        print("epoch %s: " % (epoch+1))
        print("loss: %s, time: %s" % (loss_batch.numpy(), time_e - time_s))


