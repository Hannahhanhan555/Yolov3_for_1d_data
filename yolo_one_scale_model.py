import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
os.environ['CUDA_VISIBLE_DEVICES'] = '1'


class Conv1D(tf.keras.layers.Layer):
    def __init__(self, filters, kernel_size, name="Conv1D", **kwargs):
        super(Conv1D, self).__init__(name=name)
        self.kwargs = {}
        self.kwargs.update(kwargs)
        self.conv = tf.keras.layers.Conv1D(filters, kernel_size, padding="same", strides=1, **kwargs)

    def call(self, inputs):
        return self.conv(inputs)


class Conv1D_BN_Leaky(tf.keras.Model):
    def __init__(self, filters, kernel_size, name="Conv1D_BN_Leaky", **kwargs):
        super(Conv1D_BN_Leaky, self).__init__(name=name)
        self.kwargs = {}
        self.kwargs.update(kwargs)
        self.Conv1D = Conv1D(filters, kernel_size, **self.kwargs)
        self.LeakyReLU = tf.keras.layers.LeakyReLU()
        self.BN = tf.keras.layers.BatchNormalization()

    def call(self, input_tensor, training=True):
        x = self.Conv1D(input_tensor)
        x = self.BN(x)
        x = self.LeakyReLU(x)
        return x


class Res_unit(tf.keras.Model):
    def __init__(self, filters1, kernel_size1, filters2, kernel_size2, name="Res_unit"):
        super(Res_unit, self).__init__(name=name)
        self.conv1 = Conv1D_BN_Leaky(filters1, kernel_size1, padding="same", bias_initializer="LecunNormal")
        self.conv2 = Conv1D_BN_Leaky(filters2, kernel_size2, padding="same", bias_initializer="LecunNormal")
        self.add = tf.keras.layers.Add()

    def call(self, input_tensor, training=True):
        x = self.conv1(input_tensor)
        x = self.conv2(x)
        return self.add([input_tensor, x])


class Resblock_body(tf.keras.Model):
    def __init__(self, unit_num, name="Resblock_body", **kwargs):
        super(Resblock_body, self).__init__(name=name)
        self.unit_num = unit_num

        self.DBL_args = kwargs["DBL"]
        self.res_unit_arg = {}
        self.res_unit_arg.update(kwargs["res_unit1"])
        self.res_unit_arg.update(kwargs["res_unit2"])
        self.zero_padding = tf.keras.layers.ZeroPadding1D()
        self.DBL = Conv1D_BN_Leaky(**self.DBL_args)
        self.res_unit = Res_unit(**self.res_unit_arg)

    def call(self, input_tensor, training=True):
        x = input_tensor
        x = self.DBL(x)
        for i in range(self.unit_num):
            x = self.res_unit(x)
        return x


class Darknet53(tf.keras.Model):
    def __init__(self, name="Darknet53"):
        super(Darknet53, self).__init__(name=name)
        self.DBL = Conv1D_BN_Leaky(32, 3, strides=1)  
        # conv1D input_shape[batch_size,steps,input_dim] output_shape[batch_size,new_steps,input_dim]
        self.RB1_args = {}
        self.RB1_args["DBL"] = {"filters": 64, "kernel_size": 3, "strides": 2}
        self.RB1_args["res_unit1"] = {"filters1": 32, "kernel_size1": 1}
        self.RB1_args["res_unit2"] = {"filters2": 64, "kernel_size2": 3}
        self.RB1 = Resblock_body(unit_num=1, **self.RB1_args)

        self.RB2_args = {}
        self.RB2_args["DBL"] = {"filters": 128, "kernel_size": 3, "strides": 2}
        self.RB2_args["res_unit1"] = {"filters1": 64, "kernel_size1": 1}
        self.RB2_args["res_unit2"] = {"filters2": 128, "kernel_size2": 3}
        self.RB2 = Resblock_body(unit_num=2, **self.RB2_args)

        self.RB8_args = {}
        self.RB8_args["DBL"] = {"filters": 256, "kernel_size": 3, "strides": 2}
        self.RB8_args["res_unit1"] = {"filters1": 128, "kernel_size1": 1}
        self.RB8_args["res_unit2"] = {"filters2": 256, "kernel_size2": 3}
        self.RB8 = Resblock_body(unit_num=8, **self.RB8_args)

        self.RB8_2_args = {}
        self.RB8_2_args["DBL"] = {"filters": 512, "kernel_size": 3, "strides": 2}
        self.RB8_2_args["res_unit1"] = {"filters1": 256, "kernel_size1": 1}
        self.RB8_2_args["res_unit2"] = {"filters2": 512, "kernel_size2": 3}
        self.RB8_2 = Resblock_body(unit_num=8, **self.RB8_2_args)

        self.RB4_args = {}
        self.RB4_args["DBL"] = {"filters": 1024, "kernel_size": 3, "strides": 2}
        self.RB4_args["res_unit1"] = {"filters1": 512, "kernel_size1": 1}
        self.RB4_args["res_unit2"] = {"filters2": 1024, "kernel_size2": 3}
        self.RB4 = Resblock_body(unit_num=4, **self.RB4_args)

    def call(self, input_tensor):
        x = input_tensor
        x = self.DBL(x)
        x = self.RB1(x)
        x = self.RB2(x)
        x = self.RB8(x)
        out3 = x
        x = self.RB8_2(x)
        out2 = x
        x = self.RB4(x)
        out1 = x
        return out1, out2, out3


class DBL_5L(tf.keras.Model):
    def __init__(self, name="DBL_5L", **kwargs):
        super(DBL_5L, self).__init__(name=name)
        self.DBL1 = Conv1D_BN_Leaky(filters=kwargs["filters1"], kernel_size=1)
        self.DBL2 = Conv1D_BN_Leaky(filters=kwargs["filters2"], kernel_size=3)
        self.DBL3 = Conv1D_BN_Leaky(filters=kwargs["filters3"], kernel_size=1)
        self.DBL4 = Conv1D_BN_Leaky(filters=kwargs["filters4"], kernel_size=3)
        self.DBL5 = Conv1D_BN_Leaky(filters=kwargs["filters5"], kernel_size=1)

    def call(self, input_tensor):
        x = input_tensor
        x = self.DBL1(x)
        x = self.DBL2(x)
        x = self.DBL3(x)
        x = self.DBL4(x)
        x = self.DBL5(x)
        return x


class yolov3(tf.keras.Model):
    def __init__(self, train, name="yolov3"):
        super(yolov3, self).__init__(name=name)
        self.darknet53 = Darknet53()
        self.DBL_5L_y1 = DBL_5L(filters1=512, filters2=1024, filters3=512, filters4=1024, filters5=512)
        self.DBL_y1 = Conv1D_BN_Leaky(filters=1024, kernel_size=3)
        self.conv_y1 = tf.keras.layers.Conv1D(filters=15, kernel_size=1)  ##y1 output

        self.DBL_y2_1 = Conv1D_BN_Leaky(filters=256, kernel_size=1)
        self.up_sample_y2 = tf.keras.layers.UpSampling1D()
        self.DBL_5L_y2 = DBL_5L(filters1=256, filters2=512, filters3=256, filters4=512, filters5=256)
        self.DBL_y2_2 = Conv1D_BN_Leaky(filters=512, kernel_size=3)
        self.conv_y2 = tf.keras.layers.Conv1D(filters=15, kernel_size=1)  ##y2 output

        self.DBL_y3_1 = Conv1D_BN_Leaky(filters=128, kernel_size=1)
        self.up_sample_y3 = tf.keras.layers.UpSampling1D()
        self.DBL_5L_y3 = DBL_5L(filters1=128, filters2=256, filters3=128, filters4=256, filters5=128)
        self.DBL_y3_2 = Conv1D_BN_Leaky(filters=256, kernel_size=3)
        self.conv_y3 = tf.keras.layers.Conv1D(filters=15, kernel_size=1)  ##y3 output
        self.anchor_num = 3
        self.class_num = 2
        self.train = train
    def call(self, input_tensor):
        # feature map 1   output_size  8*18
        out1, out2, out3 = self.darknet53(input_tensor)
        mid1 = self.DBL_5L_y1(out1)
        # y1 = self.DBL_y1(mid1)
        # y1 = self.conv_y1(y1)

        # feature map 2  output_size 16*18
        mid2 = self.DBL_y2_1(mid1)
        mid2 = self.up_sample_y2(mid2)
        mid2 = tf.keras.layers.concatenate([out2, mid2])
        mid2 = self.DBL_5L_y2(mid2)
        # y2 = self.DBL_y2_2(mid2)
        # y2 = self.conv_y2(y2)  # y2 output

        # feature map 3  output_size 32*18, just use this scale, so y1 and y2 not output. 
        mid3 = self.DBL_y3_1(mid2)
        mid3 = self.up_sample_y3(mid3)
        mid3 = tf.keras.layers.concatenate([out3, mid3])
        mid3 = self.DBL_5L_y3(mid3)
        y3 = self.DBL_y3_2(mid3)
        y3 = self.conv_y3(y3)
        return [y3]


class Yolo_v3_loss(tf.keras.losses.Loss):
    def __init__(self, class_num, batch_size):
        super(Yolo_v3_loss, self).__init__()
        self.class_num = class_num
        self.batch_size = batch_size
        # change self.anchors according your data. 
        self.anchors = tf.constant([[159.30358463, 2212.28297724, 4239.47105541]], dtype=tf.float32)
        self.scales = [8]
        self.iou_threshold = 0.5

        self.lambda_obj = 5
        self.lambda_noobj = 1
        self.lambda_class = 5
        self.lambda_xl = 1

        self.loss_mse = tf.keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE)


    def call(self, y_true, y_pred):
        # create mask for each scale
        scale_idx = 0
        y_true_i = y_true[scale_idx]    # y_true_i & y_pred_i'shape [16, 5000, 3, 5]
        y_pred_i = y_pred[scale_idx]

        batch_mask_obj, batch_mask_noobj, batch_mask_best_anchor = \
            self.create_mask_one_scale_for_batch(y_true_i, y_pred_i, scale_idx)

        loss_conf = self.loss_mse(tf.expand_dims(y_true_i[:, :, :, 2], axis=-1),
                                  tf.expand_dims(y_pred_i[:, :, :, 2], axis=-1))
        loss_conf_obj = self.lambda_obj * tf.reduce_sum(loss_conf * batch_mask_obj) / tf.reduce_sum(
            batch_mask_obj) / 3

        loss_conf_noobj = self.lambda_noobj * tf.reduce_sum(loss_conf * batch_mask_noobj) / tf.reduce_sum(
            batch_mask_noobj) / 3

        # loss_class = self.loss_ce(y_true_i[:, :, :, 3:], y_pred_i[:, :, :, 3:])
        loss_class = self.loss_mse(y_true_i[:,:,:, 3:], tf.nn.softmax(y_pred_i[:,:,:,3:]))
        loss_class *= batch_mask_best_anchor
        loss_class = self.lambda_class * tf.reduce_sum(loss_class) / tf.reduce_sum(batch_mask_best_anchor)

        loss_xl = (2 - y_true_i[:, :, :, 1]) * (
                self.loss_mse(tf.expand_dims(y_true_i[:, :, :, 0], axis=-1),
                              tf.expand_dims(y_pred_i[:, :, :, 0], axis=-1)) +
                self.loss_mse(tf.expand_dims(y_true_i[:, :, :, 1], axis=-1),
                              tf.expand_dims(y_pred_i[:, :, :, 1], axis=-1)))
        loss_xl *= batch_mask_best_anchor
        loss_xl = self.lambda_xl * tf.reduce_sum(loss_xl) / tf.reduce_sum(batch_mask_best_anchor)

        print("loss_conf_obj: %s, loss_conf_noobj: %s, loss_class: %s, loss_xl: %s" %
              (loss_conf_obj.numpy(), loss_conf_noobj.numpy(), loss_class.numpy(), loss_xl.numpy()))
        loss_all = loss_xl + loss_class + loss_conf_noobj + loss_conf_obj
        return loss_all

       

    def calculate_iou_one_scale_for_batch(self, true, pred, scale_i):
        """
        :param true: [bs, output_size, anchor_nums, class_num + 2]
        :param pred:  same as true
        :return:
        """

        true_shape = tf.shape(true)
        bs = true_shape[0]
        output_size = true_shape[1]
        anchor_num = true_shape[2]

        anchors = tf.tile([self.anchors], [bs, output_size, 1])
        # 转换到原始尺寸上计算iou
        grid_idx = tf.cast(tf.tile([tf.range(output_size)], [bs, anchor_num]), tf.float32)
        grid_idx = tf.transpose(tf.reshape(grid_idx, [tf.shape(grid_idx)[0], anchor_num, -1]), [0, 2, 1])

        true_x = (true[:, :, :, 0] + grid_idx) * self.scales[scale_i]
        pred_x = (pred[:, :, :, 0] + grid_idx) * self.scales[scale_i]
        true_l = tf.exp(true[:, :, :, 1]) * anchors
        pred_l = tf.exp(pred[:, :, :, 1]) * anchors

        true_end_point = true_x + true_l / 2
        true_start_point = true_x - true_l / 2
        pred_end_point = pred_x + pred_l / 2
        pred_start_point = pred_x - pred_l / 2

        u = tf.where(true_end_point > pred_end_point, true_end_point, pred_end_point) - tf.where(
            true_start_point < pred_start_point, true_start_point, pred_start_point)
        u = tf.where(u > pred_l + true_l, pred_l + true_l, u)
        i = tf.where(true_end_point < pred_end_point, true_end_point, pred_end_point) - tf.where(
            true_start_point > pred_start_point, true_start_point, pred_start_point)
        # tf.print("i: ", i)
        i = tf.where(i < 0.0, 0.0, i)
        iou = tf.divide(i, u)
        return iou

    def create_mask_one_scale_for_batch(self, y_true_i, y_pred_i, scale_i):
        """
        :param y_true_i:  shape as [bs, output_size, anchor_num, class_num + 3]
        :param y_pred_i:
        :param scale_i:  here always 0
        :return:
        """
        true_shape = tf.shape(y_true_i)
        bs = true_shape[0]
        output_size = true_shape[1]
        anchor_num = true_shape[2]

        grid_idx = tf.cast(tf.tile([tf.range(output_size)], [bs, anchor_num]), tf.int64)
        grid_idx = tf.transpose(tf.reshape(grid_idx, [tf.shape(grid_idx)[0], anchor_num, -1]), [0, 2, 1])
        # gird_idx shape as [bs, output_size, anchor_num]
        bs_idx = tf.cast(tf.transpose(tf.tile([tf.range(bs)], [output_size, 1])), tf.int64) # shape as [bs, output_size]

        # first find best anchors
        y_true_have_obj_idx = tf.where(y_true_i[:, :, 0, 2] == 1)
        iou = self.calculate_iou_one_scale_for_batch(y_true_i, y_pred_i, scale_i)  # iou shape [bs, output_shape, 3]
        best_anchors = tf.argmax(iou, axis=-1)
        best_anchor_idx = tf.concat([tf.expand_dims(bs_idx, axis=-1), tf.expand_dims(grid_idx[:, :, 0], axis=-1),
                                     tf.expand_dims(best_anchors, axis=-1)], axis=-1)

        # have obj and have max iou
        best_anchor_idx = tf.gather_nd(best_anchor_idx, y_true_have_obj_idx)
        best_anchor_idx = tf.cast(best_anchor_idx, tf.int64)  # [n, 3] 3 - batch_idx, grid_idx, anchor_idx
        mask_best_anchor = tf.SparseTensor(indices=best_anchor_idx, values=tf.tile([1.], [tf.shape(best_anchor_idx)[0]]),
                                           dense_shape=tf.cast(tf.shape(y_true_i)[0: -1], tf.int64)) # [bs, output_size, 3]
        mask_best_anchor = tf.sparse.to_dense(mask_best_anchor)
        # mask_conf_obj:  have_obj best anchor and iou lower than threshold
        iou_lower_than_threshold = tf.where(iou < self.iou_threshold, 1., 0.)
        mask_conf_obj = mask_best_anchor + iou_lower_than_threshold

        # mask_conf_noobj
        mask_conf_noobj = tf.where(y_true_i[:, :, :, 2] != 1, 1., 0.)

        return mask_conf_obj, mask_conf_noobj, mask_best_anchor




    def calculate_iou(self, true, pred, scale_index, anchor_index, grid_idx):
        """
        :param true:  for one sample
        :param pred:
        :param scale_index:
        :param anchor_index:
        :param grid_idx:
        :return:
        """
        # 转换到原始尺寸上计算iou
        grid_idx = tf.cast(grid_idx, tf.float32)
        true_x = (true[0] + grid_idx) * self.scales[scale_index]
        pred_x = (pred[0] + grid_idx) * self.scales[scale_index]
        true_l = tf.exp(true[1]) * self.anchors[scale_index][anchor_index]
        pred_l = tf.exp(pred[1]) * self.anchors[scale_index][anchor_index]

        true_end_point = true_x + true_l / 2
        true_start_point = true_x - true_l / 2
        pred_end_point = pred_x + pred_l / 2
        pred_start_point = pred_x - pred_l / 2

        u = tf.where(true_end_point > pred_end_point, true_end_point, pred_end_point) - tf.where(
            true_start_point < pred_start_point, true_start_point, pred_start_point)
        u = tf.where(u > pred_l + true_l, pred_l + true_l, u)
        i = tf.where(true_end_point < pred_end_point, true_end_point, pred_end_point) - tf.where(
            true_start_point > pred_start_point, true_start_point, pred_start_point)
        # tf.print("i: ", i)
        i = tf.where(i < 0.0, 0.0, i)
        iou = tf.divide(i, u)
        return iou

    def create_mask_one_scale_one_sample(self, y_true_i, y_pred_i, scale_i):
        """
        :param y_true_i: one scale one sample shaped as  (output_size, 3, class_num + 3)  after label assignment
        :param y_pred_i: same as y_true_i after decode_for_train
        :param scale_i: scale_index
        :return:
        """
        mask_conf_obj = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
        mask_conf_noobj = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
        
        mask_best_anchor = tf.TensorArray(size=0, dtype=tf.float32, dynamic_size=True)
        have_obj = tf.cast(tf.where(y_true_i[:, 0, 2] == 1), tf.int32)  # 对于一个scale，有object时，每个anchor都conf都是1所以任取一个就好
        output_size = 1500
        for grid_idx in tf.range(output_size):
            if tf.shape(tf.where(have_obj == grid_idx))[0] == 0:
                # have no obj
                for anchor_idx in tf.range(3):
                    mask_conf_obj = mask_conf_obj.write(grid_idx * 3 + anchor_idx, tf.constant([0.]))
                    mask_conf_noobj = mask_conf_noobj.write(grid_idx * 3 + anchor_idx, tf.constant([1.]))
                    mask_best_anchor = mask_best_anchor.write(grid_idx * 3 + anchor_idx, tf.constant([0.]))
            else:
                # have object
                # max iou
                ious = tf.TensorArray(dtype=tf.float32, size=0, dynamic_size=True)
                for anchor_idx in tf.range(3):
                # for anchor_idx in range(3):
                    anchor_true = y_true_i[grid_idx, anchor_idx]
                    anchor_pred = y_pred_i[grid_idx, anchor_idx]
                    ious = ious.write(anchor_idx, self.calculate_iou(anchor_true, anchor_pred, scale_i,
                                                                     anchor_idx, grid_idx))
                ious = ious.stack()
                iou_max_idx = tf.cast(tf.argmax(ious), tf.int32)
                for anchor_idx in tf.range(3):
                    if iou_max_idx == anchor_idx:
                        mask_conf_obj = mask_conf_obj.write(grid_idx * 3 + anchor_idx, tf.constant([1.]))
                        mask_conf_noobj = mask_conf_noobj.write(grid_idx * 3 + anchor_idx, tf.constant([0.]))
                        mask_best_anchor = mask_best_anchor.write(grid_idx * 3 + anchor_idx, tf.constant([1.]))

                    else:
                        # grid中包含obj，anchor iou非最高但高于阈值
                        if ious[anchor_idx] > self.iou_threshold:
                            mask_conf_obj = mask_conf_obj.write(grid_idx * 3 + anchor_idx, tf.constant([0.]))
                            mask_conf_noobj = mask_conf_noobj.write(grid_idx * 3 + anchor_idx, tf.constant([0.]))
                            mask_best_anchor = mask_best_anchor.write(grid_idx * 3 + anchor_idx, tf.constant([0.]))
                        # grid中包含obj，anchor iou < 阈值
                        else:
                            mask_conf_obj = mask_conf_obj.write(grid_idx * 3 + anchor_idx, tf.constant([1.]))
                            mask_conf_noobj = mask_conf_noobj.write(grid_idx * 3 + anchor_idx, tf.constant([0.]))
                            mask_best_anchor = mask_best_anchor.write(grid_idx * 3 + anchor_idx, tf.constant([0.]))

        mask_conf_obj = tf.reshape(mask_conf_obj.stack(), (-1, 3, 1))  # from [n * 3, 5] -> [n, 3, 5]
        mask_conf_noobj = tf.reshape(mask_conf_noobj.stack(), (-1, 3, 1))
        mask_best_anchor = tf.reshape(mask_best_anchor.stack(), (-1, 3, 1))

        # return mask_conf_obj, mask_conf_noobj, mask_class, mask_x_l
        return mask_conf_obj, mask_conf_noobj, mask_best_anchor

if __name__ == '__main__':
    # 为了保证
    input_tensor = tf.zeros([2, 12000, 1])
    # print("input shape:", input_tensor.shape)
    DBL = yolov3(train=True) # 用于训练 decode结果不同
    y0, y1, y2 = DBL(input_tensor)

    label0 = np.zeros((2, 375, 3, 5))
    label1 = np.zeros((2, 750, 3, 5))
    label2 = np.zeros((2, 1500, 3, 5))

    label0[0, 10, :, 2] = 1
    label0[1, 55, :, 2] = 1
    label1[0, 555, :, 2] = 1
    label1[1, 256, :, 2] = 1
    label2[0, 1111, :, 2] = 1
    label2[1, 512, :, 2] = 1
    label0 = tf.constant(label0, dtype=tf.float32)
    label1 = tf.constant(label1, dtype=tf.float32)
    label2 = tf.constant(label2, dtype=tf.float32)

    opt = tf.keras.optimizers.Adam()
    loss = Yolo_v3_loss(class_num=2, batch_size=2)