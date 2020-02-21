import tensorflow as tf
from tensorflow import keras

def focal(alpha=0.25, gamma=2.0):
    """ Create a functor for computing the focal loss.
    Args
        alpha: Scale the focal weight with alpha.
        gamma: Take the power of the focal weight with gamma.
    Returns
        A functor that computes the focal loss using the alpha and gamma.
    """
    def _focal(y_true, y_pred):
        """ Compute the focal loss given the target tensor and the predicted tensor.
        As defined in https://arxiv.org/abs/1708.02002
        Args
            y_true: Tensor of target data from the generator with shape (B, N, num_classes).
            y_pred: Tensor of predicted data from the network with shape (B, N, num_classes).
        Returns
            The focal loss of y_pred w.r.t. y_true.
        """
        # labels         = y_true[:, :, :-1]
        labels         = y_true[:, :,4]
        anchor_state   = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred

        # filter out "ignore" anchors
        indices        = tf.where(keras.backend.not_equal(anchor_state, -1))
        labels         = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = tf.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        normalizer = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)

        return keras.backend.sum(cls_loss) / normalizer

    return _focal


def smooth_l1(sigma=3.0):
    """ Create a smooth L1 loss functor.
    Args
        sigma: This argument defines the point where the loss changes from L2 to L1.
    Returns
        A functor for computing the smooth L1 loss given target data and predicted data.
    """
    sigma_squared = sigma ** 2

    def _smooth_l1(y_true, y_pred):
        """ Compute the smooth L1 loss of y_pred w.r.t. y_true.
        Args
            y_true: Tensor from the generator of shape (B, N, 5). The last value for each box is the state of the anchor (ignore, negative, positive).
            y_pred: Tensor from the network of shape (B, N, 4).
        Returns
            The smooth L1 loss of y_pred w.r.t. y_true.
        """
        # separate target and state
        # regression        = y_pred
        # regression_target = y_true[:, :, :-1]
        # anchor_state      = y_true[:, :, -1]
        regression        = y_pred[:,:,:4]
        regression_target = y_true[:, :, :4]
        anchor_state      = y_true[:, :, -1]

        # filter out "ignore" anchors
        indices           = tf.where(keras.backend.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        return keras.backend.sum(regression_loss) / normalizer

    return _smooth_l1

def focal_plus_smooth(sigma=0.3,alpha=0.25, gamma=2.0):

    sigma_squared = sigma ** 2
    
    def _added_loss(y_true,y_pred):
        """
        y_true : (B,N,6) : first 4 values are anchors , 5th is label and 6th is positive index
        y_pred : (B,N,5) 
        """
        regression        = y_pred[:,:,:4]
        regression_target = y_true[:, :, :4]
        anchor_state      = y_true[:, :, -1]

        print("ytrue shape: ",y_true.shape)
        print("ypred shape: ",y_pred)

        # filter out "ignore" anchors
        indices           = tf.where(keras.backend.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        # normalizer = keras.backend.cast(normalizer, dtype='float64')
        smooth_loss = keras.backend.sum(regression_loss) / normalizer
        ###############################################################################
        ## computing focal loss now
        ###############################################################################
        labels         = y_true[:, :,4]
        anchor_state   = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred[:,:,4]

        # filter out "ignore" anchors
        indices        = tf.where(keras.backend.not_equal(anchor_state, -1))
        # indices        = tf.where(keras.backend.equal(anchor_state, 1))
        labels         = tf.gather_nd(labels, indices)
        classification = tf.gather_nd(classification, indices)

        # compute the focal loss
        alpha_factor = keras.backend.ones_like(labels) * alpha
        alpha_factor = tf.where(keras.backend.equal(labels, 1), alpha_factor, 1 - alpha_factor)
        focal_weight = tf.where(keras.backend.equal(labels, 1), 1 - classification, classification)
        focal_weight = alpha_factor * focal_weight ** gamma

        # cls_loss = focal_weight * keras.backend.binary_crossentropy(labels, classification)

        cls_loss = keras.backend.binary_crossentropy(labels, classification)

        # compute the normalizer: the number of positive anchors
        # normalizer = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer = tf.where(keras.backend.not_equal(anchor_state, -1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)
        # normalizer = keras.backend.cast(normalizer, dtype='float64')
        focal_loss = keras.backend.sum(cls_loss) / normalizer
        
        return focal_loss+(0.25*smooth_loss)
        # return smooth_loss

    return _added_loss

def ohem_plus_smooth_plus_fovial(sigma=0.3,alpha=0.25, gamma=2.0):

    sigma_squared = sigma ** 2
    
    def _added_loss(y_true,y_pred):
        """
        y_true : (B,N,6) : first 4 values are anchors , 5th is label and 6th is positive index
        y_pred : (B,N,5) 
        """
        ###############################################################################
        ## computing regression targets loss now
        ###############################################################################
        regression        = y_pred[:,:,:4]
        regression_target = y_true[:, :, :4]
        anchor_state      = y_true[:, :, -1]

        print("ytrue shape: ",y_true.shape)
        print("ypred shape: ",y_pred)

        # filter out "ignore" anchors and negative anchors
        indices           = tf.where(keras.backend.equal(anchor_state, 1))
        regression        = tf.gather_nd(regression, indices)
        regression_target = tf.gather_nd(regression_target, indices)

        # compute smooth L1 loss
        # f(x) = 0.5 * (sigma * x)^2          if |x| < 1 / sigma / sigma
        #        |x| - 0.5 / sigma / sigma    otherwise
        regression_diff = regression - regression_target
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        num_pos = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        num_pos = keras.backend.cast(num_pos, dtype=keras.backend.floatx())
        # normalizer = keras.backend.cast(normalizer, dtype='float64')
        smooth_loss = keras.backend.sum(regression_loss) / num_pos
        ###############################################################################
        ## computing classification loss now
        ###############################################################################
        labels         = y_true[:, :,15]
        anchor_state   = y_true[:, :, -1]  # -1 for ignore, 0 for background, 1 for object
        classification = y_pred[:,:,14]

        # filter out "ignore" anchors
        indices        = tf.where(keras.backend.equal(anchor_state, 1))
        # indices        = tf.where(keras.backend.equal(anchor_state, 1))
        print("labels shape:",labels.shape)
        labels_pos     = tf.gather_nd(labels, indices)
        print("labels_pos shape: ",labels_pos.shape)
        class_pos      = tf.gather_nd(classification, indices)

        neg_indices    = tf.where(keras.backend.equal(anchor_state,0))
        labels_neg     = tf.gather_nd(labels,neg_indices)
        class_neg      = tf.gather_nd(classification,neg_indices)

        neg_ratio      = 3.0

        num_pos = tf.where(keras.backend.equal(anchor_state,1))
        num_pos = keras.backend.cast(keras.backend.shape(num_pos)[0], keras.backend.floatx())
        num_pos = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), num_pos)

        num_neg = keras.backend.cast(keras.backend.shape(neg_indices)[0], keras.backend.floatx())
        num_neg = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), num_neg)

        num_neg_used = tf.cast(tf.math.minimum(neg_ratio*num_pos,num_neg),tf.int32)

        # print("num_neg used: ",num_neg_used.shape)

        sorted_class_neg = tf.sort(class_neg,axis=-1,direction='DESCENDING',name=None)

        # num_neg_used_value = tf.Session.run()

        # _ , top_k = tf.math.top_k(class_neg,k = num_neg_used) 
        print("class neg shape: ",class_neg.shape)
        class_neg = sorted_class_neg[:num_neg_used]
        labels_neg = labels_neg[:num_neg_used]
        # print("class_neg shape: ",class_neg)

        # print("top_k shape: ",top_k.shape)

        # labels_neg = tf.gather_nd(labels_neg,top_k)
        # class_neg = tf.gather_nd(class_neg,top_k)

        cls_loss_pos = keras.backend.binary_crossentropy(labels_pos, class_pos)
        cls_loss_neg = keras.backend.binary_crossentropy(labels_neg, class_neg)

        # compute the normalizer: the number of positive anchors
        # normalizer = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer = tf.where(keras.backend.equal(anchor_state, 1))
        normalizer = keras.backend.cast(keras.backend.shape(normalizer)[0], keras.backend.floatx())
        normalizer = keras.backend.maximum(keras.backend.cast_to_floatx(1.0), normalizer)
        # normalizer = keras.backend.cast(normalizer, dtype='float64')
        cls_loss = ( keras.backend.sum(cls_loss_pos) + keras.backend.sum(cls_loss_neg) ) / normalizer
        
        ###############################################################################
        ## computing fovial points loss now
        ###############################################################################
        fovial_true = y_true[:,:,4:14]
        fovial_state = y_true[:,:,14]
        fovial_predicted = y_pred[:,:,4:14]

        indices = tf.where(keras.backend.equal(fovial_state,1))
        ft = tf.gather_nd(fovial_true,indices)
        fp = tf.gather_nd(fovial_predicted,indices)

        regression_diff = fp - ft
        regression_diff = keras.backend.abs(regression_diff)
        regression_loss = tf.where(
            keras.backend.less(regression_diff, 1.0 / sigma_squared),
            0.5 * sigma_squared * keras.backend.pow(regression_diff, 2),
            regression_diff - 0.5 / sigma_squared
        )

        # compute the normalizer: the number of positive anchors
        normalizer = keras.backend.maximum(1, keras.backend.shape(indices)[0])
        normalizer = keras.backend.cast(normalizer, dtype=keras.backend.floatx())
        # normalizer = keras.backend.cast(normalizer, dtype='float64')
        fovial_loss = keras.backend.sum(regression_loss) / normalizer



        return cls_loss+(0.25*smooth_loss) + (0.1*fovial_loss)
        # return smooth_loss

    return _added_loss