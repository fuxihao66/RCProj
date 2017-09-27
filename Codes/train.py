import tensorflow as tf
from model import Model
class single_GPU_trainer:
    def __init__(self, config, model):

        
        self.config = config
        self.lr = config.init_lr
        self.model = model
        # self.opt = tf.train.AdadeltaOptimizer(config.init_lr)
        self.opt = tf.train.AdadeltaOptimizer(learning_rate=model.get_lr())
        self.var_list = model.get_var_list()
        self.global_step = model.get_global_step()
        self.summary = model.summary

        self.loss = model.get_loss()
        self.grads = self.opt.compute_gradients(self.loss, var_list=self.var_list)
        self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)

    def get_train_op(self):
        return self.train_op
    
    def change_lr(self, new_lr):
        self.lr = new_lr
    def step(self, sess, batch, get_summary=False):
        assert isinstance(sess, tf.Session)
        ds = batch 
        feed_dict = self.model.get_feed_dict(ds, self.lr, True)
        # lr_dict = {self.learning_rate: self.lr}
        # feed_dict = feed_dict.update(lr_dict)

        if get_summary:
            loss, summary, train_op = \
                sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op

class MultiGPUTrainer(object):
    def __init__(self, config, models):
        model = models[0]
        assert isinstance(model, Model)

        self.lr = config.init_lr
        
        self.config = config
        self.model = model
        self.opt = tf.train.AdadeltaOptimizer(learning_rate=model.get_lr())
        self.var_list = model.get_var_list()
        self.global_step = model.get_global_step()
        self.summary = model.summary
        self.models = models
        losses = []
        grads_list = []
        
        for gpu_idx, model in enumerate(models):
            with tf.name_scope("grads_{}".format(gpu_idx)), tf.device("/{}:{}".format(config.device_type, gpu_idx)):
                loss = model.get_loss()
                grads = self.opt.compute_gradients(loss, var_list=self.var_list)
                losses.append(loss)
                grads_list.append(grads)
        
        self.loss = tf.add_n(losses)/len(losses)
        self.grads = average_gradients(grads_list)
        self.train_op = self.opt.apply_gradients(self.grads, global_step=self.global_step)
    def change_lr(self, new_lr):
        self.lr = new_lr
    def step(self, sess, batches, get_summary=False):
        assert isinstance(sess, tf.Session)
        feed_dict = {}
        for batch, model in zip(batches, self.models):
            ds = batch
            feed_dict.update(model.get_feed_dict(ds, self.lr, True))

        if get_summary:
            loss, summary, train_op = \
                sess.run([self.loss, self.summary, self.train_op], feed_dict=feed_dict)
        else:
            loss, train_op = sess.run([self.loss, self.train_op], feed_dict=feed_dict)
            summary = None
        return loss, summary, train_op


def average_gradients(tower_grads):
    """Calculate the average gradient for each shared variable across all towers.

    Note that this function provides a synchronization point across all towers.
    Args:
      tower_grads: List of lists of (gradient, variabsts of (gradient, variable) tuples. The outer list
        is over individual gradients. The inner list is over the gradient
        calculation for each tower.
    Returns:
       List of pairs of (gradient, variable) where the gradient has been averaged
       across all towers.
    """
    average_grads = []
    for grad_and_vars in zip(*tower_grads):
        # Note that each grad_and_vars looks like the following:
        #   ((grad0_gpu0, var0_gpu0), ... , (grad0_gpuN, var0_gpuN))
        grads = []
        for g, var in grad_and_vars:
            # Add 0 dimension to the gradients to represent the tower.
            assert g is not None, var.name
            expanded_g = tf.expand_dims(g, 0)

            # Append on a 'tower' dimension which we will average over below.
            grads.append(expanded_g)

        # Average over the 'tower' dimension.
        grad = tf.concat( grads, 0)
        grad = tf.reduce_mean(grad, 0)

        # Keep in mind that the Variables are redundant because they are shared
        # across towers. So .. we will just return the first tower's pointer to
        # the Variable.
        v = grad_and_vars[0][1]
        grad_and_var = (grad, v)
        average_grads.append(grad_and_var)
    return average_grads
