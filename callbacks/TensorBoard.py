import tensorflow as tf
from tensorflow.python.ops import summary_ops_v2
from tensorflow.python.eager import context


class MyTensorBoardCallback(tf.keras.callbacks.TensorBoard):
    def __init__(self, args, log_dir='logs', histogram_freq=0, write_graph=True, write_images=False, update_freq='epoch',
                 profile_batch=2, embeddings_freq=0, embeddings_metadata=None, **kwargs):
        super().__init__(log_dir, histogram_freq, write_graph, write_images, update_freq, profile_batch,
                         embeddings_freq, embeddings_metadata, **kwargs)

        self.args = args

    @staticmethod
    def _parse_args(args):
        header_row = 'Parameter | Value\n' \
                     '----------|------\n'

        args_dict = vars(args)

        # TODO: Move logic below out of this class
        if not args_dict['comment']:
            del args_dict['comment']

        if not args_dict['use_early_stopping']:
            del args_dict['early_stopping_patience']
            del args_dict['early_stopping_min_delta']

        table_body = ["{} | {}".format(key, value) for key, value in args_dict.items()]

        markdown = header_row + "\n".join(table_body)
        return markdown

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)

        writer_name = self._train_run_name
        with context.eager_mode():
            with summary_ops_v2.always_record_summaries():
                writer = self._get_writer(writer_name)
                with writer.as_default():
                    tensor = tf.convert_to_tensor(self._parse_args(self.args))
                    tf.summary.text("run_settings", tensor, step=1)
