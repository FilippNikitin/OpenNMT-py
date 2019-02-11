class Opts(object):
    def __init__(self,
                 data,
                 save_model,
                 accum_count=1, adagrad_accumulator_init=0, adam_beta1=0.9, adam_beta2=0.999, audio_enc_pooling='1',
                 batch_size=64, batch_type='sents', bridge=False, brnn=None, cnn_kernel_width=3, context_gate=None,
                 copy_attn=False, copy_attn_force=False, copy_loss_by_seqlength=False, coverage_attn=False,
                 dec_layers=2, dec_rnn_size=500, decay_method='', decay_steps=10000,
                 decoder_type='rnn',
                 dropout=0.3, enc_layers=2, enc_rnn_size=500, encoder_type='rnn', epochs=0, exp='', exp_host='',
                 feat_merge='concat', feat_vec_exponent=0.7, feat_vec_size=-1, fix_word_vecs_dec=False,
                 fix_word_vecs_enc=False, generator_function='log_softmax', global_attention='general',
                 global_attention_function='softmax', gpu_backend='nccl', gpu_ranks=[], gpu_verbose_level=0, gpuid=[],
                 heads=8, image_channel_size=3, input_feed=1, keep_checkpoint=-1, label_smoothing=0.0,
                 lambda_coverage=1,
                 layers=-1, learning_rate=1.0, learning_rate_decay=0.5, log_file='', master_ip='localhost',
                 master_port=10000, max_generator_batches=32, max_grad_norm=5, model_type='text', normalization='sents',
                 optim='sgd', param_init=0.1, param_init_glorot=False, position_encoding=False, pre_word_vecs_dec=None,
                 pre_word_vecs_enc=None, report_every=50, reset_optim='none', reuse_copy_attn=False, rnn_size=-1,
                 rnn_type='LSTM', sample_rate=16000, save_checkpoint_steps=5000, seed=-1,
                 self_attn_type='scaled-dot', share_decoder_embeddings=False, share_embeddings=False,
                 src_word_vec_size=500, start_decay_steps=50000, tensorboard=False, tensorboard_log_dir='runs/onmt',
                 tgt_word_vec_size=500, train_from='', train_steps=100000, transformer_ff=2048, truncated_decoder=0,
                 valid_batch_size=32, valid_steps=10000, warmup_steps=4000, window_size=0.02, word_vec_size=-1,
                 world_size=1
                 ):
        self.data = data
        self.save_model = save_model
        self.accum_count = accum_count
        self.adagrad_accumulator_init = adagrad_accumulator_init
        self.adam_beta1 = adam_beta1
        self.adam_beta2 = adam_beta2
        self.audio_enc_pooling = audio_enc_pooling
        self.batch_size = batch_size
        self.batch_type = batch_type
        self.bridge = bridge
        self.brnn = brnn
        self.cnn_kernel_width = cnn_kernel_width
        self.context_gate = context_gate
        self.copy_attn = copy_attn
        self.copy_attn_force = copy_attn_force
        self.copy_loss_by_seqlength = copy_loss_by_seqlength
        self.coverage_attn = coverage_attn
        self.dec_layers = dec_layers
        self.dec_rnn_size = dec_rnn_size
        self.decay_method = decay_method
        self.decay_steps = decay_steps
        self.decoder_type = decoder_type
        self.dropout = dropout
        self.enc_layers = enc_layers
        self.enc_rnn_size = enc_rnn_size
        self.encoder_type = encoder_type
        self.epochs = epochs
        self.exp = exp
        self.exp_host = exp_host
        self.feat_merge = feat_merge
        self.feat_vec_exponent = feat_vec_exponent
        self.feat_vec_size = feat_vec_size
        self.fix_word_vecs_dec = fix_word_vecs_dec
        self.fix_word_vecs_enc = fix_word_vecs_enc
        self.generator_function = generator_function
        self.global_attention = global_attention
        self.global_attention_function = global_attention_function
        self.gpu_backend = gpu_backend
        self.gpu_ranks = gpu_ranks
        self.gpu_verbose_level = gpu_verbose_level
        self.gpuid = list(gpuid)
        self.heads = heads
        self.image_channel_size = image_channel_size
        self.input_feed = input_feed
        self.keep_checkpoint = keep_checkpoint
        self.label_smoothing = label_smoothing
        self.lambda_coverage = lambda_coverage
        self.layers = layers
        self.learning_rate = learning_rate
        self.learning_rate_decay = learning_rate_decay
        self.log_file = log_file
        self.master_ip = master_ip
        self.master_port = master_port
        self.max_generator_batches = max_generator_batches
        self.max_grad_norm = max_grad_norm
        self.model_type = model_type
        self.normalization = normalization
        self.optim = optim
        self.param_init = param_init
        self.param_init_glorot = param_init_glorot
        self.position_encoding = position_encoding
        self.pre_word_vecs_dec = pre_word_vecs_dec
        self.pre_word_vecs_enc = pre_word_vecs_enc
        self.report_every = report_every
        self.reset_optim = reset_optim
        self.reuse_copy_attn = reuse_copy_attn
        self.rnn_size = rnn_size
        self.rnn_type = rnn_type
        self.sample_rate = sample_rate
        self.save_checkpoint_steps = save_checkpoint_steps
        self.seed = seed
        self.self_attn_type = self_attn_type
        self.share_decoder_embeddings = share_decoder_embeddings
        self.share_embeddings = share_embeddings
        self.src_word_vec_size = src_word_vec_size
        self.start_decay_steps = start_decay_steps
        self.tensorboard = tensorboard
        self.tensorboard_log_dir = tensorboard_log_dir
        self.tgt_word_vec_size = tgt_word_vec_size
        self.train_from = train_from
        self.train_steps = train_steps
        self.transformer_ff = transformer_ff
        self.truncated_decoder = truncated_decoder
        self.valid_batch_size = valid_batch_size
        self.valid_steps = valid_steps
        self.warmup_steps = warmup_steps
        self.window_size = window_size
        self.word_vec_size = word_vec_size
        self.world_size = world_size


class PreprocessorOpts(object):
    def __init__(self,
                 save_data,
                 train_src,
                 train_tgt,
                 valid_src,
                 valid_tgt,
                 data_type="text",
                 src_dir="",
                 max_shard_size=0,
                 shard_size=1000000,
                 src_vocab="",
                 tgt_vocab="",
                 features_vocabs_prefix="",
                 src_vocab_size=50000,
                 tgt_vocab_size=50000,
                 src_words_min_frequency=0,
                 tgt_words_min_frequency=0,
                 dynamic_dict=False,
                 share_vocab=False,
                 src_seq_length=50,
                 src_seq_length_trunc=0,
                 tgt_seq_length=50,
                 tgt_seq_length_trunc=0,
                 lower=False,
                 shuffle=0,
                 seed=345,
                 report_every=100000,
                 log_file="",
                 sample_rate=16000,
                 window_size=.02,
                 window_stride=.01,
                 window="hamming",
                 image_channel_size=3

                 ):
        self.save_data = save_data
        self.train_src = train_src
        self.train_tgt = train_tgt
        self.valid_src = valid_src
        self.valid_tgt = valid_tgt
        self.data_type = data_type
        self.src_dir = src_dir
        self.max_shard_size = max_shard_size
        self.shard_size = shard_size
        self.src_vocab = src_vocab
        self.tgt_vocab = tgt_vocab
        self.features_vocabs_prefix = features_vocabs_prefix
        self.src_vocab_size = src_vocab_size
        self.tgt_vocab_size = tgt_vocab_size
        self.src_words_min_frequency = src_words_min_frequency
        self.tgt_words_min_frequency = tgt_words_min_frequency
        self.dynamic_dict = dynamic_dict
        self.share_vocab = share_vocab
        self.src_seq_length = src_seq_length
        self.src_seq_length_trunc = src_seq_length_trunc
        self.tgt_seq_length = tgt_seq_length
        self.tgt_seq_length_trunc = tgt_seq_length_trunc
        self.lower = lower
        self.shuffle = shuffle
        self.seed = seed
        self.report_every = report_every
        self.log_file = log_file
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.image_channel_size = image_channel_size


class TranslateOpts:
    """ Translation / inference options """
    def __init__(self,
                 #model,
                 src,
                 models=[],
                 avg_raw_probs=True,
                 data_type="text",
                 src_dir="",
                 tgt="",
                 output="pred.txt",
                 report_bleu=False,
                 report_rouge=False,
                 dynamic_dict=False,
                 share_vocab=False,
                 fast=False,
                 beam_size=5,
                 min_length=0,
                 max_length=100,
                 max_sent_lenght=100,
                 stepwise_penalty=True,
                 length_penalty='none',
                 coverage_penalty='none',
                 alpha=0.,
                 beta=-0.,
                 block_ngram_repeat=0.,
                 ignore_when_blocking=[],
                 replace_unk=False,
                 verbose=True,
                 log_file="",
                 attn_debug=False,
                 dump_beam="",
                 n_best=1,
                 batch_size=30,
                 gpu=-1,
                 sample_rate=16000,
                 window_size=.02,
                 window_stride=0.01,
                 window="hamming",
                 image_channel_size=3
                 ):
        #self.model = model
        self.avg_raw_probs = avg_raw_probs
        self.data_type = data_type
        self.models = models
        self.src = src
        self.src_dir = src_dir
        self.tgt = tgt
        self.output = output
        self.report_bleu = report_bleu
        self.report_rouge = report_rouge
        self.dynamic_dict = dynamic_dict
        self.share_vocab = share_vocab
        self.fast = fast
        self.beam_size = beam_size
        self.min_length = min_length
        self.max_length = max_length
        self.max_sent_lenght = max_sent_lenght
        self.stepwise_penalty = stepwise_penalty
        self.length_penalty = length_penalty
        self.coverage_penalty = coverage_penalty
        self.alpha = alpha
        self.beta = beta
        self.block_ngram_repeat = block_ngram_repeat
        self.ignore_when_blocking = ignore_when_blocking
        self.replace_unk = replace_unk
        self.verbose = verbose
        self.log_file = log_file
        self.attn_debug = attn_debug
        self.dump_beam = dump_beam
        self.n_best = n_best
        self.batch_size = batch_size
        self.gpu = gpu
        self.sample_rate = sample_rate
        self.window_size = window_size
        self.window_stride = window_stride
        self.window = window
        self.image_channel_size = image_channel_size


