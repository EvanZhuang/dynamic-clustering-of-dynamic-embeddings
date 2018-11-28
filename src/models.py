import numpy as np
import os
import pickle
import tensorflow as tf
from tensorflow.contrib.distributions import Normal, Bernoulli
from tensorflow.contrib.tensorboard.plugins import projector

from sklearn.manifold import TSNE
from utils import plot_with_labels, variable_summaries

import tensorflow_probability as tfp
# from tensorflow_probability import edward2 as ed
# import functools

tfd = tfp.distributions
tfb = tfp.bijectors
# dtype = np.float32

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


class emb_model(object):
    def __init__(self, args, d, logdir):
        self.args = args

        self.K = args.K
        self.cs = args.cs
        self.ns = args.ns
        self.sig = args.sig
        self.dynamic = args.dynamic
        self.logdir = logdir
        self.N = d.N
        self.L = d.L
        self.T = d.T
        self.n_minibatch = d.n_train
        self.n_test = d.n_test
        self.labels = d.labels
        self.unigram = d.unigram
        self.dictionary = d.dictionary
        self.query_words = d.query_words
        self.train_feed = d.train_feed
        # self.valid_data = d.valid_data
        self.test_feed = d.test_feed
        self.n_iter = args.n_iter
        self.n_epochs = d.n_epochs
        self.n_valid = d.n_valid
        self.alpha_trainable = True
        self.VI = args.VI
        self.HMC = args.HMC
        self.components = args.components
        if args.init:
            fname = os.path.join('fits', d.name, args.init)
            if 'alpha_constant' in args.init:
                self.alpha_trainable = False
                fname = fname.replace('/alpha_constant', '')
            fit = pickle.load(open(fname, 'rb'))
            self.rho_init = fit['rho']
            self.alpha_init = fit['alpha']
        else:
            self.rho_init = (
                np.random.randn(
                    self.L,
                    self.K) /
                self.K).astype('float32')
            self.alpha_init = (
                np.random.randn(
                    self.L,
                    self.K) /
                self.K).astype('float32')
        if not self.alpha_trainable:
            self.rho_init = (
                0.1 *
                np.random.randn(
                    self.L,
                    self.K) /
                self.K).astype('float32')

        with open(os.path.join(self.logdir, "log_file.txt"), "a") as text_file:
            text_file.write(str(self.args))
            text_file.write('\n')

    def dump(self, fname):
        raise NotImplementedError()

    def detect_drift(self):
        raise NotImplementedError()

    def eval_log_like(self, feed_dict):
        return self.sess.run(
            tf.log(
                self.y_pos.mean() +
                0.000001),
            feed_dict=feed_dict)

    def plot_params(self, plot_only=500):
        with self.sess.as_default():
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            low_dim_embs_alpha2 = tsne.fit_transform(
                self.alpha.eval()[:plot_only])
            plot_with_labels(
                low_dim_embs_alpha2[:plot_only],
                self.labels[:plot_only], self.logdir + '/alpha.eps')

            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            low_dim_embs_rho2 = tsne.fit_transform(self.rho.eval()[:plot_only])
            plot_with_labels(
                low_dim_embs_rho2[:plot_only],
                self.labels[:plot_only], self.logdir + '/rho.eps')

    def print_word_similarities(self, words, num):
        query_word = tf.placeholder(dtype=tf.int32)
        # query_rho = tf.expand_dims(self.rho, [0])

        val_rho, idx_rho = tf.nn.top_k(
            tf.matmul(
                tf.nn.l2_normalize(
                    self.rho, dim=0), tf.nn.l2_normalize(
                    self.alpha, dim=1), transpose_b=True), num)

        for x in words:
            f_name = os.path.join(self.logdir, '%s_queries.txt' % (x))
            with open(f_name, "w+") as text_file:
                vr, ir = self.sess.run([val_rho, idx_rho], {
                                       query_word: self.dictionary[x]})
                text_file.write(
                    "\n\n===================================\
                    ==\n%s\n=====================================" %
                    (x))
                for ii in range(num):
                    text_file.write("\n%-20s %6.4f" %
                                    (self.labels[ir[0, ii]], vr[0, ii]))

    def print_topics(self, num):
        pass

    def initialize_training(self):
        optimizer = tf.train.AdamOptimizer()
        self.train = optimizer.minimize(self.loss)
        self.sess = tf.Session()
        with self.sess.as_default():
            tf.global_variables_initializer().run()

        variable_summaries('alpha', self.alpha)
        with tf.name_scope('objective'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('priors', self.log_prior)
            tf.summary.scalar('ll_pos', self.ll_pos)
            tf.summary.scalar('ll_neg', self.ll_neg)
        self.summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        self.saver = tf.train.Saver()
        config = projector.ProjectorConfig()
        alpha = config.embeddings.add()
        alpha.tensor_name = 'model/embeddings/alpha'
        alpha.metadata_path = '../vocab.tsv'
        if not self.dynamic:
            rho = config.embeddings.add()
            rho.tensor_name = 'model/embeddings/rho'
            rho.metadata_path = '../vocab.tsv'
        else:
            for t in range(self.T):
                rho = config.embeddings.add()
                rho.tensor_name = 'model/embeddings/rho_' + str(t)
                rho.metadata_path = '../vocab.tsv'
        projector.visualize_embeddings(self.train_writer, config)

    def train_embeddings(self):
        import matplotlib.pyplot as plt
        train_loss_list = []
        for data_pass in range(self.n_iter):
            for step in range(self.n_epochs):
                if step % 10 == 0:
                    print(str(step) + '/' + str(self.n_epochs) +
                          '   iter ' + str(data_pass))
                    summary, _, loss = self.sess.run(
                        [self.summaries, self.train, self.loss],
                        feed_dict=self.train_feed(self.placeholders))
                    self.train_writer.add_summary(
                        summary, data_pass * (self.n_epochs) + step)
                else:
                    _, loss = self.sess.run(
                        [self.train, self.loss],
                        feed_dict=self.train_feed(self.placeholders))
                train_loss_list.append(loss)
            self.dump(self.logdir + "/variational" + str(data_pass) + ".dat")
            self.saver.save(
                self.sess,
                os.path.join(
                    self.logdir,
                    "model.ckpt"),
                data_pass)
        title = "Training Loss over the process"
        plt.figure()
        plt.plot(train_loss_list, 'black', label='loss')
        plt.xlabel("N iteration")
        plt.ylabel("Training Loss")
        plt.title(title)
        plt.savefig(self.logdir + "/" + title + ".png")
        self.print_word_similarities(self.query_words, 10)
        if self.dynamic:
            words = self.detect_drift()
            self.print_word_similarities(words[:10], 1)
        self.plot_params(500)


class bern_emb_model(emb_model):
    def __init__(self, args, d, logdir):
        super(bern_emb_model, self).__init__(args, d, logdir)
        self.n_minibatch = self.n_minibatch.sum()

        with tf.name_scope('model'):
            # Data Placeholder
            with tf.name_scope('input'):
                self.placeholders = tf.placeholder(tf.int32)
                self.words = self.placeholders

            # Index Masks
            with tf.name_scope('context_mask'):
                self.p_mask = tf.cast(
                    tf.range(
                        int(self.cs / 2),
                        self.n_minibatch + int(self.cs / 2)), tf.int32)
                rows = tf.cast(tf.tile(tf.expand_dims(
                    tf.range(0, int(self.cs / 2)), [0]), [
                        self.n_minibatch, 1]), tf.int32)
                columns = tf.cast(
                    tf.tile(
                        tf.expand_dims(
                            tf.range(
                                0, self.n_minibatch), [1]), [
                            1, int(
                                self.cs / 2)]), tf.int32)
                self.ctx_mask = tf.concat(
                    [rows + columns, rows + columns + int(self.cs / 2) + 1], 1)

            with tf.name_scope('embeddings'):
                self.rho = tf.Variable(self.rho_init, name='rho')
                self.alpha = tf.Variable(
                    self.alpha_init,
                    name='alpha',
                    trainable=self.alpha_trainable)

                with tf.name_scope('priors'):
                    prior = Normal(loc=0.0, scale=self.sig)
                    if self.alpha_trainable:
                        self.log_prior = tf.reduce_sum(prior.log_prob(
                            self.rho) + prior.log_prob(self.alpha))
                    else:
                        self.log_prior = tf.reduce_sum(
                            prior.log_prob(self.rho))

            with tf.name_scope('natural_param'):
                # Taget and Context Indices
                with tf.name_scope('target_word'):
                    self.p_idx = tf.gather(self.words, self.p_mask)
                    self.p_rho = tf.squeeze(tf.gather(self.rho, self.p_idx))

                # Negative samples
                with tf.name_scope('negative_samples'):
                    unigram_logits = tf.tile(
                        tf.expand_dims(
                            tf.log(
                                tf.constant(
                                    self.unigram)), [0]), [
                            self.n_minibatch, 1])
                    self.n_idx = tf.multinomial(unigram_logits, self.ns)
                    self.n_rho = tf.gather(self.rho, self.n_idx)

                with tf.name_scope('context'):
                    self.ctx_idx = tf.squeeze(
                        tf.gather(self.words, self.ctx_mask))
                    self.ctx_alphas = tf.gather(self.alpha, self.ctx_idx)

                # Natural parameter
                ctx_sum = tf.reduce_sum(self.ctx_alphas, [1])
                self.p_eta = tf.expand_dims(tf.reduce_sum(
                    tf.multiply(self.p_rho, ctx_sum), -1), 1)
                self.n_eta = tf.reduce_sum(
                    tf.multiply(
                        self.n_rho, tf.tile(
                            tf.expand_dims(
                                ctx_sum, 1), [
                                1, self.ns, 1])), -1)

            # Conditional likelihood
            self.y_pos = Bernoulli(logits=self.p_eta)
            self.y_neg = Bernoulli(logits=self.n_eta)

            self.ll_pos = tf.reduce_sum(self.y_pos.log_prob(1.0))
            self.ll_neg = tf.reduce_sum(self.y_neg.log_prob(0.0))

            self.log_likelihood = self.ll_pos + self.ll_neg

            # scale = 1.0 * self.N / self.n_minibatch
            self.loss = - (self.log_likelihood + self.log_prior)

    def dump(self, fname):
        with self.sess.as_default():
            dat = {'rho': self.rho.eval(),
                   'alpha': self.alpha.eval()}
        pickle.dump(dat, open(fname, "wb"))


class dynamic_bern_emb_model(emb_model):
    def __init__(self, args, d, logdir):
        super(dynamic_bern_emb_model, self).__init__(args, d, logdir)

        with tf.name_scope('model'):
            with tf.name_scope('embeddings'):
                self.alpha = tf.Variable(
                    self.alpha_init,
                    name='alpha',
                    trainable=self.alpha_trainable)

                self.rho_t = {}
                for t in range(-1, self.T):
                    self.rho_t[t] = tf.Variable(
                      self.rho_init +
                      0.001 * tf.random_normal([self.L, self.K]) / self.K,
                      name='rho_' + str(t))

                with tf.name_scope('priors'):
                    global_prior = Normal(loc=0.0, scale=self.sig)
                    local_prior = Normal(loc=0.0, scale=self.sig / 100.0)

                    self.log_prior = tf.reduce_sum(
                        global_prior.log_prob(self.alpha))
                    self.log_prior += tf.reduce_sum(
                        global_prior.log_prob(self.rho_t[-1]))
                    for t in range(self.T):
                        self.log_prior += tf.reduce_sum(
                            local_prior.log_prob(
                                self.rho_t[t] - self.rho_t[t - 1]))

            with tf.name_scope('likelihood'):
                self.placeholders = {}
                self.y_pos = {}
                self.y_neg = {}
                self.ll_pos = 0.0
                self.ll_neg = 0.0
                for t in range(self.T):
                    # Index Masks
                    p_mask = tf.range(int(self.cs / 2),
                                      self.n_minibatch[t] + int(self.cs / 2))
                    rows = tf.tile(tf.expand_dims(
                        tf.range(0, int(self.cs / 2)), [0]), [
                            self.n_minibatch[t], 1])
                    columns = tf.tile(
                        tf.expand_dims(
                            tf.range(
                                0, self.n_minibatch[t]), [1]), [
                            1, int(
                                self.cs / 2)])

                    ctx_mask = tf.concat(
                        [rows + columns, rows + columns + int(
                            self.cs / 2) + 1], 1)

                    # Data Placeholder
                    self.placeholders[t] = tf.placeholder(
                        tf.int32, shape=(self.n_minibatch[t] + self.cs))

                    # Taget and Context Indices
                    p_idx = tf.gather(self.placeholders[t], p_mask)
                    ctx_idx = tf.squeeze(
                        tf.gather(
                            self.placeholders[t],
                            ctx_mask))

                    # Negative samples
                    unigram_logits = tf.tile(
                        tf.expand_dims(
                            tf.log(
                                tf.constant(
                                    self.unigram)), [0]), [
                            self.n_minibatch[t], 1])
                    n_idx = tf.multinomial(unigram_logits, self.ns)

                    # Context vectors
                    ctx_alphas = tf.gather(self.alpha, ctx_idx)

                    p_rho = tf.squeeze(tf.gather(self.rho_t[t], p_idx))
                    n_rho = tf.gather(self.rho_t[t], n_idx)

                    # Natural parameter
                    ctx_sum = tf.reduce_sum(ctx_alphas, [1])
                    p_eta = tf.expand_dims(tf.reduce_sum(
                        tf.multiply(p_rho, ctx_sum), -1), 1)
                    n_eta = tf.reduce_sum(
                        tf.multiply(
                            n_rho, tf.tile(
                                tf.expand_dims(
                                    ctx_sum, 1), [
                                    1, self.ns, 1])), -1)

                    # Conditional likelihood
                    self.y_pos[t] = Bernoulli(logits=p_eta)
                    self.y_neg[t] = Bernoulli(logits=n_eta)

                    self.ll_pos += tf.reduce_sum(self.y_pos[t].log_prob(1.0))
                    self.ll_neg += tf.reduce_sum(self.y_neg[t].log_prob(0.0))

            self.loss = - ((self.ll_pos + self.ll_neg) + self.log_prior)

    def dump(self, fname):
        with self.sess.as_default():
            dat = {'alpha': self.alpha.eval()}
            for t in range(self.T):
                dat['rho_' + str(t)] = self.rho_t[t].eval()
        pickle.dump(dat, open(fname, "wb"))

    def eval_log_like(self, feed_dict):
        log_p = np.zeros((0, 1))
        for t in range(self.T):
            log_p_t = self.sess.run(
                tf.log(
                    self.y_pos[t].mean() +
                    0.000001),
                feed_dict=feed_dict)
            log_p = np.vstack((log_p, log_p_t))
        return log_p

    def plot_params(self, plot_only=500):
        with self.sess.as_default():
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            low_dim_embs_alpha = tsne.fit_transform(
                self.alpha.eval()[:plot_only])
            plot_with_labels(
                low_dim_embs_alpha[:plot_only],
                self.labels[:plot_only],
                self.logdir + '/alpha.eps')
            for t in [0, int(self.T / 2), self.T - 1]:
                w_idx_t = range(plot_only)
                np_rho = self.rho_t[t].eval()
                tsne = TSNE(
                    perplexity=30,
                    n_components=2,
                    init='pca',
                    n_iter=5000)
                low_dim_embs_rho = tsne.fit_transform(np_rho[w_idx_t, :])
                plot_with_labels(
                    low_dim_embs_rho,
                    self.labels[w_idx_t],
                    self.logdir +
                    '/rho_' +
                    str(t) +
                    '.eps')

    def detect_drift(self, metric='total_dist'):
        if metric == 'total_dist':
            tf_dist, tf_w_idx = tf.nn.top_k(tf.reduce_sum(
                tf.square(self.rho_t[self.T - 1] - self.rho_t[0]), 1), 500)
        else:
            print('unknown metric')
            return
        dist, w_idx = self.sess.run([tf_dist, tf_w_idx])
        words = self.labels[w_idx]
        f_name = self.logdir + '/top_drifting_words.txt'
        with open(f_name, "w+") as text_file:
            for (w, drift) in zip(w_idx, dist):
                text_file.write("\n%-20s %6.4f" % (self.labels[w], drift))
        return words

    def print_word_similarities(self, words, num):
        query_word = tf.placeholder(dtype=tf.int32)
        query_rho_t = tf.placeholder(dtype=tf.float32)

        val_rho, idx_rho = tf.nn.top_k(
            tf.matmul(
                tf.nn.l2_normalize(
                    query_rho_t, dim=0), tf.nn.l2_normalize(
                    self.alpha, dim=1), transpose_b=True), num)

        for x in words:
            f_name = os.path.join(self.logdir, '%s_queries.txt' % (x))
            with open(f_name, "w+") as text_file:
                for t_idx in range(self.T):
                    with self.sess.as_default():
                        rho_t = self.rho_t[t_idx].eval()
                    vr, ir = self.sess.run(
                        [val_rho, idx_rho],
                        {query_word: self.dictionary[x],
                            query_rho_t: rho_t})
                    text_file.write(
                        "\n\n===================================\
                          ==\n%s, t = %d\n======================\
                          ===============" %
                        (x, t_idx))
                    for ii in range(num):
                        text_file.write("\n%-20s %6.4f" %
                                        (self.labels[ir[0, ii]], vr[0, ii]))

    def print_word_similarities_test(self, words, num):
        query_word = tf.placeholder(dtype=tf.int32)
        query_rho_t = tf.placeholder(dtype=tf.float32)
        # calOnly = 500

        for x in words:
            f_name = os.path.join(self.logdir, '%s_queries.txt' % (x))
            with open(f_name, "w+") as text_file:
                for t_idx in range(self.T):
                    val_rho, idx_rho = tf.nn.top_k(
                        tf.transpose(tf.matmul(tf.nn.l2_normalize(
                            query_rho_t, axis=1), tf.nn.l2_normalize(
                                tf.expand_dims(query_rho_t[self.dictionary[
                                    x], :], -1), axis=0))), num)
                    with self.sess.as_default():
                        rho_t = self.rho_t[t_idx]
                        print(rho_t.shape)
                    vr, ir = self.sess.run([val_rho, idx_rho], {
                                           query_word: self.dictionary[
                                               x], query_rho_t: rho_t})
                    text_file.write(
                        "\n\n===================================\
                        ==\n%s, t = %d\n========================\
                        =============" %
                        (x, t_idx))
                    for ii in range(num):
                        text_file.write("\n%-20s %6.4f" %
                                        (self.labels[ir[0, ii]], vr[0, ii]))

    def print_topics(self, num):
        pass


class dynamic_context_dynamic_bern_emb_model(emb_model):
    def __init__(self, args, d, logdir):
        super(
            dynamic_context_dynamic_bern_emb_model,
            self).__init__(
            args,
            d,
            logdir)

        with tf.name_scope('model'):
            with tf.name_scope('embeddings'):
                if args.dinit:
                    fname = os.path.join('fits', d.name, args.dinit)
                    if 'alpha_constant' in args.dinit:
                        self.alpha_trainable = False
                        fname = fname.replace('/alpha_constant', '')
                    fit = pickle.load(open(fname, 'rb'))
                    self.rho_t = {}
                    self.rho_t[-1] = tf.Variable(
                        self.rho_init + 0.001 * tf.random_normal(
                            [self.L, self.K]) / self.K, name='rho_' + str(-1))
                    for t in range(self.T):
                        self.rho_t[t] = fit['rho_' + str(t)]
                    self.alpha = fit['alpha']
                else:
                    # self.alpha = tf.Variable(
                    # self.alpha_init, name='alpha', tr
                    self.alpha_t = {}
                    for t in range(-1, self.T):
                        self.alpha_t[t] = tf.Variable(
                            self.alpha_init
                            + 0.001 * tf.random_normal(
                                [self.L, self.K]) / self.K,
                            name='alpha_' + str(t))

                    self.rho_t = {}
                    for t in range(-1, self.T):
                        self.rho_t[t] = tf.Variable(
                            self.rho_init
                            + 0.001 * tf.random_normal(
                                [self.L, self.K]) / self.K,
                            name='rho_' + str(t))

                with tf.name_scope('priors'):
                    global_prior = Normal(loc=0.0, scale=self.sig)
                    local_prior = Normal(loc=0.0, scale=self.sig / 100.0)

                    # self.log_prior = tf.reduce_sum(
                    # global_prior.log_prob(self.
                    self.log_prior = tf.reduce_sum(
                        global_prior.log_prob(self.alpha_t[-1]))
                    for t in range(self.T):
                        self.log_prior += tf.reduce_sum(local_prior.log_prob(
                            self.alpha_t[t] - self.alpha_t[t - 1]))

                    self.log_prior += tf.reduce_sum(
                        global_prior.log_prob(self.rho_t[-1]))
                    for t in range(self.T):
                        self.log_prior += tf.reduce_sum(
                            local_prior.log_prob(
                                self.rho_t[t] - self.rho_t[t - 1]))

            with tf.name_scope('likelihood'):
                self.placeholders = {}
                self.y_pos = {}
                self.y_neg = {}
                self.ll_pos = 0.0
                self.ll_neg = 0.0
                for t in range(self.T):
                    # Index Masks
                    p_mask = tf.range(int(self.cs / 2),
                                      self.n_minibatch[t] + int(self.cs / 2))
                    rows = tf.tile(tf.expand_dims(
                        tf.range(0, int(self.cs / 2)), [0]),
                         [self.n_minibatch[t], 1])
                    columns = tf.tile(
                        tf.expand_dims(
                            tf.range(
                                0, self.n_minibatch[t]), [1]), [
                            1, int(
                                self.cs / 2)])

                    ctx_mask = tf.concat(
                        [rows + columns, rows + columns
                         + int(self.cs / 2) + 1], 1)

                    # Data Placeholder
                    self.placeholders[t] = tf.placeholder(
                        tf.int32, shape=(self.n_minibatch[t] + self.cs))

                    # Taget and Context Indices
                    p_idx = tf.gather(self.placeholders[t], p_mask)
                    # ctx_idx = tf.squeeze(
                    # tf.gather(self.placeholders[t], ctx_m
                    ctx_idx = tf.gather(self.placeholders[t], ctx_mask)

                    # Negative samples
                    unigram_logits = tf.tile(
                        tf.expand_dims(
                            tf.log(
                                tf.constant(
                                    self.unigram)), [0]), [
                            self.n_minibatch[t], 1])
                    n_idx = tf.multinomial(unigram_logits, self.ns)

                    # Context vectors
                    # ctx_alphas = tf.gather(self.alpha, ctx_idx)#######
                    ctx_alphas = tf.squeeze(
                        tf.gather(self.alpha_t[t], ctx_idx))

                    p_rho = tf.squeeze(tf.gather(self.rho_t[t], p_idx))
                    n_rho = tf.gather(self.rho_t[t], n_idx)

                    # Natural parameter
                    ctx_sum = tf.reduce_sum(ctx_alphas, [1])
                    p_eta = tf.expand_dims(tf.reduce_sum(
                        tf.multiply(p_rho, ctx_sum), -1), 1)
                    n_eta = tf.reduce_sum(
                        tf.multiply(
                            n_rho, tf.tile(
                                tf.expand_dims(
                                    ctx_sum, 1), [
                                    1, self.ns, 1])), -1)

                    # Conditional likelihood
                    self.y_pos[t] = Bernoulli(logits=p_eta)
                    self.y_neg[t] = Bernoulli(logits=n_eta)

                    self.ll_pos += tf.reduce_sum(self.y_pos[t].log_prob(1.0))
                    self.ll_neg += tf.reduce_sum(self.y_neg[t].log_prob(0.0))

            self.loss = - (self.n_epochs * (self.ll_pos +
                                            self.ll_neg) + self.log_prior)

    def initialize_training(self):
        optimizer = tf.train.AdamOptimizer()
        self.train = optimizer.minimize(self.loss)
        self.sess = tf.Session()
        with self.sess.as_default():
            tf.global_variables_initializer().run()

        # variable_summaries('alpha', self.alpha)#######
        for t in range(self.T):
            variable_summaries('alpha' + str(t), self.alpha_t[t])
        with tf.name_scope('objective'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('priors', self.log_prior)
            tf.summary.scalar('ll_pos', self.ll_pos)
            tf.summary.scalar('ll_neg', self.ll_neg)
        self.summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        self.saver = tf.train.Saver()
        config = projector.ProjectorConfig()
        alpha = config.embeddings.add()
        alpha.tensor_name = 'model/embeddings/alpha'
        alpha.metadata_path = '../vocab.tsv'
        if not self.dynamic:
            rho = config.embeddings.add()
            rho.tensor_name = 'model/embeddings/rho'
            rho.metadata_path = '../vocab.tsv'
        else:
            for t in range(self.T):
                rho = config.embeddings.add()
                rho.tensor_name = 'model/embeddings/rho_' + str(t)
                rho.metadata_path = '../vocab.tsv'
        projector.visualize_embeddings(self.train_writer, config)

    def dump(self, fname):
        with self.sess.as_default():
            dat = {}
            # dat = {'alpha':  self.alpha.eval()}#######
            for t in range(self.T):
                dat['alpha_' + str(t)] = self.alpha_t[t].eval()
                dat['rho_' + str(t)] = self.rho_t[t].eval()
        pickle.dump(dat, open(fname, "wb"))

    def eval_log_like(self, feed_dict):
        log_p = np.zeros((0, 1))
        for t in range(self.T):
            log_p_t = self.sess.run(
                tf.log(
                    self.y_pos[t].mean() +
                    0.000001),
                feed_dict=feed_dict)
            log_p = np.vstack((log_p, log_p_t))
        return log_p

    def plot_params(self, plot_only=500):
        with self.sess.as_default():
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            # low_dim_embs_alpha = tsne.fit_transform(
            # self.alpha.eval()[:plot_on
            # plot_with_labels(low_dim_embs_alpha[:plot_only],
            # self.labels[:plot_only], self.logdir + '/alpha.eps')
            for t in [0, int(self.T / 2), self.T - 1]:
                w_idx_t = range(plot_only)
                np_rho = self.rho_t[t].eval()
                tsne = TSNE(
                    perplexity=30,
                    n_components=2,
                    init='pca',
                    n_iter=5000)
                low_dim_embs_rho = tsne.fit_transform(np_rho[w_idx_t, :])
                plot_with_labels(
                    low_dim_embs_rho,
                    self.labels[w_idx_t],
                    self.logdir +
                    '/rho_' +
                    str(t) +
                    '.eps')

    def detect_drift(self, metric='total_dist'):
        if metric == 'total_dist':
            tf_dist, tf_w_idx = tf.nn.top_k(tf.reduce_sum(
                tf.square(self.rho_t[self.T - 1] - self.rho_t[0]), 1), 500)
        else:
            print('unknown metric')
            return
        dist, w_idx = self.sess.run([tf_dist, tf_w_idx])
        words = self.labels[w_idx]
        f_name = self.logdir + '/top_drifting_words.txt'
        with open(f_name, "w+") as text_file:
            for (w, drift) in zip(w_idx, dist):
                text_file.write("\n%-20s %6.4f" % (self.labels[w], drift))
        return words

    def print_word_similarities(self, words, num):
        pass

    def print_word_similarities_test(self, words, num):
        pass

    def print_topics(self, num):
        pass


class MVNCholPrecisionTriL(tfd.TransformedDistribution):
    """MVN from loc and (Cholesky) precision matrix."""

    def __init__(self, loc, chol_precision_tril, name=None):
        super(MVNCholPrecisionTriL, self).__init__(
            distribution=tfd.Independent(tfd.Normal(tf.zeros_like(loc),
                                                    scale=tf.ones_like(loc)),
                                         reinterpreted_batch_ndims=1),
            bijector=tfb.Chain([
                tfb.Affine(shift=loc),
                tfb.Invert(tfb.Affine(scale_tril=chol_precision_tril,
                                      adjoint=True)),
            ]),
            name=name)


class GMM_SGAVI(object):
    def __init__(self, xn, K=3, maxIter=50, BATCH_SIZE=100):
        from scipy import random
        from numpy.linalg import det, inv
        from scipy.special import psi

        def dirichlet_expectation(alpha):
            return tf.subtract(
                tf.digamma(
                    tf.add(
                        alpha, np.finfo(
                            np.float32).eps)), tf.digamma(
                    tf.reduce_sum(alpha)))

        def dirichlet_expectation_k(alpha, k):
            # Dirichlet expectation computation
            # \Psi(\alpha_{k}) - \Psi(\sum_{i=1}^{K}(\alpha_{i}))
            return psi(alpha[k] + np.finfo(np.float32).eps) - \
                psi(np.sum(alpha))

        def log_beta_function(x):
            return tf.subtract(
                tf.reduce_sum(tf.lgamma(tf.add(x, np.finfo(np.float32).eps))),
                tf.lgamma(tf.reduce_sum(tf.add(x, np.finfo(np.float32).eps))))

        def softmax(x):
            """
            Softmax computation
            e^{x} / sum_{i=1}^{K}(e^x_{i})
            """
            e_x = np.exp(x - np.max(x))
            return (e_x + np.finfo(np.float32).eps) / \
                   (e_x.sum(axis=0) + np.finfo(np.float32).eps)

        def log_(x):
            return tf.log(tf.add(x, np.finfo(np.float32).eps))

        N, D = xn.shape
        dtype = np.float32

        # Priors
        alpha_o = np.array([1.0] * K)
        nu_o = np.array([float(D)])
        w_o = np.dot(random.rand(D, D), random.rand(D, D).transpose())
        m_o = np.array([0.0] * D)
        beta_o = np.array([0.7])

        # Variational parameters intialization
        lambda_phi_var = np.random.dirichlet(alpha_o, N)
        lambda_pi_var = np.zeros(shape=K)
        lambda_beta_var = np.zeros(shape=K)
        lambda_nu_var = np.zeros(shape=K) + D
        lambda_m_var = np.random.rand(K, D)
        lambda_w_var = np.array([np.copy(w_o) for _ in range(K)])

        lambda_phi = tf.Variable(lambda_phi_var, trainable=False, dtype=dtype)
        lambda_pi_var = tf.Variable(lambda_pi_var, dtype=dtype)
        lambda_beta_var = tf.Variable(lambda_beta_var, dtype=dtype)
        lambda_nu_var = tf.Variable(lambda_nu_var, dtype=dtype)
        lambda_m = tf.Variable(lambda_m_var, dtype=dtype)
        lambda_w_var = tf.Variable(lambda_w_var, dtype=dtype)

        # Maintain numerical stability
        lambda_pi = tf.nn.softplus(lambda_pi_var)
        lambda_beta = tf.nn.softplus(lambda_beta_var)
        lambda_nu = tf.add(
            tf.nn.softplus(lambda_nu_var), tf.cast(
                D, dtype=dtype))

        mats = []
        for k in range(K):
            aux1 = tf.matrix_set_diag(
                tf.matrix_band_part(lambda_w_var[k], -1, 0),
                tf.nn.softplus(tf.diag_part(lambda_w_var[k])))
            mats.append(tf.matmul(aux1, aux1, transpose_b=True))
        lambda_w = tf.convert_to_tensor(mats)

        idx_tensor = tf.placeholder(tf.int32, shape=(BATCH_SIZE))

        alpha_o = tf.convert_to_tensor(alpha_o, dtype=dtype)
        nu_o = tf.convert_to_tensor(nu_o, dtype=dtype)
        w_o = tf.convert_to_tensor(w_o, dtype=dtype)
        m_o = tf.convert_to_tensor(m_o, dtype=dtype)
        beta_o = tf.convert_to_tensor(beta_o, dtype=dtype)

        # Evidence Lower Bound definition
        e3 = tf.convert_to_tensor(0., dtype=dtype)
        e2 = tf.convert_to_tensor(0., dtype=dtype)
        h2 = tf.convert_to_tensor(0., dtype=dtype)
        e1 = tf.add(-log_beta_function(alpha_o),
                    tf.reduce_sum(tf.multiply(
                        tf.subtract(alpha_o, tf.ones(K, dtype=dtype)),
                        dirichlet_expectation(lambda_pi))))
        h1 = tf.subtract(log_beta_function(lambda_pi),
                         tf.reduce_sum(tf.multiply(
                             tf.subtract(lambda_pi, tf.ones(K, dtype=dtype)),
                             dirichlet_expectation(lambda_pi))))
        logdet = tf.log(tf.convert_to_tensor([
            tf.matrix_determinant(lambda_w[k, :, :]) for k in range(K)]))
        logDeltak = tf.add(tf.digamma(tf.div(lambda_nu, 2.)),
                           tf.add(tf.digamma(tf.div(tf.subtract(
                               lambda_nu, tf.cast(1., dtype=dtype)),
                               tf.cast(2., dtype=dtype))),
                               tf.add(tf.multiply(
                                tf.cast(2., dtype=dtype),
                                tf.cast(tf.log(2.), dtype=dtype)), logdet)))
        for i in range(BATCH_SIZE):
            n = idx_tensor[i]
            # w = tf.Print(tf.subtract(tf.gather(xn, n), lambda_m[k, :]),
            # [tf.subtract(tf.gather(xn, n), lambda_m[k, :])])
            e2 = tf.add(e2, tf.reduce_sum(
                tf.multiply(tf.gather(lambda_phi, n),
                            dirichlet_expectation(lambda_pi))))
            h2 = tf.add(
                h2, -tf.reduce_sum(
                    tf.multiply(
                        tf.gather(
                            lambda_phi, n), log_(
                            tf.gather(
                                lambda_phi, n)))))
            product = tf.convert_to_tensor([tf.reduce_sum(tf.matmul(
                tf.matmul(
                    tf.reshape(tf.subtract(tf.gather(xn, n), lambda_m[k, :]),
                               [1, D]),
                    lambda_w[k, :, :]),
                tf.reshape(tf.transpose(tf.subtract(
                    tf.gather(xn, n), lambda_m[k, :])),
                           [D, 1]))) for k in range(K)])
            aux = tf.transpose(tf.subtract(
                logDeltak, tf.add(tf.multiply(tf.cast(2., dtype=dtype),
                                              tf.cast(tf.log(2. * np.pi),
                                                      dtype=dtype)),
                                  tf.add(tf.multiply(lambda_nu, product),
                                         tf.div(tf.cast(2., dtype=dtype),
                                                lambda_beta)))))
            e3 = tf.add(e3, tf.reduce_sum(
                tf.multiply(tf.cast(1 / 2., dtype=dtype),
                            tf.multiply(tf.gather(lambda_phi, n), aux))))

        product = tf.convert_to_tensor([tf.reduce_sum(tf.matmul(
            tf.matmul(tf.reshape(tf.subtract(lambda_m[k, :], m_o), [1, D]),
                      lambda_w[k, :, :]),
            tf.reshape(tf.transpose(tf.subtract(
                lambda_m[k, :], m_o)), [D, 1]))) for
            k in range(K)])
        traces = tf.convert_to_tensor([tf.trace(tf.matmul(
            tf.matrix_inverse(w_o), lambda_w[k, :, :])) for k in range(K)])
        h4 = tf.reduce_sum(
            tf.add(tf.cast(1., dtype=dtype),
                   tf.subtract(tf.log(tf.cast(2., dtype=dtype) * np.pi),
                               tf.multiply(tf.cast(1. / 2., dtype=dtype),
                                           tf.add(tf.cast(tf.log(lambda_beta),
                                                          dtype=dtype),
                                                  logdet)))))
        aux = tf.add(tf.multiply(tf.cast(1. / 2., dtype=dtype), tf.log(
            tf.cast(tf.constant(np.pi), dtype=dtype))),
            tf.add(tf.lgamma(
                tf.div(lambda_nu, tf.cast(2., dtype=dtype))),
            tf.lgamma(tf.div(
                tf.subtract(lambda_nu, tf.cast(1., dtype=dtype)),
                tf.cast(2., dtype=dtype)))))
        logB = tf.add(
            tf.multiply(
                tf.div(
                    lambda_nu, 2.), logdet), tf.add(
                tf.multiply(
                    lambda_nu, tf.log(
                        tf.cast(
                            2., dtype=dtype))), aux))
        h5 = tf.reduce_sum(
            tf.subtract(
                tf.add(
                    logB, lambda_nu), tf.multiply(
                    tf.div(
                        tf.subtract(
                            lambda_nu, tf.cast(
                                3., dtype=dtype)), tf.cast(
                            2., dtype=dtype)), logDeltak)))
        aux = tf.add(
            tf.multiply(
                tf.cast(
                    2.,
                    dtype=dtype),
                tf.log(
                    tf.cast(
                        2.,
                        dtype=dtype) * np.pi)),
            tf.add(
                tf.multiply(
                    beta_o,
                    tf.multiply(
                        lambda_nu,
                        product)),
                tf.multiply(
                    tf.cast(
                        2.,
                        dtype=dtype),
                    tf.div(
                        beta_o,
                        lambda_beta))))
        e4 = tf.reduce_sum(
            tf.multiply(
                tf.cast(
                    1. / 2.,
                    dtype=dtype),
                tf.subtract(
                    tf.add(
                        tf.log(beta_o),
                        logDeltak),
                    aux)))
        logB = tf.add(
            tf.multiply(tf.div(nu_o, tf.cast(2., dtype=dtype)),
                        tf.log(tf.matrix_determinant(w_o))),
            tf.add(tf.multiply(nu_o, tf.cast(tf.log(2.), dtype=dtype)),
                   tf.add(tf.multiply(tf.cast(1. / 2., dtype=dtype),
                                      tf.cast(tf.log(np.pi), dtype=dtype)),
                          tf.add(tf.lgamma(
                              tf.div(nu_o, tf.cast(2., dtype=dtype))),
                              tf.lgamma(tf.div(tf.subtract(
                                  nu_o, tf.cast(1., dtype=dtype)),
                                  tf.cast(2., dtype=dtype)))))))
        e5 = tf.reduce_sum(tf.add(-logB, tf.subtract(
            tf.multiply(tf.div(tf.subtract(nu_o, tf.cast(3., dtype=dtype)),
                               tf.cast(2., dtype=dtype)), logDeltak),
            tf.multiply(tf.div(lambda_nu, tf.cast(2., dtype=dtype)), traces))))
        LB = e1 + e2 + e3 + e4 + e5 + h1 + h2 + h4 + h5

        def update_lambda_phi(lambda_phi, lambda_pi, lambda_m,
                              lambda_nu, lambda_w, lambda_beta, xn, idx, K, D):
            # Update lambda_phi
            # softmax[dirichlet_expectation(lambda_pi) +
            #         lambda_m * lambda_nu * lambda_w^{-1} * x_{n} -
            #         1/2 * lambda_nu * lambda_w^{-1} * x_{n} * x_{n}.T -
            #         1/2 * lambda_beta^{-1} -
            #         lambda_nu * lambda_m.T * lambda_w^{-1} * lambda_m +
            #         D/2 * log(2) +
            #         1/2 * sum_{i=1}^{D}(\Psi(lambda_nu/2 + (1-i)/2)) -
            #         1/2 log(|lambda_w|)]
            for n in idx:
                for k in range(K):
                    inv_lambda_w = inv(lambda_w[k, :, :])
                    lambda_phi[n, k] = dirichlet_expectation_k(lambda_pi, k)
                    lambda_phi[n, k] += np.dot(lambda_m[k, :], np.dot(
                        lambda_nu[k] * inv_lambda_w, xn[n, :]))
                    lambda_phi[n, k] -= np.trace(
                        np.dot((1 / 2.) * lambda_nu[k] * inv_lambda_w,
                               np.outer(xn[n, :], xn[n, :])))
                    lambda_phi[n, k] -= (D / 2.) * (1 / lambda_beta[k])
                    lambda_phi[n, k] -= (1. / 2.) * np.dot(
                        np.dot(lambda_nu[k] * lambda_m[k, :].T, inv_lambda_w),
                        lambda_m[k, :])
                    lambda_phi[n, k] += (D / 2.) * np.log(2.)
                    lambda_phi[n, k] += (1 / 2.) * np.sum(
                        [psi((lambda_nu[k] / 2.) +
                         ((1 - i) / 2.)) for i in range(D)])
                    lambda_phi[n, k] -= (1 / 2.) * \
                        np.log(det(lambda_w[k, :, :]))
                lambda_phi[n, :] = softmax(lambda_phi[n, :])
            return lambda_phi

        global_step = tf.Variable(0)
        learning_rate = tf.train.exponential_decay(0.05, global_step,
                                                   100, 0.96, staircase=True)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        grads_and_vars = optimizer.compute_gradients(
            -LB, var_list=[lambda_pi_var, lambda_m,
                           lambda_beta_var, lambda_nu_var, lambda_w_var])
        train = optimizer.apply_gradients(
            grads_and_vars, global_step=global_step)

        # Inference
        init = tf.global_variables_initializer()
        sess.run(init)
        lbs = []
        aux_lbs = []
        n_iters = 0

        phi_out = sess.run(lambda_phi)
        pi_out = sess.run(lambda_pi)
        m_out = sess.run(lambda_m)
        nu_out = sess.run(lambda_nu)
        w_out = sess.run(lambda_w)
        beta_out = sess.run(lambda_beta)

        for i in range(int(maxIter * (N / BATCH_SIZE))):
            print(i, maxIter * (N / BATCH_SIZE))
            try:
                # Sample xn
                idx = np.random.randint(N, size=BATCH_SIZE)

                # Update local variational parameter lambda_phi
                new_lambda_phi = update_lambda_phi(
                    phi_out, pi_out, m_out, nu_out,
                    w_out, beta_out, xn, idx, K, D)
                sess.run(lambda_phi.assign(new_lambda_phi))

                # ELBO computation and global variational parameter updates
                _, lb, pi_out, phi_out, m_out, beta_out, nu_out, w_out =\
                    sess.run(
                        [train, LB, lambda_pi, lambda_phi, lambda_m,
                         lambda_beta, lambda_nu, lambda_w],
                        feed_dict={idx_tensor: idx})

                lb = lb * (N / BATCH_SIZE)
                aux_lbs.append(lb)
                if len(aux_lbs) == (N / BATCH_SIZE):
                    lbs.append(np.mean(aux_lbs))
                    n_iters += 1
                    aux_lbs = []
            except BaseException:
                break
            # Cov: w_out
            # Mean: m_out
            # pi: pi_out
            # phi: assignment
        covs = []
        for k in range(K):
            covs.append(np.linalg.cholesky(w_out[k, :, :]))
        zn = np.array([np.argmax(phi_out[n, :]) for n in range(N)])
        max_zn = np.array([np.amax(phi_out[n, :]) for n in range(N)])
        self.pi_out = pi_out
        self.mu_out = m_out
        self.sigma_out = np.stack(covs, axis=0)
        self.elbo = lbs
        self.assignment = zn
        self.assignment_likelihood = max_zn


class GMM_HMC(object):
    def __init__(
            self,
            observations,
            components=3,
            run_steps=1000,
            leap_steps=3,
            burn_in=100):
        # Create a mixture of two Gaussians:
        import tensorflow as tf
        import numpy as np
        import tensorflow_probability as tfp
        # from tensorflow_probability import edward2 as ed
        import functools

        tfd = tfp.distributions
        tfb = tfp.bijectors
        dtype = np.float32
        N, dims = observations.shape

        def joint_log_prob(observations, mix_probs, loc, chol_precision):
            rv_observations = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(probs=mix_probs),
                components_distribution=MVNCholPrecisionTriL(
                    loc=loc,
                    chol_precision_tril=chol_precision))
            log_prob_parts = [
                rv_observations.log_prob(observations),  # Sum over samples.
                pi.log_prob(mix_probs)[..., tf.newaxis],
                mu.log_prob(loc),  # Sum over components.
                sigma.log_prob(chol_precision),  # Sum over components.
            ]
            sum_log_prob = tf.reduce_sum(
                tf.concat(log_prob_parts, axis=-1), axis=-1)
            # Note: for easy debugging, uncomment the following:
            # sum_log_prob = tf.Print(sum_log_prob, log_prob_parts)
            return sum_log_prob

        pi = tfd.Dirichlet(
            concentration=np.ones(components, dtype) * 1e3,
            name='pi_dist')
        mu = tfd.Independent(
            tfd.Normal(
                loc=np.stack([np.zeros(dims, dtype)] * components),
                scale=tf.ones([components, dims], dtype)),
            reinterpreted_batch_ndims=1,
            name='mu_dist')

        sigma = tfd.Wishart(
            df=dims,
            scale_tril=np.stack([np.eye(dims, dtype=dtype)] * components),
            input_output_cholesky=True,
            name='sigma_dist')

        # testing
        unnormalized_posterior_log_prob = functools.partial(
            joint_log_prob, observations)
        initial_state = [
            tf.fill([components],
                    value=np.array(1. / components, dtype),
                    name='mix_probs'),
            tf.constant(np.stack([np.zeros(dims, dtype)] * components),
                        name='mu'),
            tf.eye(dims, batch_shape=[components],
                   dtype=dtype, name='chol_precision'),
        ]
        unconstraining_bijectors = [
            tfb.SoftmaxCentered(),
            tfb.Identity(),
            tfb.Chain([
                tfb.TransformDiagonal(tfb.Softplus()),
                tfb.FillTriangular(),
            ])]
        with tf.variable_scope("alice_world", reuse=tf.AUTO_REUSE):
            mcstep_size = tf.get_variable(
                name='step_size',
                initializer=np.array(0.05, dtype),
                use_resource=True,
                trainable=False)
            [mix_probs, loc, chol_precision],\
                kernel_results = tfp.mcmc.sample_chain(
                    num_results=run_steps,
                    num_burnin_steps=burn_in,
                    current_state=initial_state,
                    kernel=tfp.mcmc.TransformedTransitionKernel(
                        inner_kernel=tfp.mcmc.HamiltonianMonteCarlo(
                            target_log_prob_fn=unnormalized_posterior_log_prob,
                            step_size=mcstep_size,
                            step_size_update_fn=tfp.mcmc.
                            make_simple_step_size_update_policy(),
                            num_leapfrog_steps=leap_steps),
                        bijector=unconstraining_bijectors))

            acceptance_rate = tf.reduce_mean(tf.to_float(
                kernel_results.inner_results.is_accepted))
            mean_mix_probs = tf.reduce_mean(mix_probs, axis=0)
            mean_loc = tf.reduce_mean(loc, axis=0)
            mean_chol_precision = tf.reduce_mean(chol_precision, axis=0)

            obs = tf.convert_to_tensor(observations)
            c = tf.get_variable("c",
                                [len(observations)],
                                dtype=dtype,
                                initializer=tf.zeros_initializer)
            c_log = tf.get_variable("c_log",
                                    [len(observations)],
                                    dtype=dtype,
                                    initializer=tf.zeros_initializer)
            # tf.get_variable_scope().reuse_variables()

            def condition(i, px): return i < len(observations)

            def c_assign(i, c):
                theta = tfd.MultivariateNormalTriL(
                    loc=mean_loc, scale_tril=mean_chol_precision)
                c_log = tf.get_variable("c_log")
                c_log = tf.scatter_update(
                    c_log,
                    indices=i,
                    updates=tf.reduce_max(
                        tf.log(mean_mix_probs) +
                        theta.log_prob(
                            obs[i])))
                c = tf.get_variable("c")
                c = tf.scatter_update(
                    c,
                    indices=i,
                    updates=tf.to_float(
                        tf.argmax(
                            tf.log(mean_mix_probs) +
                            theta.log_prob(
                                obs[i]))))
                return i + 1, c

            i, c = tf.while_loop(condition, c_assign, [0, c])
        with tf.Session() as sess:
            init_op = tf.global_variables_initializer()
            sess.run(init_op)
            # sess.run(init_op)
            [
                self.acceptance_rate_,
                self.pi_out,
                self.mu_out,
                self.sigma_out,
                self.assignment,
                self.assignment_likelihood
            ] = sess.run([
                acceptance_rate,
                mean_mix_probs,
                mean_loc,
                mean_chol_precision,
                c,
                c_log
            ])


class dynamic_clustering_dynamic_context_bern_emb_model(emb_model):
    def __init__(self, args, d, logdir):
        import scipy
        super(
            dynamic_clustering_dynamic_context_bern_emb_model,
            self).__init__(
            args,
            d,
            logdir)

        def generate_random_positive_chol_matrix(D, K=1):
            chol_list = []
            for k in range(K):
                aux = scipy.random.rand(D, D)
                aux = np.dot(aux, aux.transpose())
                chol_list.append(np.linalg.cholesky(aux))
            return np.stack(chol_list)
        dims = self.K
        dtype = np.float32
        with tf.name_scope('model'):
            self.scale = tf.constant(
                generate_random_positive_chol_matrix(
                    self.K, self.components), dtype=dtype)
            self.mix_prob = tf.fill(
                [self.components], value=np.array(1. / self.components, dtype))
            self.mean = tf.constant(
                np.random.randn(
                    self.components,
                    self.K),
                dtype=dtype)
            self.alpha_t = {}
            for t in range(-1, self.T):
                self.alpha_t[t] = tf.Variable(
                    self.alpha_init +
                    0.001 * tf.random_normal([self.L, self.K]) /
                    self.K, name='alpha_' + str(t))

            pi = tfd.Dirichlet(
                concentration=np.ones(
                    self.components,
                    dtype) / self.components,
                name='pi_dist')

            mu = tfd.Independent(
                tfd.Normal(
                    loc=np.stack([np.zeros(dims, dtype)] * self.components),
                    scale=tf.ones([self.components, dims], dtype)),
                reinterpreted_batch_ndims=1,
                name='mu_dist')

            sigma = tfd.Wishart(
                df=dims,
                scale_tril=np.stack([np.eye(dims, dtype=dtype)] *
                                    self.components),
                input_output_cholesky=True,
                name='sigma_dist')
            mix = tfd.MixtureSameFamily(
                mixture_distribution=tfd.Categorical(
                    probs=tf.nn.softmax(
                        self.mix_prob)),
                components_distribution=MVNCholPrecisionTriL(
                    loc=self.mean, chol_precision_tril=self.scale))

            self.rho = tf.Variable(mix.sample(self.L), name='rho', dtype=dtype)

            def joint_log_prob(observations, mix_probs, loc, chol_precision):
                rv_observations = tfd.MixtureSameFamily(
                    mixture_distribution=tfd.Categorical(probs=mix_probs),
                    components_distribution=MVNCholPrecisionTriL(
                        loc=loc,
                        chol_precision_tril=chol_precision))
                log_prob_parts = [
                    # Sum over samples.
                    rv_observations.log_prob(observations),
                    pi.log_prob(mix_probs)[..., tf.newaxis],
                    mu.log_prob(loc),  # Sum over components.
                    sigma.log_prob(chol_precision),  # Sum over components.
                ]
                sum_log_prob = tf.reduce_sum(
                    tf.concat(log_prob_parts, axis=-1), axis=-1)
                return sum_log_prob
            with tf.name_scope('priors'):
                global_prior = Normal(loc=0.0, scale=self.sig)
                local_prior = Normal(loc=0.0, scale=self.sig / 100.0)

                self.log_prior = tf.reduce_sum(
                    global_prior.log_prob(self.alpha_t[-1]))
                for t in range(self.T):
                    self.log_prior += tf.reduce_sum(local_prior.log_prob(
                        self.alpha_t[t] - self.alpha_t[t - 1]))

        with tf.name_scope('likelihood'):
            self.placeholders = {}
            # self.testholders = {}
            self.y_pos = {}
            self.y_neg = {}
            self.ll_pos = 0.0
            self.ll_neg = 0.0
            for t in range(self.T):
                # Index Masks
                p_mask = tf.range(int(self.cs / 2),
                                  self.n_minibatch[t] + int(self.cs / 2))
                rows = tf.tile(tf.expand_dims(
                    tf.range(0, int(self.cs / 2)), [0]),
                               [self.n_minibatch[t], 1])
                columns = tf.tile(
                    tf.expand_dims(
                        tf.range(
                            0, self.n_minibatch[t]), [1]), [
                        1, int(
                            self.cs / 2)])

                ctx_mask = tf.concat(
                    [rows + columns, rows + columns + int(self.cs / 2) + 1], 1)

                # Data Placeholder
                # self.testholders[t] = tf.placeholder(tf.int32,
                # shape=(self.n_test[t] + self.cs),
                # name="testholder" + str(t))
                self.placeholders[t] = tf.placeholder(
                    tf.int32, shape=(self.n_minibatch[t] + self.cs))

                # Taget and Context Indices
                p_idx = tf.gather(self.placeholders[t], p_mask)
                # ctx_idx = tf.squeeze(tf.gather(self.placeholders[t], ctx_mask
                ctx_idx = tf.gather(self.placeholders[t], ctx_mask)

                # Negative samples
                unigram_logits = tf.tile(
                    tf.expand_dims(
                        tf.log(
                            tf.constant(
                                self.unigram)), [0]), [
                        self.n_minibatch[t], 1])
                n_idx = tf.multinomial(unigram_logits, self.ns)

                # Context vectors
                # ctx_alphas = tf.gather(self.alpha, ctx_idx)#######
                ctx_alphas = tf.squeeze(tf.gather(self.alpha_t[t], ctx_idx))

                p_rho = tf.squeeze(tf.gather(self.rho, p_idx))
                n_rho = tf.gather(self.rho, n_idx)

                # Natural parameter
                ctx_sum = tf.reduce_sum(ctx_alphas, [1])
                p_eta = tf.expand_dims(
                    tf.reduce_sum(
                        tf.multiply(
                            p_rho, ctx_sum), -1), 1)
                n_eta = tf.reduce_sum(
                    tf.multiply(
                        n_rho, tf.tile(
                            tf.expand_dims(
                                ctx_sum, 1), [
                                1, self.ns, 1])), -1)

                # Conditional likelihood
                self.y_pos[t] = Bernoulli(logits=p_eta)
                self.y_neg[t] = Bernoulli(logits=n_eta)

                self.ll_pos += tf.reduce_sum(self.y_pos[t].log_prob(1.0))
                self.ll_neg += tf.reduce_sum(self.y_neg[t].log_prob(0.0))
            self.log_prob = joint_log_prob(
                self.rho, tf.nn.softmax(
                    self.mix_prob), self.mean, self.scale)
        self.loss = - ((self.ll_pos + self.ll_neg +
                        self.log_prob) + self.log_prior)

    def initialize_training(self):
        optimizer = tf.train.AdamOptimizer()
        self.train = optimizer.minimize(self.loss)
        self.sess = tf.Session()
        with self.sess.as_default():
            tf.global_variables_initializer().run()

        for t in range(self.T):
            variable_summaries('alpha_' + str(t), self.alpha_t[t])
        with tf.name_scope('objective'):
            tf.summary.scalar('loss', self.loss)
            tf.summary.scalar('priors', self.log_prior)
            tf.summary.scalar('ll_pos', self.ll_pos)
            tf.summary.scalar('ll_neg', self.ll_neg)
        self.summaries = tf.summary.merge_all()
        self.train_writer = tf.summary.FileWriter(self.logdir, self.sess.graph)
        self.saver = tf.train.Saver()
        config = projector.ProjectorConfig()
        rho = config.embeddings.add()
        rho.tensor_name = 'model/embeddings/rho'
        rho.metadata_path = '../vocab.tsv'
        if not self.dynamic:
            alpha = config.embeddings.add()
            alpha.tensor_name = 'model/embeddings/alpha'
            alpha.metadata_path = '../vocab.tsv'
        else:
            for t in range(self.T):
                alpha = config.embeddings.add()
                alpha.tensor_name = 'model/embeddings/alpha_' + str(t)
                alpha.metadata_path = '../vocab.tsv'
        projector.visualize_embeddings(self.train_writer, config)

    def train_embeddings(self):
        import matplotlib.pyplot as plt
        train_loss_list = []
        log_loss_list = []

        for data_pass in range(self.n_iter):
            for step in range(self.n_epochs):
                print(step)
                if step % 100 == 0 and step > 0:
                    print(str(step) + '/' + str(self.n_epochs) +
                          '   iter ' + str(data_pass))
                    summary, _, train_loss, log_loss, rho = self.sess.run(
                        [self.summaries, self.train, self.loss,
                            self.log_prob, self.rho],
                        feed_dict=self.train_feed(self.placeholders))

                    print("GMM Start")
                    if self.VI:
                        self.GMM = GMM_SGAVI(
                            xn=rho, maxIter=5, K=self.components)
                    elif self.HMC:
                        self.GMM = GMM_HMC(
                            observations=rho, components=self.components)
                    print("GMM Over")
                    self.scale = tf.convert_to_tensor(self.GMM.sigma_out)
                    self.mean = tf.convert_to_tensor(self.GMM.mu_out)
                    self.mix_prob = tf.convert_to_tensor(self.GMM.pi_out)
                    self.train_writer.add_summary(
                        summary, data_pass * (self.n_epochs) + step)
                else:
                    _, train_loss, log_loss = self.sess.run(
                        [self.train, self.loss, self.log_prob],
                        feed_dict=self.train_feed(self.placeholders))
                train_loss_list.append(train_loss)
                log_loss_list.append(log_loss)
            self.dump(self.logdir + "/variational" + str(data_pass) + ".dat")
            self.saver.save(
                self.sess,
                os.path.join(
                    self.logdir,
                    "model.ckpt"),
                data_pass)
            self.print_topics()
        self.print_word_similarities_alpha(self.query_words, 10)
        if self.dynamic:
            words = self.detect_drift()
            self.print_word_similarities_alpha(words[:10], 10)
        title = "Training Loss over the process"
        plt.figure()
        plt.plot(train_loss_list, 'black', label='loss')
        plt.xlabel("N iteration")
        plt.ylabel("Training Loss")
        plt.title(title)
        # save image
        plt.savefig(self.logdir + "/" + title + ".png")
        plt.figure()
        title = "Log mixture prob over the process"
        plt.plot(log_loss_list, 'black', label='loss')
        plt.xlabel("N iteration")
        plt.ylabel("Log mixture prob")
        plt.title(title)
        plt.savefig(self.logdir + "/" + title + ".png")
        self.plot_params(500)
        # print(self.eval_log_like(feed_dict=self.test_feed(self.testholders)))

    def dump(self, fname):
        with self.sess.as_default():
            dat = {'rho': self.rho.eval()}
            for t in range(self.T):
                dat['alpha_' + str(t)] = self.alpha_t[t].eval()
            dat['mean'] = self.mean.eval()
            dat['scale'] = self.scale.eval()
            dat['pi'] = self.mix_prob.eval()

        pickle.dump(dat, open(fname, "wb"))

    def eval_log_like(self, feed_dict):
        with self.sess.as_default():
            log_p = np.zeros((0, 1))
            for t in range(self.T):
                log_p_t = self.sess.run(
                    tf.log(
                        self.y_pos[t].mean() +
                        0.000001),
                    feed_dict=feed_dict)
                log_p = np.vstack((log_p, log_p_t))
            return log_p

    def plot_params(self, plot_only=500):
        with self.sess.as_default():
            tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000)
            low_dim_embs_rho = tsne.fit_transform(self.rho.eval()[:plot_only])
            plot_with_labels(
                low_dim_embs_rho[:plot_only],
                self.labels[:plot_only],
                self.logdir + '/rho.eps')
            for t in [0, int(self.T / 2), self.T - 1]:
                w_idx_t = range(plot_only)
                np_alpha = self.alpha_t[t].eval()
                tsne = TSNE(
                    perplexity=30,
                    n_components=2,
                    init='pca',
                    n_iter=5000)
                low_dim_embs_alpha = tsne.fit_transform(np_alpha[w_idx_t, :])
                plot_with_labels(
                    low_dim_embs_alpha,
                    self.labels[w_idx_t],
                    self.logdir +
                    '/alpha_' +
                    str(t) +
                    '.eps')

    def detect_drift(self, metric='total_dist'):
        if metric == 'total_dist':
            tf_dist, tf_w_idx = tf.nn.top_k(tf.reduce_sum(
                tf.square(self.alpha_t[self.T - 1] - self.alpha_t[0]), 1), 500)
        else:
            print('unknown metric')
            return
        dist, w_idx = self.sess.run([tf_dist, tf_w_idx])
        words = self.labels[w_idx]
        f_name = self.logdir + '/top_drifting_words.txt'
        with open(f_name, "w+") as text_file:
            for (w, drift) in zip(w_idx, dist):
                text_file.write("\n%-20s %6.4f" % (self.labels[w], drift))
        return words

    def print_word_similarities_alpha(self, words, num):
        query_word = tf.placeholder(dtype=tf.int32)
        query_alpha_t = tf.placeholder(dtype=tf.float32)

        val_rho, idx_rho = tf.nn.top_k(
            tf.matmul(
                tf.nn.l2_normalize(
                    self.rho, dim=0), tf.nn.l2_normalize(
                    query_alpha_t, dim=1), transpose_b=True), num)

        for x in words:
            f_name = os.path.join(self.logdir, '%s_queries_alpha.txt' % (x))
            with open(f_name, "w+") as text_file:
                for t_idx in range(self.T):
                    with self.sess.as_default():
                        alpha_t = self.alpha_t[t_idx].eval()
                    vr, ir = self.sess.run([val_rho, idx_rho], {
                                           query_word: self.dictionary[x],
                                           query_alpha_t: alpha_t})
                    text_file.write(
                        "\n\n====================================\
                        =\n%s, t = %d\n=====================================" %
                        (x, t_idx))
                    for ii in range(num):
                        text_file.write("\n%-20s %6.4f" %
                                        (self.labels[ir[0, ii]], vr[0, ii]))

    def print_topics(self):
        with self.sess.as_default():
            ass = self.GMM.assignment
            ass_likelihood = self.GMM.assignment_likelihood
            c_ = ass.tolist()
            cl_ = ass_likelihood.tolist()
            c_ = [x for _, x in sorted(zip(cl_, c_), reverse=True)]
            for i in range(self.components):
                word_list = []
                for j in range(self.L):
                    if c_[j] == i:
                        word_list.append(self.labels[j])
                file_name = self.logdir + "/topic_" + str(i) + ".txt"
                with open(file_name, 'w') as f:
                    for item in word_list:
                        f.write("%s\n" % item)


def define_model(args, d, logdir):
    def session_options(enable_gpu_ram_resizing=True):
        """Convenience function which sets common `tf.Session` options."""
        config = tf.ConfigProto()
        config.log_device_placement = True
        if enable_gpu_ram_resizing:
            # `allow_growth=True` makes it possible to
            # connect multiple colabs to your
            # GPU. Otherwise the colab malloc's all GPU ram.
            config.gpu_options.allow_growth = True
        return config

    def reset_sess(config=None):
        # Convenience function to create the
        # TF graph and session, or reset them.
        if config is None:
            config = session_options()
        tf.reset_default_graph()
        global sess
        try:
            sess.close()
        except BaseException:
            pass
        sess = tf.InteractiveSession(config=config)

    reset_sess()

    if args.dynamic:
        if args.dclustering:
            m = dynamic_clustering_dynamic_context_bern_emb_model(
                args, d, logdir)
            return m
        m = dynamic_bern_emb_model(args, d, logdir)
    else:
        m = bern_emb_model(args, d, logdir)
    return m
