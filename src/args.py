import argparse


def parse_args():
        parser = argparse.ArgumentParser(
            description="run exponential family embeddings on text")

        parser.add_argument('--K', type=int, default=100,
                            help='Number of dimensions. Default is 100.')

        parser.add_argument(
            '--sig', type=float, default=1.0,
            help='Noise on random walk for dynamic model.Default is 1.')

        parser.add_argument(
            '--n_iter', type=int, default=1,
            help='Number of passes over the data. Default is 1.')

        parser.add_argument('--n_epochs', type=int, default=10000,
                            help='Number of epochs. Default is 10000.')

        parser.add_argument('--cs', type=int, default=4,
                            help='Context size. Default is 4.')

        parser.add_argument('--ns', type=int, default=10,
                            help='Number of negative samples. Default is 10.')

        parser.add_argument('--dynamic', type=bool, default=False,
                            help='Dynamics on rho. Default is False.')

        parser.add_argument('--dclustering', type=bool, default=False,
                            help='Clustering on rho. Default is False.')

        parser.add_argument('--VI', type=bool, default=False,
                            help='Use VI for GMM. Default is False.')

        parser.add_argument('--HMC', type=bool, default=True,
                            help='Use HMC for GMM. Default is True.')

        parser.add_argument('--components', type=int, default=3,
                            help='Number of components. Default is 3.')

        parser.add_argument('--run_steps', type=int, default=1000,
                            help='Number of steps for HMC. Default is 1000.')

        parser.add_argument('--burnin_steps', type=int, default=100,
                            help='Number of components. Default is 3.')

        parser.add_argument(
            '--init', type=str, default='',
            help='Folder name to load variational.dat for initialization. \
            Default is \'\' for no initialization')

        parser.add_argument('--fpath', type=str, default='../dat/arxiv_ML/',
                            help='path to data')

        args = parser.parse_args()
        return args
