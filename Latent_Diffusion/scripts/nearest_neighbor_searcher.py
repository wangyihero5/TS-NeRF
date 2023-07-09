import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import argparse, os, sys, glob
import torch
import numpy as np
from tqdm import tqdm, trange
from itertools import islice
import scann
import time
from multiprocessing import cpu_count

from Latent_Diffusion.ldm.util import instantiate_from_config, parallel_data_prefetch
from Latent_Diffusion.ldm.modules.encoders.modules import FrozenClipImageEmbedder, FrozenCLIPTextEmbedder

device = "cuda" if torch.cuda.is_available() else "cpu"

DATABASES = [
    "openimages",
    "artbench-art_nouveau",
    "artbench-baroque",
    "artbench-expressionism",
    "artbench-impressionism",
    "artbench-post_impressionism",
    "artbench-realism",
    "artbench-romanticism",
    "artbench-renaissance",
    "artbench-surrealism",
    "artbench-ukiyo_e",
]


def chunk(it, size):
    it = iter(it)
    return iter(lambda: tuple(islice(it, size)), ())


def load_model_from_config(config, ckpt, verbose=False):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt, map_location="cpu")
    if "global_step" in pl_sd:
        print(f"Global Step: {pl_sd['global_step']}")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print("missing keys:")
        print(m)
    if len(u) > 0 and verbose:
        print("unexpected keys:")
        print(u)

    model.cuda()
    model.eval()
    return model


class Searcher(object):
    def __init__(self, database, retriever_version='ViT-L/14'):  # ViT-L/14 ViT-L/14
        # assert database in DATABASES
        # self.database = self.load_database(database)
        self.database_name = database
        self.searcher_savedir = f'Latent_Diffusion/data/rdm/searchers/{self.database_name}'
        self.database_path = f'Latent_Diffusion/data/rdm/retrieval_databases/{self.database_name}'
        self.retriever = self.load_retriever(version=retriever_version)
        self.database = {'embedding': [],
                         'img_id': [],
                         'patch_coords': []}
        self.load_database()
        self.load_searcher()

    def train_searcher(self, k,
                       metric='dot_product',
                       searcher_savedir=None):

        print('Start training searcher')
        searcher = scann.scann_ops_pybind.builder(self.database['embedding'] /
                                                  np.linalg.norm(self.database['embedding'], axis=1)[:, np.newaxis],
                                                  k, metric)
        self.searcher = searcher.score_brute_force().build()
        print('Finish training searcher')

        if searcher_savedir is not None:
            print(f'Save trained searcher under "{searcher_savedir}"')
            os.makedirs(searcher_savedir, exist_ok=True)
            self.searcher.serialize(searcher_savedir)

    def load_single_file(self, saved_embeddings):
        compressed = np.load(saved_embeddings)
        self.database = {key: compressed[key] for key in compressed.files}
        print('Finished loading of clip embeddings.')

    def load_multi_files(self, data_archive):
        out_data = {key: [] for key in self.database}
        for d in tqdm(data_archive, desc=f'Loading datapool from {len(data_archive)} individual files.'):
            for key in d.files:
                out_data[key].append(d[key])

        return out_data

    def load_database(self):

        print(f'Load saved patch embedding from "{self.database_path}"')
        file_content = glob.glob(os.path.join(self.database_path, '*.npz'))

        if len(file_content) == 1:
            self.load_single_file(file_content[0])
        elif len(file_content) > 1:
            data = [np.load(f) for f in file_content]
            prefetched_data = parallel_data_prefetch(self.load_multi_files, data,
                                                     n_proc=min(len(data), cpu_count()), target_data_type='dict')

            self.database = {key: np.concatenate([od[key] for od in prefetched_data], axis=1)[0] for key in
                             self.database}
        else:
            raise ValueError(f'No npz-files in specified path "{self.database_path}" is this directory existing?')

        print(f'Finished loading of retrieval database of length {self.database["embedding"].shape[0]}.')

    def load_retriever(self, version='ViT-L/14', ):
        model = FrozenClipImageEmbedder(model=version)
        if torch.cuda.is_available():
            model.cuda()
        model.eval()
        return model

    def load_searcher(self):
        print(f'load searcher for database {self.database_name} from {self.searcher_savedir}')
        self.searcher = scann.scann_ops_pybind.load_searcher(self.searcher_savedir)
        print('Finished loading searcher.')

    def search(self, x, k):
        if self.searcher is None and self.database['embedding'].shape[0] < 2e4:
            self.train_searcher(k)  # quickly fit searcher on the fly for small databases
        assert self.searcher is not None, 'Cannot search with uninitialized searcher'
        if isinstance(x, torch.Tensor):
            x = x.detach().cpu().numpy()
        if len(x.shape) == 3:
            x = x[:, 0]
        query_embeddings = x / np.linalg.norm(x, axis=1)[:, np.newaxis]

        start = time.time()
        nns, distances = self.searcher.search_batched(query_embeddings, final_num_neighbors=k)
        end = time.time()

        out_embeddings = self.database['embedding'][nns]
        out_img_ids = self.database['img_id'][nns]
        out_pc = self.database['patch_coords'][nns]

        out = {'nn_embeddings': out_embeddings / np.linalg.norm(out_embeddings, axis=-1)[..., np.newaxis],
               'img_ids': out_img_ids,
               'patch_coords': out_pc,
               'queries': x,
               'exec_time': end - start,
               'nns': nns,
               'q_embeddings': query_embeddings}

        return out

    def __call__(self, x, n):
        return self.search(x, n)


# configuration files
def readConfig():
    config = type('Dummy', (object,), {})
    config.prompt = "a painting of monet's san giorgio maggiore at dusk"
    config.outdir = "outputs/txt2img-samples"
    config.skip_grid = True
    config.ddim_steps = 50
    config.n_repeat = 1
    config.plms = True
    config.ddim_eta = 0
    config.n_iter = 1
    config.H = 768
    config.W = 768
    config.n_samples = 1
    config.n_rows = 0
    config.scale = 2.0
    config.from_file = None
    config.config = "Latent_Diffusion/configs/retrieval-augmented-diffusion/768x768.yaml"
    config.ckpt = "Latent_Diffusion/models/rdm/rdm768x768/model.ckpt"
    config.clip_type = 'ViT-L/14'
    config.database = "artbench-impressionism"
    config.use_neighbors = True
    config.knn = 1  # neighbor number
    return config


def get_nearest_neighbor_vector(text_target):
    opt = readConfig()
    searcher = None
    if opt.use_neighbors:
        searcher = Searcher(opt.database)

    with torch.no_grad():
        if searcher is not None:
            opt.knn = 3  # The number of nearest neighbor vectors, up to 10
            nn_dict = searcher(text_target, opt.knn)
            text_target = torch.cat([text_target.unsqueeze(1), torch.from_numpy(nn_dict['nn_embeddings']).cuda()], dim=1).detach()

    return text_target 
