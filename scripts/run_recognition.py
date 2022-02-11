import torch
import argparse
import sys
import pickle
import os
import csv
import tqdm
import h5py
import PIL
import numpy as np
from moviepy.editor import *

import slowfast.utils.checkpoint as cu
import slowfast.utils.multiprocessing as mpu
from slowfast.config.defaults import get_cfg
import slowfast.utils.distributed as du
import slowfast.utils.logging as logging
import slowfast.utils.misc as misc
from slowfast.datasets import loader
from slowfast.models import build_model
from slowfast.utils.meters import AVAMeter, TestMeter, EPICTestMeter

from core.utils.vis_utils import add_caption_to_img, init_wandb, dump_video_wandb


TARGET_DATA_DIR = "/private/home/kpertsch/data/human_kitchen_sub1_224/MW_BB_TB_SC"
VERB_TAXONOMY = "/private/home/kpertsch/code/epic-kitchens-100-annotations/EPIC_100_verb_classes.csv"
NOUN_TAXONOMY = "/private/home/kpertsch/code/epic-kitchens-100-annotations/EPIC_100_noun_classes.csv"
N_SEQS = 3
TAG = 'vis_EPIC'
VIS_TOP_N = 5       # top N skill predictions to visualize
FILE_ENDING = ".h5"
RESIZE_DIM = 256

NORM_MEAN = 0.45
NORM_STD = 0.225


def parse_args():
    """
    Parse the following arguments for the video training and testing pipeline.
    Args:
        shard_id (int): shard id for the current machine. Starts from 0 to
            num_shards - 1. If single machine is used, then set shard id to 0.
        num_shards (int): number of shards using by the job.
        init_method (str): initialization method to launch the job with multiple
            devices. Options includes TCP or shared file-system for
            initialization. details can be find in
            https://pytorch.org/docs/stable/distributed.html#tcp-initialization
        cfg (str): path to the config file.
        opts (argument): provide addtional options from the command line, it
            overwrites the config loaded from file.
        """
    parser = argparse.ArgumentParser(
        description="Provide SlowFast video training and testing pipeline."
    )
    parser.add_argument(
        "--shard_id",
        help="The shard id of current node, Starts from 0 to num_shards - 1",
        default=0,
        type=int,
    )
    parser.add_argument(
        "--num_shards",
        help="Number of shards using by the job",
        default=1,
        type=int,
    )
    parser.add_argument(
        "--init_method",
        help="Initialization method, includes TCP or shared file-system",
        default="tcp://localhost:9999",
        type=str,
    )
    parser.add_argument(
        "--cfg",
        dest="cfg_file",
        help="Path to the config file",
        default="configs/Kinetics/SLOWFAST_4x16_R50.yaml",
        type=str,
    )
    parser.add_argument(
        "opts",
        help="See slowfast/config/defaults.py for all options",
        default=None,
        nargs=argparse.REMAINDER,
    )
    if len(sys.argv) == 1:
        parser.print_help()
    return parser.parse_args()


def load_config(args):
    """
    Given the arguemnts, load and initialize the configs.
    Args:
        args (argument): arguments includes `shard_id`, `num_shards`,
            `init_method`, `cfg_file`, and `opts`.
    """
    # Setup cfg.
    cfg = get_cfg()
    # Load config from cfg.
    if args.cfg_file is not None:
        cfg.merge_from_file(args.cfg_file)
    # Load config from command line, overwrite config from opts.
    if args.opts is not None:
        cfg.merge_from_list(args.opts)

    # Inherit parameters from args.
    if hasattr(args, "num_shards") and hasattr(args, "shard_id"):
        cfg.NUM_SHARDS = args.num_shards
        cfg.SHARD_ID = args.shard_id
    if hasattr(args, "rng_seed"):
        cfg.RNG_SEED = args.rng_seed
    if hasattr(args, "output_dir"):
        cfg.OUTPUT_DIR = args.output_dir

    # Create the checkpoint dir.
    cu.make_checkpoint_dir(cfg.OUTPUT_DIR)
    return cfg


def read_taxonomy(filepath):
    with open(filepath) as fp:
        reader = csv.reader(fp, delimiter=",", quotechar='"')
        next(reader, None)  # skip the headers
        data_read = [row[1] for row in reader]
    return data_read


def get_filenames():
    print("Collect filenames...")
    filenames = []
    for root, dirs, files in tqdm.tqdm(os.walk(TARGET_DATA_DIR)):
        for file in files:
            if file.endswith(FILE_ENDING):
                filename = os.path.join(root, file)
                filenames.append(filename)
    n_files = len(filenames)
    print("\nDone! Found {} files!".format(n_files))
    return filenames


def get_frames(filename):
    if FILE_ENDING == ".h5":
        with h5py.File(filename, 'r') as F:
            images = F['traj0/images'][()]
    elif FILE_ENDING == ".mp4":
        images = []
        clip = VideoFileClip(filename)
        print("Reading {} frames...".format(clip.reader.nframes))
        for frame in tqdm.tqdm(clip.iter_frames()):
            frame = np.array(PIL.Image.fromarray(frame).resize((RESIZE_DIM, RESIZE_DIM)))
            images.append(frame)
        images = np.stack(images)
    else:
        raise NotImplementedError
    return images


def main():
    args = parse_args()
    cfg = load_config(args)
    np.random.seed(cfg.RNG_SEED)
    torch.manual_seed(cfg.RNG_SEED)

    # Build the video model and print model statistics.
    model = build_model(cfg)
    if du.is_master_proc():
        misc.log_model_info(model, cfg, is_train=False)

    # Load a checkpoint to test if applicable.
    if cfg.TEST.CHECKPOINT_FILE_PATH != "":
        cu.load_checkpoint(
            cfg.TEST.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TEST.CHECKPOINT_TYPE == "caffe2",
        )
    elif cu.has_checkpoint(cfg.OUTPUT_DIR):
        last_checkpoint = cu.get_last_checkpoint(cfg.OUTPUT_DIR)
        cu.load_checkpoint(last_checkpoint, model, cfg.NUM_GPUS > 1)
    elif cfg.TRAIN.CHECKPOINT_FILE_PATH != "":
        # If no checkpoint found in TEST.CHECKPOINT_FILE_PATH or in the current
        # checkpoint folder, try to load checkpint from
        # TRAIN.CHECKPOINT_FILE_PATH and test it.
        cu.load_checkpoint(
            cfg.TRAIN.CHECKPOINT_FILE_PATH,
            model,
            cfg.NUM_GPUS > 1,
            None,
            inflation=False,
            convert_from_caffe2=cfg.TRAIN.CHECKPOINT_TYPE == "caffe2",
        )
    else:
        raise NotImplementedError("Unknown way to load checkpoint.")

    # Enable eval mode.
    model.eval()

    # load data filenames
    filenames = get_filenames()

    # load taxonomy
    verb_taxonomy = read_taxonomy(VERB_TAXONOMY)
    noun_taxonomy = read_taxonomy(NOUN_TAXONOMY)
    print("#Verbs: {}, #Nouns: {}".format(len(verb_taxonomy), len(noun_taxonomy)))

    # initialize wandb
    init_wandb('clvr', 'comp_imitation', tag=TAG)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # run inference for target data sequences
    for filename in filenames[:N_SEQS]:
        images = get_frames(filename)

        # sliding inference window over sequence
        seq_len = images.shape[0]
        window_size = cfg.DATA.NUM_FRAMES
        offset = int(window_size / 2)
        frames, skills = [], []
        for i in range(offset, seq_len - offset):
            subseq = images[i - offset: i + offset]
            subseq = subseq[None].transpose(0, 4, 1, 2, 3)

            # normalize
            subseq = subseq / 255.
            subseq = (subseq - NORM_MEAN) / NORM_STD

            # create subsampled version for fast-slow architecture
            subseq_slow = subseq[:, :, np.arange(0, window_size, step=cfg.SLOWFAST.ALPHA)]

            outputs = model([torch.tensor(subseq_slow, device=device).float(),
                             torch.tensor(subseq, device=device).float()])

            # extract skill string
            top_n_verbs = torch.topk(outputs[0][0], VIS_TOP_N).indices
            top_n_nouns = torch.topk(outputs[1][0], VIS_TOP_N).indices
            skill = []
            for top_n_verb_idx, top_n_noun_idx in zip(top_n_verbs, top_n_nouns):
                skill.append(verb_taxonomy[top_n_verb_idx.data.cpu().numpy()] \
                        + ' ' + noun_taxonomy[top_n_noun_idx.data.cpu().numpy()])

            # log frame and skill for later visualization
            frames.append(images[i])
            skills.append(skill)

        # visualize predicted skills, dump to wandb
        dump_video_wandb(np.stack([add_caption_to_img(img,
                        info={"p_{}".format(k+1): s for k, s in enumerate(skill)}) for (img, skill) in zip(frames, skills)])
                         .transpose(0, 3, 1, 2), tag="skill_vis", fps=10)


if __name__ == "__main__":
    main()
