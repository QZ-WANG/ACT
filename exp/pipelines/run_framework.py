import argparse
import os
from os.path import join
from utils.data import load_yaml
from datetime import datetime
from random import seed, randint


def do_exp():
    # Get timestamp
    print("="*35 + "A NEW RUN" + "="*35)
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d-%H-%M-%S")
    print("Pipeline timestamp: %s" % timestamp)

    src_ckpt_dir = join(args.shared_dir, "ckpt", "src_gdev", args.src_dset_name+"_"+timestamp)

    dev_str = ""
    if args.device is not None:
        dev_str = " --device %d" % args.device

    # train source model
    print("Source training...")
    os.system("python -u %s --config_fn %s --timestamp %s  --ckpt_dir %s --seed %d  --proj_dir %s%s" %
              (args.src_script, args.src_config, timestamp, src_ckpt_dir, randint(0, 2 ** 32),
               args.proj_dir, dev_str))

    # adaptation
    print("\nAdapting...")
    adapt_task_name = "%s_to_%s" % (args.src_dset_name, args.tar_dset_name)
    tar_ckpt_dir = join(args.shared_dir, "ckpt", "act", adapt_task_name + "_" + timestamp)

    os.system("python -u %s --config_fn %s --timestamp %s  --ckpt_dir %s --src_ckpt_dir %s --seed %d --proj_dir %s%s" %
              (args.adapt_script, args.adapt_config, timestamp, tar_ckpt_dir, src_ckpt_dir,
               randint(0, 2 ** 16), args.proj_dir, dev_str))

    print("\nPseudo Labeling...")

    # Pseudo Labelling
    # with Isolation Forest
    ckpts = "%s" % tar_ckpt_dir
    os.system("python -u %s --config_fn %s --ckpts %s --seed %d  --proj_dir %s  --which_label iForest --src_ckpt_dir %s%s" %
              (args.pl_script, args.pl_config, ckpts, randint(0, 2 ** 32), args.proj_dir, src_ckpt_dir,
               dev_str))

    print("=" * 35 + "RUN ENDED" + "=" * 35 + "\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--proj_dir", type=str, default="/media/nvme1/pycharm_mirror/GUDA_release")
    parser.add_argument("--config_fn", type=str, default="htl_to_res_rdc_0")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=int, default=None)
    parser.add_argument("--batch_id", type=str, default=None)
    args = parser.parse_args()
    print(os.getcwd())

    seed(args.seed)

    if args.config_fn is not None:
        config = load_yaml(join(args.proj_dir, "exp", "pipelines/config", args.config_fn + ".yaml"))
        args = argparse.Namespace(**{**vars(args), **config})
    else:
        raise RuntimeError("Config file not found!")

    do_exp()
