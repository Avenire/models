import os
import numpy as np
import glob
import argparse
parser = argparse.ArgumentParser(description='Copy latest checkpoint to train dir')
parser.add_argument('--train_dir', type=str, help='Train dir')
parser.add_argument('--purge_stale_thresh', type=int, default=0, help='How many stale checkpoints to purge')
def get_step(x):
    return int(x.split('_')[-1])

def check_ckpt(x, step=None):
    if step is None:
      step = get_step(x)
    required_files = ['%s/model.ckpt-%d.data-00000-of-00001'%(x,step), '%s/model.ckpt-%d.index' % (x, step), '%s/model.ckpt-%d.meta' % (x, step), '%s/checkpoint' % x]
    return np.all(np.array([os.path.isfile(r) and os.path.getsize(r) > 0 for r in required_files]))


def main():
    args = parser.parse_args()
    train_dir=os.path.abspath(os.path.expanduser(args.train_dir))
    checkpoints = glob.glob('%s/checkpoint_*'%train_dir)
    checkpoints = sorted(checkpoints, key=lambda x:get_step(x))
    checkpoints = [c for c in checkpoints if check_ckpt(c)]
    if len(checkpoints) == 0:
        print("No checkpoints...")
        return
    latest_checkpoint = checkpoints[-1]
    step = get_step(latest_checkpoint)
    print('Latest checkpoint located at %s' % latest_checkpoint)
    os.system('echo %s > %s/latest_ckpt'%(step, train_dir))
    if not check_ckpt(train_dir, step):
        os.system('rm -rf %s/model.ckpt-*' % train_dir)
        os.system('rm -rf %s/checkpoint' % train_dir)
        os.system('cp %s/checkpoint %s' % (latest_checkpoint, train_dir))
        os.system('cp %s/model.ckpt-* %s' % (latest_checkpoint, train_dir))
    else:
        print("No need to copy checkpoints.")
    to_remove_thresh = args.purge_stale_thresh
    if 0 < to_remove_thresh < len(checkpoints):
        to_remove_checkpoints = checkpoints[:-to_remove_thresh]
        for ckpt in to_remove_checkpoints:
            print("Purging checkpoint %s" % ckpt)
            os.system('rm -rf %s' % (ckpt))
    else:
        print("No purgin now")

if __name__ == "__main__":
    main()
