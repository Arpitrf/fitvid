import torch
from torch import nn as nn
import torch.nn.functional as F
from absl import flags
from absl import app
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import glob
import json
from copy import deepcopy
from tqdm import tqdm
import moviepy
import wandb

from fitvid.data.robomimic_data import load_dataset_robomimic_torch
from fitvid.data.hdf5_data_loader import load_hdf5_data
from fitvid.model.fitvid import FitVid
from fitvid.utils.utils import dict_to_cuda
from fitvid.utils.vis_utils import save_moviepy_gif, build_visualization

FLAGS = flags.FLAGS

# Model training
flags.DEFINE_boolean('debug', False, 'Debug mode.')  # changed
flags.DEFINE_integer('batch_size', 32, 'Batch size.')  # changed
flags.DEFINE_integer('n_past', 2, 'Number of past frames.')
flags.DEFINE_integer('n_future', 10, 'Number of future frames.')  # not used, inferred directly from data
flags.DEFINE_integer('num_epochs', 1000, 'Number of steps to train for.')
flags.DEFINE_float('beta', 1e-4, 'Weight on KL.')
flags.DEFINE_string('data_in_gpu', 'True', 'whether to put data in GPU, or RAM')
flags.DEFINE_string('loss', 'l2', 'whether to use l2 or l1 loss')
flags.DEFINE_float('lr', 1e-3, 'learning rate')
flags.DEFINE_float('weight_decay', 0.0, 'weight decay value')
flags.DEFINE_boolean('stochastic', True, 'Use a stochastic model.')
flags.DEFINE_boolean('multistep', False, 'Multi-step training.')  # changed
flags.DEFINE_boolean('fp16', False, 'Use lower precision training for perf improvement.')  # changed

# Model architecture
flags.DEFINE_integer('z_dim', 10, 'LSTM output size.')  #
flags.DEFINE_integer('rnn_size', 256, 'LSTM hidden size.')  #
flags.DEFINE_integer('g_dim', 128, 'CNN feature size / LSTM input size.')  #
flags.DEFINE_list('stage_sizes', [1, 1, 1, 1], 'Number of layers for encoder/decoder.')  #
flags.DEFINE_list('first_block_shape', [8, 8, 512], 'Decoder first conv size')  #
flags.DEFINE_string('action_conditioned', 'True', 'Action conditioning.')
flags.DEFINE_integer('num_base_filters', 32, 'num_filters = num_base_filters * 2^layer_index.')  # 64
flags.DEFINE_integer('expand', 1, 'multiplier on decoder\'s num_filters.')  # 4
flags.DEFINE_integer('action_size', 4, 'number of actions')  # 4
flags.DEFINE_string('skip_type', 'residual', 'skip type: residual, concat, no_skip, pixel_residual')  # residual

# Model saving
flags.DEFINE_string('output_dir', None, 'Path to model checkpoints/summaries.')
flags.DEFINE_integer('save_freq', 10, 'number of steps between checkpoints')
flags.DEFINE_boolean('wandb_online', None, 'Use wandb online mode (probably should disable on cluster)')
flags.DEFINE_string('project', 'perceptual-metrics', 'wandb project name')

# Data
flags.DEFINE_spaceseplist('dataset_file', [], 'Dataset to load.')
flags.DEFINE_boolean('hdf5_data', None, 'using hdf5 data')
flags.DEFINE_string('cache_mode', 'low_dim', 'Dataset cache mode')
flags.DEFINE_string('camera_view', 'agentview', 'Camera view of data to load. Default is "agentview".')
flags.DEFINE_list('image_size', [], 'H, W of images')
flags.DEFINE_boolean('has_segmentation', True, 'Does dataset have segmentation masks')

# depth objective
flags.DEFINE_boolean('only_depth', False, 'Depth to depth prediction model.')
flags.DEFINE_float('depth_weight', 0, 'Weight on depth objective.')
flags.DEFINE_string('depth_model_path', None, 'Path to load pretrained depth model from.')
flags.DEFINE_integer('depth_model_size', 256, 'Depth model size.')
flags.DEFINE_integer('depth_start_epoch', 2, 'When to start training with depth objective.')

# normal model
flags.DEFINE_float('normal_weight', 0, 'Weight on normal objective.')
flags.DEFINE_string('normal_model_path', None, 'Path to load pretrained depth model from.')

# Policy networks
flags.DEFINE_float('policy_weight', 0, 'Weight on policy loss.')
flags.DEFINE_spaceseplist('policy_network_paths', [], 'Policy feature network locations.')
flags.DEFINE_string('policy_network_layer', 'fc0', 'Layer to use for policy feature network.')

# post hoc analysis
flags.DEFINE_string('re_eval', 'False', 'Re evaluate all available checkpoints saved.')
flags.DEFINE_integer('re_eval_bs', 20, 'bs override')


def load_data(dataset_files, data_type='train', depth=False, normal=False, seg=True):
    video_len = FLAGS.n_past + FLAGS.n_future
    if FLAGS.image_size:
        assert len(FLAGS.image_size) == 2, "Image size should be (H, W)"
        image_size = [int(i) for i in FLAGS.image_size]
    else:
        image_size = None
    return load_dataset_robomimic_torch(dataset_files, FLAGS.batch_size, video_len, image_size,
                                        data_type, depth, normal, view=FLAGS.camera_view, cache_mode=FLAGS.cache_mode,
                                        seg=seg, only_depth=False)


def get_most_recent_checkpoint(dir):
    if os.path.isdir(dir):
        pass
    else:
        os.mkdir(dir)
    existing_checkpoints = glob.glob(os.path.join(dir, 'model_epoch*'))
    if existing_checkpoints:
        checkpoint_nums = [int(s.split('/')[-1][len('model_epoch'):]) for s in existing_checkpoints]
        best_ind = checkpoint_nums.index(max(checkpoint_nums))
        return existing_checkpoints[best_ind], max(checkpoint_nums)
    else:
        return None, -1


def main(argv):
    import random
    random.seed(0)
    torch.manual_seed(0)
    np.random.seed(0)
    os.environ['PYTHONHASHSEED'] = str(0)

    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True

    if FLAGS.depth_model_path:
        depth_predictor_kwargs = {
            'depth_model_type': 'mns',
            'pretrained_weight_path': os.path.abspath(FLAGS.depth_model_path),
            'input_size': FLAGS.depth_model_size,
        }
    else:
        depth_predictor_kwargs = None

    if FLAGS.normal_model_path:
        normal_predictor_kwargs = {
            'pretrained_weight_path': os.path.abspath(FLAGS.normal_model_path),
        }
    else:
        normal_predictor_kwargs = None

    if FLAGS.policy_network_paths:
        policy_network_kwargs = {
            'pretrained_weight_paths': [os.path.abspath(p) for p in FLAGS.policy_network_paths],
            'layer': FLAGS.policy_network_layer
        }
    else:
        policy_network_kwargs = None

    other_weights = FLAGS.depth_weight + FLAGS.normal_weight
    depth_weight = FLAGS.depth_weight / (other_weights + 1)
    normal_weight = FLAGS.normal_weight / (other_weights + 1)
    other_weights = other_weights / (other_weights + 1)

    policy_weight = FLAGS.policy_weight

    loss_weights = {
        'kld': FLAGS.beta,
        'rgb': 1 - other_weights,
        'depth': depth_weight,
        'normal': normal_weight,
        'policy': policy_weight,
    }

    model_kwargs = dict(
        stage_sizes=[int(i) for i in FLAGS.stage_sizes],
        z_dim=FLAGS.z_dim,
        g_dim=FLAGS.g_dim,
        rnn_size=FLAGS.rnn_size,
        num_base_filters=FLAGS.num_base_filters,
        first_block_shape=[int(i) for i in FLAGS.first_block_shape],
        skip_type=FLAGS.skip_type,
        n_past=FLAGS.n_past,
        action_conditioned=eval(FLAGS.action_conditioned),
        action_size=FLAGS.action_size,
        expand_decoder=FLAGS.expand,
        stochastic=FLAGS.stochastic,
    )

    model = FitVid(model_kwargs=model_kwargs,
                   beta=FLAGS.beta,
                   loss_fn=FLAGS.loss,
                   multistep=FLAGS.multistep,
                   is_inference=False,
                   depth_predictor=depth_predictor_kwargs,
                   normal_predictor=normal_predictor_kwargs,
                   policy_networks=policy_network_kwargs,
                   loss_weights=loss_weights,
                   )

    NGPU = torch.cuda.device_count()
    print('CUDA available devices: ', NGPU)

    checkpoint, resume_epoch = get_most_recent_checkpoint(FLAGS.output_dir)
    print(f'Attempting to load from checkpoint {checkpoint if checkpoint else "None, no model found"}')
    if checkpoint:
        model.load_parameters(checkpoint)

    # dump model config
    with open(os.path.join(FLAGS.output_dir, 'config.json'), 'w') as config_file:
        print('Writing config file...')
        json.dump(model.config, config_file)

    model = torch.nn.DataParallel(model)
    model.to('cuda:0')

    optimizer = torch.optim.Adam(model.parameters(), lr=FLAGS.lr, weight_decay=FLAGS.weight_decay)

    if FLAGS.hdf5_data:
        image_size = [int(i) for i in FLAGS.image_size]
        data_loader = load_hdf5_data(FLAGS.dataset_file, FLAGS.batch_size, image_size=image_size, data_type='train')
        test_data_loader = load_hdf5_data(FLAGS.dataset_file, FLAGS.batch_size, image_size=image_size, data_type='val')
        prep_data = prep_data_test = lambda x: x
    else:
        if FLAGS.debug:
            data_loader, prep_data = load_data(FLAGS.dataset_file, data_type='valid',
                                               depth=FLAGS.depth_model_path, normal=FLAGS.normal_model_path,
                                               seg=FLAGS.has_segmentation)
        else:
            data_loader, prep_data = load_data(FLAGS.dataset_file, data_type='train',
                                               depth=FLAGS.depth_model_path, normal=FLAGS.normal_model_path,
                                               seg=FLAGS.has_segmentation)
        test_data_loader, prep_data_test = load_data(FLAGS.dataset_file, data_type='valid',
                                                     depth=FLAGS.depth_model_path,
                                                     normal=FLAGS.normal_model_path,
                                                     seg=FLAGS.has_segmentation)

    wandb.init(
        project=FLAGS.project,
        reinit=True,
        mode='online' if FLAGS.wandb_online else 'offline'
    )

    if FLAGS.output_dir is not None:
        wandb.run.name = f"{FLAGS.output_dir.split('/')[-1]}"
        wandb.run.save()

    wandb.config.update(FLAGS)
    wandb.config.slurm_job_id = os.getenv('SLURM_JOB_ID', 0)

    train_losses = []
    train_mse = []
    test_mse = []
    num_epochs = FLAGS.num_epochs
    train_steps = 0
    num_batch_to_save = 1

    from torch.cuda.amp import autocast, GradScaler
    from contextlib import ExitStack
    scaler = GradScaler()

    for epoch in range(resume_epoch + 1, num_epochs):
        print(f'\nEpoch {epoch} / {num_epochs}')
        print('Evaluating')
        model.eval()
        with torch.no_grad():
            epoch_mse = []
            for iter_item in enumerate(tqdm(test_data_loader)):
                wandb_log = dict()
                test_batch_idx, batch = iter_item
                batch = dict_to_cuda(prep_data_test(batch))
                with autocast() if FLAGS.fp16 else ExitStack() as ac:
                    metrics, eval_preds = model.module.evaluate(batch, compute_metrics=test_batch_idx % 1 == 0)
                    if test_batch_idx < num_batch_to_save:
                        if depth_predictor_kwargs:
                            with torch.no_grad():
                                gt_depth_preds = model.module.depth_predictor(batch['video'][:, 1:])
                            test_videos_log = {
                                'gt': batch['video'][:, 1:],
                                'pred': eval_preds['rgb'],
                                'gt_depth': batch['depth_video'][:, 1:],
                                'gt_depth_pred': gt_depth_preds,
                                'pred_depth': eval_preds['depth'],
                                'rgb_loss': metrics['loss/mse_per_sample'],
                                'depth_loss_weight': model.module.loss_weights['depth'],
                                'depth_loss': metrics['loss/depth_loss_per_sample'],
                                'name': 'Depth',
                            }
                            test_depth_visualization = build_visualization(**test_videos_log)
                            wandb_log.update(
                                {'test_depth_vis': wandb.Video(test_depth_visualization, fps=4, format='gif')})
                        else:  # only depth case
                            test_videos_log = {
                                'gt': batch['video'][:, 1:],
                                'pred': eval_preds['rgb'],
                                'rgb_loss': metrics['loss/mse_per_sample'],
                            }
                            test_visualization = build_visualization(**test_videos_log)
                            wandb_log.update({'test_vis': wandb.Video(test_visualization, fps=4, format='gif')})
                        if normal_predictor_kwargs:
                            with torch.no_grad():
                                gt_normal_preds = model.module.normal_predictor(batch['video'][:, 1:])
                            test_videos_log = {
                                'gt': batch['video'][:, 1:],
                                'pred': eval_preds['rgb'],
                                'gt_depth': batch['normal'][:, 1:],
                                'gt_depth_pred': gt_normal_preds,
                                'pred_depth': eval_preds['normal'],
                                'rgb_loss': metrics['loss/mse_per_sample'],
                                'depth_loss_weight': model.module.loss_weights['normal'],
                                'depth_loss': metrics['loss/normal_loss_per_sample'],
                                'name': 'Normal',
                            }
                            test_normal_visualization = build_visualization(**test_videos_log)
                            wandb_log.update({'test_normal_vis': wandb.Video(test_normal_visualization, fps=4, format='gif')})
                epoch_mse.append(metrics['loss/mse'].item())
                wandb_log.update({f'eval/{k}': v for k, v in metrics.items()})
                wandb.log(wandb_log)
                if FLAGS.debug and test_batch_idx > 25:
                    break
                if test_batch_idx == 50:
                    break
            test_mse.append(np.mean(epoch_mse))
        print(f'Test MSE: {epoch_mse[-1]}')

        if test_mse[-1] == np.min(test_mse):
            if not os.path.isdir(FLAGS.output_dir):
                os.mkdir(FLAGS.output_dir)
            save_path = os.path.join(FLAGS.output_dir, f'model_best')
            torch.save(model.module.state_dict(), save_path)
            print(f'Saved new best model to {save_path}')

        print('Training')
        model.train()

        if epoch < FLAGS.depth_start_epoch:
            model.module.loss_weights['depth'] = 0
            model.module.loss_weights['normal'] = 0
        else:
            model.module.loss_weights['depth'] = depth_weight
            model.module.loss_weights['normal'] = normal_weight

        for iter_item in enumerate(tqdm(data_loader)):
            wandb_log = {}

            batch_idx, batch = iter_item
            batch = dict_to_cuda(prep_data(batch))
            if 'segmentation' in batch:
                inputs = batch['video'], batch['actions'], batch['segmentation']
            else:
                inputs = batch['video'], batch['actions'], None

            if depth_predictor_kwargs:
                inputs = inputs + (batch['depth_video'],)
            else:
                inputs = inputs + (None, )

            if normal_predictor_kwargs:
                inputs = inputs + (batch['normal'],)
            else:
                inputs = inputs + (None,)

            optimizer.zero_grad()
            with autocast() if FLAGS.fp16 else ExitStack() as ac:
                loss, preds, metrics = model(*inputs, compute_metrics=(batch_idx % 200 == 0))

            if NGPU > 1:
                loss = loss.mean()
                for k, v in metrics.items():
                    if len(metrics[k].shape) == 1:
                        if metrics[k].shape[0] == NGPU:
                            metrics[k] = v.mean()

            if FLAGS.fp16:
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 100)
                optimizer.step()
            train_steps += 1
            train_losses.append(loss.item())
            train_mse.append(metrics['loss/mse'].item())
            if batch_idx < num_batch_to_save:
                if depth_predictor_kwargs:
                    with torch.no_grad():
                        gt_depth_preds = model.module.depth_predictor(batch['video'][:, 1:])
                if depth_predictor_kwargs:
                    train_videos_log = {
                        'gt': batch['video'][:, 1:],
                        'pred': preds['rgb'],
                        'gt_depth': batch['depth_video'][:, 1:],
                        'gt_depth_pred': gt_depth_preds,
                        'pred_depth': preds['depth'],
                        'rgb_loss': metrics['loss/mse_per_sample'],
                        'depth_loss_weight': model.module.loss_weights['depth'],
                        'depth_loss': metrics['loss/depth_loss_per_sample'],
                    }
                    train_depth_visualization = build_visualization(**train_videos_log)
                    wandb_log.update({'train_depth_vis': wandb.Video(train_depth_visualization, fps=4, format='gif')})
                else:
                    train_videos_log = {
                        'gt': batch['video'][:, 1:],
                        'pred': preds['rgb'],
                        'rgb_loss': metrics['loss/mse_per_sample'],
                    }
                    train_visualization = build_visualization(**train_videos_log)
                    wandb_log.update({'train_vis': wandb.Video(train_visualization, fps=4, format='gif')})
                if normal_predictor_kwargs:
                    with torch.no_grad():
                        gt_normal_preds = model.module.normal_predictor(batch['video'][:, 1:])
                    train_videos_log = {
                        'gt': batch['video'][:, 1:],
                        'pred': preds['rgb'],
                        'gt_depth': batch['normal'][:, 1:],
                        'gt_depth_pred': gt_normal_preds,
                        'pred_depth': preds['normal'],
                        'rgb_loss': metrics['loss/mse_per_sample'],
                        'depth_loss_weight': model.module.loss_weights['normal'],
                        'depth_loss': metrics['loss/normal_loss_per_sample'],
                        'name': 'Normal',
                    }
                    train_normal_visualization = build_visualization(**train_videos_log)
                    wandb_log.update({'train_normal_vis': wandb.Video(train_normal_visualization, fps=4, format='gif')})

            wandb_log.update({f'train/{k}': v for k, v in metrics.items()})
            wandb.log(wandb_log)
            if FLAGS.debug and batch_idx > 25:
                break

        if epoch % FLAGS.save_freq == 0:
            if os.path.isdir(FLAGS.output_dir):
                pass
            else:
                os.mkdir(FLAGS.output_dir)
            save_path = os.path.join(FLAGS.output_dir, f'model_epoch{epoch}')
            if NGPU > 1:
                torch.save(model.module.state_dict(), save_path)
            else:
                torch.save(model.state_dict(), save_path)
            print(f'Saved model to {save_path}')
        else:
            print('Skip saving models')


if __name__ == "__main__":
    flags.mark_flags_as_required(['output_dir'])
    app.run(main)
