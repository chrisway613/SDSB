import os, torch, argparse, tqdm, random, numpy as np
from util.dataset import create_data
from util.noiser import create_noiser
from util.model import create_model
from util.visualize import InferenceResultVisualizer, TrajectoryVisualizer

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def match_ckpt(ckpt):
    _ckpt = {}
    for k, v in ckpt.items():
        if 'module.' in k:
            k = k.replace('network.module.', 'network.')
        _ckpt[k] = v
    return _ckpt

def main(args):
    seed_everything(42)

    device = torch.device(f'cuda')

    _, _, prior_loader = create_data(args.prior, 1, dataset_size=args.num_sample, batch_size=args.num_sample)
    _, _, data_loader = create_data(args.dataset, 1, dataset_size=args.num_sample, batch_size=args.num_sample)

    noiser = create_noiser(args.noiser, args, device)

    backward_model = create_model(args.method, args, device, noiser, rank=0, direction='b')
    forward_model = create_model(args.method, args, device, noiser, rank=0, direction='f')

    backward_model.to(device)
    forward_model.to(device)

    ckpt = torch.load(args.ckpt, map_location='cpu')
    backward_model.load_state_dict(match_ckpt(ckpt['backward_model']), strict=True)
    forward_model.load_state_dict(match_ckpt(ckpt['forward_model']), strict=True)

    with torch.no_grad():

        backward_model.eval()
        forward_model.eval()

        save_path = os.path.join('inference', args.exp_name)

        x_prior = next(iter(prior_loader)).to(device)
        x_data = next(iter(data_loader)).to(device)

        qs = backward_model.inference(x_prior, sample=True)[0]
        ps = forward_model.inference(x_data, sample=True)[0]

        inferenceResultVisualizer = InferenceResultVisualizer(args, device, save_path=save_path)
        trajectoryVisualizer = TrajectoryVisualizer(args, device, save_path=save_path)
        
        inferenceResultVisualizer.draw(0, 0, qs[-1], subfix=f'_q')
        inferenceResultVisualizer.draw(0, 0, ps[-1], subfix=f'_p')

        trajectoryVisualizer.draw(0, 0, xs=qs, subfix='_q')
        trajectoryVisualizer.draw(0, 0, xs=ps, subfix='_p')

def create_parser():
    argparser = argparse.ArgumentParser()
    
    argparser.add_argument('--exp2d', action='store_true', help='set to true for 2d experiments')

    argparser.add_argument('--num_sample', type=int, default=65536, help='number of samples')
    
    argparser.add_argument('--method', type=str, default='dsb', help='method')
    argparser.add_argument('--simplify', action='store_true', help='whether to use simplified DSB')
    argparser.add_argument('--reparam', type=str, default=None, help='whether to use reparameterized DSB, "term" for TR-DSB, "flow" for FR-DSB')
    argparser.add_argument('--noiser', type=str, default='flow', help='noiser type, "flow" noiser for Flow Matching models, "dsb" noiser for DSB models')
    argparser.add_argument('--gamma_type', type=str, default='constant', help='gamma schedule for DSB')
    argparser.add_argument('--training_timesteps', type=int, default=16, help='training timesteps')
    argparser.add_argument('--inference_timesteps', type=int, default=16, help='inference timesteps')
    
    argparser.add_argument('--network', type=str, default='mlp', help='network architecture to use')
    argparser.add_argument('--use_amp', action='store_true', help='whether to use mixed-precision training')

    argparser.add_argument('--prior', type=str, default='standard', help='prior distribution')
    argparser.add_argument('--dataset', type=str, default='checkerboard:4', help='data distribution')

    argparser.add_argument('--exp_name', type=str, default='try', help='name of experiment')
    argparser.add_argument('--ckpt', type=str, default=None, help='checkpoint to load')

    return argparser


if __name__ == '__main__':

    argparser = create_parser()
    args = argparser.parse_args()

    if 'dsb' in args.method:
        assert args.training_timesteps == args.inference_timesteps

    main(args)
