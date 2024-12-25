from argparse import ArgumentParser


def parse_args():

    parser = ArgumentParser(description='training code')

    # Hyper--parameters
    parser.add_argument('--optimizer', help='adamax/adamw/adam/sgd', default='adamax', type=str)
    parser.add_argument('--lr', dest='lr', help='learning rate', default=1e-3, type=float)
    parser.add_argument('--nepochs', help='epoch number', default=30, type=float)

    # InComNet model
    parser.add_argument('--mode', dest='mode', help='predcls/sgcls/sgdet', default='predcls', type=str)
    parser.add_argument('--iterations', help='Number of interations of InComNet', default=10, type=int)
    parser.add_argument('--clip_model', help='Can be one of ViT_B_32/VIT_L_14_336/ViT_L_14_336_sft', default='ViT_L_14_336_sft', type=str)
    parser.add_argument('--clip_sft_path', help='Path to VIT-L-14-336-sft model', default='/home/chinthani/InComNet/pre_trained_models/clip_ViT-L-14-336-SSG-sft.pt', type=str)

    # Data path
    parser.add_argument('--data_path', default='./data/', type=str)
    parser.add_argument('--frame_path', default='/data/dataset/charades/AG/frames/', type=str)
    parser.add_argument('--datasize', dest='datasize', help='mini dataset or whole', default='large', type=str)

    # Model paths
    parser.add_argument('--save_path', default='./incomnet_trained_models', type=str) 
    parser.add_argument('--model_path', default='./incomnet_trained_models', type=str)
    parser.add_argument('--ckpt', help="trained model", default='./incomnet_trained_models/incomnet-224_epoch_0.tar', type=str)
    parser.add_argument('--ckpt_person', help="trained person model", default='./incomnet_trained_models/incomnet-224_person_epoch_0.tar', type=str)

    # Evaluation settings
    parser.add_argument('--top1', help='Verb SR evaluation setting', default=True, type=bool)

    # Other settings
    parser.add_argument('--seed', help='manual seed', default=2222, type=int)
    parser.add_argument('--device', help='cuda device', default='cuda:3', type=str)
    parser.add_argument('--wandb', help='visualize loss curves', default=False, type=bool)

    return parser.parse_args()