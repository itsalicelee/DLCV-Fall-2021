import argparse


def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV_HW3_P1_b07303024')
    # Datasets parameters
    parser.add_argument('--num_workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
    # Directories
    parser.add_argument('--train_data',default="../hw3_data/p1_data/train", type=str)
    parser.add_argument('--val_data',default="../hw3_data/p1_data/val", type=str)
    parser.add_argument('--test_data',default="../hw3_data/p1_data/val", type=str)
    
    parser.add_argument('--log_dir', type=str, default='ckpts/log')
    parser.add_argument('--save_path', type=str, default='pred.csv', 
                    help="path to the saved result file")   
    parser.add_argument('--test', type=str, default='', 
                    help="path to the trained model") # inference model
    parser.add_argument('--type', type=str, default='B_16', 
                    help="path to the trained model") # inference model
    parser.add_argument('--resume', type=str, default='', 
                    help="path to the trained model for resume") # resume model
    parser.add_argument('--model_name', type=str, default='vit_base_patch16_224', 
                    help="model name of timm create model") # resume model
    # Mode Setting
    parser.add_argument('--mode', default="train", type=str)
    # Training parameters
    parser.add_argument('--accum_iter', default=5, type=int)
    parser.add_argument('--warmup_ratio', default=0.1, type=float)
    parser.add_argument('--num_gpu', default=1, type=int,
                    help='number of GPUs')
    parser.add_argument('--size', default=384, type=int,
                    help='training image size')
    parser.add_argument('--optimizer', default="Adam", type=str)
    parser.add_argument('--epochs', default=1000, type=int,
                    help="num of training iterations")
    parser.add_argument('--train_batch', default=32, type=int,
                    help="train batch size")
    parser.add_argument('--test_batch', default=128, type=int, 
                    help="test batch size")
    parser.add_argument('--lr', default=0.0001, type=float,
                    help="initial learning rate")
    parser.add_argument('--lr_scheduler', default='none', type=str,
                    help="schedule or not")
    parser.add_argument('--weight_decay', default=0, type=float,
                    help="initial weight decay")
    parser.add_argument('--log_interval', default=5, type=int,
                    help="print in log interval iterations")
    # Random seed
    parser.add_argument('--random_seed', type=int, default=1234)

    args = parser.parse_args()
    return args
