import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV_HW1_P2_b07303024')

    # Datasets parameters
    parser.add_argument('--num_workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
    parser.add_argument('--improved', default=False, type=bool,
                    help="improved model or not")
    
    # training parameters
    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--model', default="unet", type=str)
    parser.add_argument('--train_data',default="../hw1_data/p2_data/train/", type=str)
    parser.add_argument('--val_data',default="../hw1_data/p2_data/validation/", type=str)
    parser.add_argument('--test_data',default="", type=str)

    parser.add_argument('--num_gpu', default=0, type=int,
                    help='number of GPUs')
    parser.add_argument('--optimizer', default="Adam", type=str)
    parser.add_argument('--epoch', default=100, type=int,
                    help="num of training iterations")
    parser.add_argument('--val_epoch', default=10, type=int,
                    help="num of validation iterations")
    parser.add_argument('--train_batch', default=8, type=int,
                    help="train batch size")
    parser.add_argument('--test_batch', default=1, type=int, 
                    help="test batch size")
    parser.add_argument('--lr', default=0.0004, type=float,
                    help="initial learning rate")
    parser.add_argument('--lr_scheduler', default=False, type=bool,
                    help="schedule or not")
    parser.add_argument('--weight_decay', default=0.0002, type=float,
                    help="initial weight decay")
    parser.add_argument('--log_interval', default=8, type=int,
                    help="save model in log interval epochs")


    # resume trained model
    parser.add_argument('--resume', type=str, default='', 
                    help="path to the trained model")
    # inference model
    parser.add_argument('--test', type=str, default='', 
                    help="path to the trained model")
      
    # others
    parser.add_argument('--save_dir', type=str, default='log')
    parser.add_argument('--random_seed', type=int, default=2021)
    parser.add_argument('--result_dir', type=str, default='results', 
                    help="path to the saved result file")           



    args = parser.parse_args()

    return args
