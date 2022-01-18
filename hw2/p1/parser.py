import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV_HW2_P1_b07303024')
    # Datasets parameters
    parser.add_argument('--num_workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
    # Directories
    parser.add_argument('--train_data',default="../hw2_data/face/train/", type=str)
    parser.add_argument('--test_data',default="../hw2_data/face/test/", type=str)
    parser.add_argument('--log_dir', type=str, default='ckpts/log')
    parser.add_argument('--result_dir', type=str, default='results', 
                    help="path to the saved result file")   
    parser.add_argument('--test', type=str, default='', 
                    help="path to the trained model") # inference model
    parser.add_argument('--ckpt_g', type=str, default='',  
                    help="path to the trained G model") # resume trained model
    parser.add_argument('--ckpt_d', type=str, default='',  
                    help="path to the trained D model") # resume trained model
    parser.add_argument('--save_dir', type=str, default='inference', 
                    help="path to the 1000 output images")  #TODO: specify in sh 
    # training parameters
    parser.add_argument('--mode', default="train", type=str)
    parser.add_argument('--num_gpu', default=0, type=int,
                    help='number of GPUs')
    parser.add_argument('--optimizer', default="Adam", type=str)
    parser.add_argument('--epochs', default=500, type=int,
                    help="num of training iterations")
    parser.add_argument('--g_iter', default=1, type=int,
                    help="num of training g iterations")
    parser.add_argument('--d_iter', default=1, type=int,
                    help="num of training d iterations")
    parser.add_argument('--beta1', default=0.5, type=int,
                    help="beta 1 of optimizer")
    parser.add_argument('--val_epoch', default=10, type=int,
                    help="num of validation iterations")
    parser.add_argument('--train_batch', default=128, type=int,
                    help="train batch size")
    parser.add_argument('--test_batch', default=128, type=int, 
                    help="test batch size")
    parser.add_argument('--lr_d', default=0.0001, type=float,
                    help="initial learning rate")
    parser.add_argument('--lr_g', default=0.0002, type=float,
                    help="initial learning rate")
    parser.add_argument('--lr_scheduler', default=False, type=bool,
                    help="schedule or not")
    parser.add_argument('--weight_decay', default=8e-5, type=float,
                    help="initial weight decay")
    parser.add_argument('--log_interval', default=5, type=int,
                    help="print in log interval iterations")
    # random seed
    parser.add_argument('--random_seed', type=int, default=2021)
        



    args = parser.parse_args()

    return args
