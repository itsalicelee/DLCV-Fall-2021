import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV_HW2_P2_b07303024')
    # Datasets parameters
    parser.add_argument('--num_workers', default=4, type=int,
                    help="number of data loading workers (default: 4)")
    # Directories
    parser.add_argument('--train_data',default="../hw2_data/digits", type=str)
    parser.add_argument('--source',default="", type=str)
    parser.add_argument('--target',default="", type=str)
    parser.add_argument('--test_data',default="../hw2_data/digits", type=str)
    parser.add_argument('--log_dir', type=str, default='ckpts/log')
    parser.add_argument('--result_dir', type=str, default='results', 
                    help="path to the saved result file")   
    parser.add_argument('--test', type=str, default='', 
                    help="path to the trained model") # inference model
    parser.add_argument('--ckpt_f', type=str, default='',  
                    help="path to the trained F model") # resume trained model
    parser.add_argument('--ckpt_l', type=str, default='',  
                    help="path to the trained Label predictor model") # resume trained model
    parser.add_argument('--ckpt_d', type=str, default='',  
                    help="path to the trained D model") # resume trained model
    
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
    parser.add_argument('--val_epoch', default=10, type=int,
                    help="num of validation iterations")
    parser.add_argument('--train_batch', default=128, type=int,
                    help="train batch size")
    parser.add_argument('--test_batch', default=128, type=int, 
                    help="test batch size")
    parser.add_argument('--lr_f', default=0.0002, type=float,
                    help="feature extractor learning rate")
    parser.add_argument('--lr_l', default=0.0002, type=float,
                    help="label predictor learning rate")
    parser.add_argument('--lr_d', default=0.0002, type=float,
                    help="domain classifier learning rate")
    parser.add_argument('--lr_scheduler', default=False, type=bool,
                    help="schedule or not")
    parser.add_argument('--weight_decay', default=0, type=float,
                    help="initial weight decay")
    parser.add_argument('--log_interval', default=5, type=int,
                    help="print in log interval iterations")
    # random seed
    parser.add_argument('--random_seed', type=int, default=2021)
        
    # for inference
    parser.add_argument('--improved', type=bool, default=False)
    parser.add_argument('--inf_testdata', type=str, default='')
    parser.add_argument('--inf_target', type=str, default='')
    parser.add_argument('--save_dir', type=str, default='./test_pred.csv', 
                    help="path to the output csv file")  #TODO: specify in sh 


    args = parser.parse_args()

    return args
