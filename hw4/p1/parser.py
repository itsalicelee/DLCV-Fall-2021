import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV_HW4_P1_b07303024')
    # data
    parser.add_argument('--train_csv', default='../hw4_data/mini/train.csv')
    parser.add_argument('--train_data_dir', default='../hw4_data/mini/train')
    parser.add_argument('--log_dir', default='log', type=str, help="directory of ckpts")
    # experiment setting
    parser.add_argument('--n_batch', type=int, default=256)
    parser.add_argument('--n_way', type=int, default=5)
    parser.add_argument('--k_shot', type=int, default=1)
    parser.add_argument('--k_query', type=int, default=15)
    parser.add_argument('--loss', type=str, default='euclidean', help="loss function between prototype and query image")
    parser.add_argument('--log_interval', type=int, default=50)
    
   
    # load from ckpt
    parser.add_argument('--ckpt', type=str, default='')

    # training parameters
    parser.add_argument('--seed', type=int, default=2021)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', default=4, type=int, help="number of data loading workers")
    parser.add_argument('--lr_scheduler', type=str, default='')


    args = parser.parse_args()
    return args

