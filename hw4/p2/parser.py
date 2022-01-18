import argparse

def arg_parse():
    parser = argparse.ArgumentParser(description='DLCV_HW4_P2_b07303024')
    # data
    parser.add_argument('--train_csv', default='../hw4_data/office/train.csv')
    parser.add_argument('--train_data_dir', default='../hw4_data/office/train')
    parser.add_argument('--test_csv', default='../hw4_data/office/val.csv')
    parser.add_argument('--test_data_dir', default='../hw4_data/office/val')
    parser.add_argument('--log_dir', default='ckpts/log', type=str, help="directory of ckpts")
    parser.add_argument('--save_path', default='./pred.csv', type=str, help="output csv file")
    parser.add_argument('--load', type=str, default='', help="load model for inference")
    # experiment setting
    parser.add_argument('--setting', type=str, default='', choices=['a','b','c','d','e','inference'])
    parser.add_argument('--resume', type=str, default='')
    parser.add_argument('--provided_ckpt', type=str, default='../hw4_data/pretrain_model_SL.pt')
    parser.add_argument('--ssl_ckpt', type=str, default='./ckpts/pretrained/ssl_model3.pth')
    # training parameters
    parser.add_argument('--log_interval', type=int, default=32)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--lr_scheduler', type=str, default='')
    parser.add_argument('--step', type=int, default=20)
    parser.add_argument('--weight_decay', type=float, default=1e-5)
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--workers', default=4, type=int, help="number of data loading workers")
    parser.add_argument('--seed', type=int, default=2021)
    args = parser.parse_args()
    return args

