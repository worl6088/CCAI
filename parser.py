import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description='Open Image Dataset Downloader')

    parser.add_argument("command",
                        metavar="<command> 'downloader', 'train', 'all'.",
                        help="'downloader' or 'train' or 'all'.")
    # 'all' command perform both download and trainig
    parser.add_argument('--limit', required=False, type=int, default=None,
                        metavar="integer number",
                        help='Optional limit on number of images to download')

    parser.add_argument('--n_threads', required=False, metavar="[default 20]",
                        help='Num of the threads to use')

    parser.add_argument('--classes', required=False, default='domains.txt',nargs='+',
                        metavar="list of classes",
                        help="Sequence of 'strings' of the wanted classes")

    parser.add_argument('--noLabels', required=False, action='store_true',
                        help='No labels creations')

    parser.add_argument("--gpu_num", type=int, default=0, help="select number of gpu") # multi-gpu training function will be implemented later

    parser.add_argument("--epochs", type=int, default=100, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=8, help="size of each image batch")
    parser.add_argument("--model_def", type=str, default=None, help="path to model definition file")
    parser.add_argument("--pretrained_weights", type=str, help="if specified starts from checkpoint model")
    parser.add_argument("--checkpoint_interval", type=int, default=5, help="interval between saving model weights")
    parser.add_argument("--evaluation_interval", type=int, default=5, help="interval evaluations on validation set")
    parser.add_argument("--img_size", type=int, default=416, help="size of each image dimension")

    parser.add_argument("--logdir", type=str, default="logs", help="Defines the directory where the training log files are stored")

    # From below,  setting is not essential

    parser.add_argument('--Dataset', required=False,
                        metavar="/path/to/custom/csv/",
                        help='Directory of the OID dataset folder')
    parser.add_argument('-y', '--yes', required=False, action='store_true',
                        #metavar="Yes to download missing files",
                        help='ans Yes to possible download of missing files')

    parser.add_argument('--sub', required=False, choices=['h', 'm'],
                        metavar="Subset of human verified images or machine generated (h or m)",
                        help='Download from the human verified dataset or from the machine generated one.')


    # image dataset option
    parser.add_argument('--image_IsOccluded', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates that the object is occluded by another object in the image.')
    parser.add_argument('--image_IsTruncated', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates that the object extends beyond the boundary of the image.')
    parser.add_argument('--image_IsGroupOf', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates that the box spans a group of objects (min 5).')
    parser.add_argument('--image_IsDepiction', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates that the object is a depiction.')
    parser.add_argument('--image_IsInside', required=False, choices=['0', '1'],
                        metavar="1 or 0",
                        help='Optional characteristic of the images. Indicates a picture taken from the inside of the object.')

    # training option
    parser.add_argument("--gradient_accumulations", type=int, default=2, help="number of gradient accums before step")
    parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
    parser.add_argument("--compute_map", default=False, help="if True computes mAP every tenth batch")
    parser.add_argument("--multiscale_training", default=True, help="allow for multi-scale training")
    parser.add_argument("--verbose", "-v", default=False, action='store_true', help="Makes the training more verbose")


    return parser.parse_args()
