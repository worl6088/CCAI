from data_modules.utils import *
from data_modules.downloader import *
from data_modules.csv_downloader import *
from data_modules.utils import bcolors as bc


def bounding_boxes_images(args, root_dir, default_oid_dir):
    """ This function should be called when args.command is 'all' or 'downloader' """

    if not args.Dataset:
        dataset_dir = default_oid_dir  # ../data/custom
        csv_dir = os.path.join(default_oid_dir, 'csv_folder')
    else:
        dataset_dir = os.path.join(default_oid_dir, args.Dataset)
        csv_dir = os.path.join(default_oid_dir, 'csv_folder')

    name_file_class = 'class-descriptions-boxable.csv'
    classes_csv_path = os.path.join(csv_dir, name_file_class)

    logo(args.command)
    folder = ['train', 'validation']
    file_list = ['train-annotations-bbox.csv', 'validation-annotations-bbox.csv']

    if args.classes[0].endswith('.txt'):
        with open(args.classes[0]) as f:
            args.classes = f.readlines()
            args.classes = [x.strip() for x in args.classes]
        print('download classes: ' + str(args.classes))

    domain_list = args.classes
    # domain_list =>  ['group1 Orange Apple', 'group2 Bus Traffic_light Car Fire_hydrant']
    name_file_path = os.path.join(default_oid_dir, 'domain_list')
    domain_dict = make_domain_list(name_file_path,
                                   domain_list)  # create class list file for each domain in data/custom/domain_list directory
    mkdirs(dataset_dir, csv_dir, domain_dict)

    for domain_name, class_list in domain_dict.items():
        print(bc.INFO + 'Downloading {} together.'.format(str(class_list[1:])) + bc.ENDC)
        error_csv(name_file_class, csv_dir, args.yes)
        df_classes = pd.read_csv(classes_csv_path, header=None)
        class_dict = {}
        # class_dict => : {'Orange': '/m/0cyhj_', 'Apple': '/m/014j1m'}

        for class_name in class_list[1:]:
            class_dict[class_name] = df_classes.loc[df_classes[1] == class_name].values[0][0]

        print(class_dict)

        for class_name in class_list[1:]:
            for i in range(2):
                name_file = file_list[i]
                df_val = TTV(csv_dir, name_file, args.yes)
                data_type = name_file[:5]  # train or valid

                if not args.n_threads:
                    download(args, data_type, df_val, folder[i], dataset_dir, class_name, class_dict[class_name],
                             domain_name, domain_dict)
                else:

                    download(args, data_type, df_val, folder[i], dataset_dir, class_name, class_dict[class_name],
                             domain_name, domain_dict, args.n_threads)

        make_config_file(root_dir, default_oid_dir, domain_dict)

    return domain_dict
