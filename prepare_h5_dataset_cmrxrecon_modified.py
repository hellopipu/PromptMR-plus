'''
This script is used to prepare h5 training dataset from  original matlab dataset for CMRxRecon series dataset.
'''
import glob
import os
from os.path import join
import json
import argparse
import torch
from tqdm import tqdm
import h5py
import numpy as np
from data.transforms import to_tensor
from mri_utils import ifft2c, rss_complex, load_kdata


def remove_bad_files(file_name):
    '''
    some of the cmrxrecon2024 files contain only background or very bad quality, 
    we need to remove them or remove several slices of them, 
    otherwise may cause nan loss while training.
    reference: https://www.synapse.org/Synapse:syn61491109
    '''
    rm_files = ['P077_aorta_sag','P105_tagging','P065_aorta_sag','P117_aorta_tra']
    if file_name in rm_files:
        return True
    else:
        return False

def remove_bad_slices(kdata_, file_name):
    '''
    remove several slices which are bad quality or only background
    '''
    if file_name == 'P087_tagging':
        kdata_ = kdata_[0:-3,0:-1]
    elif file_name == 'P105_aorta_tra':
        kdata_ = kdata_[0:-12]
    elif file_name == 'P109_cine_sax':
        kdata_ = kdata_[0:-3]
    elif file_name == 'P100_cine_sax':
        kdata_ = kdata_[0:-4]
        kdata_ = np.concatenate([kdata_[:,0:1], kdata_[:,2:]],axis=1)
    return kdata_


if __name__ == '__main__':
    # add argparse
    parser = argparse.ArgumentParser(description='Prepare H5 dataset for CMRxRecon series dataset')
    parser.add_argument('--output_h5_folder', type=str,
                        default='/common/users/bx64/dataset/CMRxRecon2024/h5_dataset',
                        help='path to save H5 dataset')
    parser.add_argument('--input_matlab_folder', type=str,
                        default='/common/users/bx64/dataset/CMRxRecon2024/home2/Raw_data/MICCAIChallenge2024/ChallengeData/MultiCoil',
                        help='path to the original matlab data')
    parser.add_argument('--split_json', type=str, default='configs/data_split/cmr24-cardiac.json', help='path to the split json file')
    parser.add_argument('--train_valid_test', type=str, required=True, help='train,valid,or test dataset (valid and test belong to after)')
    parser.add_argument('--year', type=int, required=True, choices=[2025,2024, 2023], help='year of the dataset')
    parser.add_argument('--symbol_only', type=int, default=0, choices=[0,1], help='whether to do symbol link only')
    parser.add_argument('--only_specific', action='store_true', help='Only process T1w, T2w, and blackblood data')

    args = parser.parse_args()
    
    save_folder = args.output_h5_folder
    mat_folder = args.input_matlab_folder
    year = args.year
    split_json = args.split_json

    print('matlab data folder: ', mat_folder)
    print('h5 save folder: ', save_folder)

    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
        
    print('## step 1: convert matlab training dataset to h5 dataset')
    if args.year!=2025:
        if args.train_valid_test=='train':
            file_list = sorted(glob.glob(join(mat_folder, '*/TrainingSet/FullSample/P*/*.mat')))
        elif args.train_valid_test=='test':
            file_list = sorted(glob.glob(join(mat_folder, '*/TestSet/FullSample/P*/*.mat')))
        elif args.train_valid_test=='valid':
            file_list = sorted(glob.glob(join(mat_folder, '*/ValidationSet/FullSample/P*/*.mat')))
    else:
        file_list = sorted(glob.glob(join(mat_folder, '*/TrainingSet/FullSample/*/*/P*/*.mat')))
    
    print('number of total matlab files: ', len(file_list))
    
    # check if cuda is available
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    if not args.symbol_only:
        for ff in tqdm(file_list):
            ##* get info from path
            if args.year!=2025:
                fid = ff.split('/')[-2]
                ftype = ff.split('/')[-1].split('.')[0]
                save_name = f'{fid}_{ftype}'
            else:
                fcenter=ff.split('/')[-4]
                fmachine=ff.split('/')[-3]
                fid=ff.split('/')[-2]
                ftype = ff.split('/')[-1].split('.')[0]
                save_name = f'{fcenter}_{fmachine}_{fid}_{ftype}'
            ##*remove bad files
            if remove_bad_files(save_name) and year == 2024:
                continue
            
            # EDIT: Only process files of type T1w, T2w, or blackblood if --only_specific is set
            if args.only_specific and ftype not in ["T1w", "T2w", "blackblood"]:
                continue

            ##* load kdata
            kdata = load_kdata(ff)

            ##* swap phase_encoding and readout
            kdata = kdata.swapaxes(-1,-2)
            
            ##* remove bad slices
            if year == 2024:
                kdata = remove_bad_slices(kdata, save_name)
            
            # EDIT: If file type is T1w, T2w, or blackblood, unsqueeze kspace data at axis 0 then repeat
            if ftype in ["T1w", "T2w", "blackblood"]:
                kdata = np.expand_dims(kdata, axis=0)
                kdata = np.repeat(kdata, 2, axis=0)

            ##* get rss from kdata
            kdata_th = to_tensor(kdata)
            img_coil = ifft2c(kdata_th).to(device)
            img_rss = rss_complex(img_coil, dim=-3).cpu().numpy()

            ##* save h5
            file = h5py.File(join(save_folder, save_name + '.h5'), 'w')
            file.create_dataset('kspace', data=kdata)
            file.create_dataset('reconstruction_rss', data=img_rss)

            file.attrs['max'] = img_rss.max()
            file.attrs['norm'] = np.linalg.norm(img_rss)
            file.attrs['acquisition'] = ftype
            file.attrs['shape'] = kdata.shape
            file.attrs['padding_left'] = 0
            file.attrs['padding_right'] = kdata.shape[-1]
            file.attrs['encoding_size'] = (kdata.shape[-2],kdata.shape[-1],1)
            file.attrs['recon_size'] = (kdata.shape[-2],kdata.shape[-1],1)
            file.attrs['patient_id'] = save_name
            file.close()
    
    print('## step 2: split h5 dataset to train and val using symbolic links')
    # split dataset to train/ val according to provided json file
    with open(split_json, 'r', encoding="utf-8") as f:
        split_dict = json.load(f)
    print('train files in json: ', len(split_dict['train']))
    print('val files in json: ', len(split_dict['val']))  
    
    train_folder = save_folder.replace(save_folder.split('/')[-1], 'train')
    val_folder = save_folder.replace(save_folder.split('/')[-1], 'val')
    aftervalid_folder = save_folder.replace(save_folder.split('/')[-1], 'after/valid')
    aftertest_folder = save_folder.replace(save_folder.split('/')[-1], 'after/test')
    if not os.path.exists(train_folder):
        os.makedirs(train_folder,exist_ok=True)
    if not os.path.exists(val_folder):
        os.makedirs(val_folder,exist_ok=True)
    os.makedirs(aftertest_folder,exist_ok=True)
    os.makedirs(aftervalid_folder,exist_ok=True)
    file_list = sorted(glob.glob(join(save_folder,'*.h5')))
    print('number of total files in folder h5_dataset: ', len(file_list))
    
    train_list = [ff.split('/')[-1] for ff in split_dict['train']]
    val_list = [ff.split('/')[-1] for ff in split_dict['val']]
    
    if args.train_valid_test=='train':
        for ff in file_list:
            save_name = ff.split('/')[-1]
            if save_name in train_list:
                os.symlink(ff, join(train_folder, save_name))
            elif save_name in val_list:
                os.symlink(ff, join(val_folder, save_name))
    elif args.train_valid_test=='valid':
        for ff in file_list:
            save_name = ff.split('/')[-1]
            os.symlink(ff, join(aftervalid_folder, save_name))
    elif args.train_valid_test=='test':
        for ff in file_list:
            save_name = ff.split('/')[-1]
            os.symlink(ff, join(aftertest_folder, save_name))
    print('Done!')
    print('number of files in h5 folder: ', len(file_list))
    print('number of symbolic link files in train folder: ', len(glob.glob(join(train_folder, '*.h5'))))
    print('number of symbolic link files in val folder: ', len(glob.glob(join(val_folder, '*.h5'))))
    print('number of symbolic link files in aftervalid folder: ', len(glob.glob(join(aftervalid_folder, '*.h5'))))
    print('number of symbolic link files in aftertest folder: ', len(glob.glob(join(aftertest_folder, '*.h5'))))