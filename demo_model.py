import torch
from models import PromptMR, count_parameters

if __name__ == "__main__":
    
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    
    n_adj_slc = 5
    n_coil = 10
    
    model_promptmr = PromptMR(            
                num_cascades=12,
                num_adj_slices=n_adj_slc,
                n_feat0=48,
                feature_dim = [72, 96, 120],
                prompt_dim = [24, 48, 72],
                sens_n_feat0=24,
                sens_feature_dim = [36, 48, 60],
                sens_prompt_dim = [12, 24, 36],
                len_prompt = [5, 5, 5],
                prompt_size = [64, 32, 16],
                n_enc_cab = [2, 3, 3],
                n_dec_cab = [2, 2, 3],
                n_skip_cab = [1, 1, 1],
                n_bottleneck_cab = 3,
                no_use_ca = False,
                adaptive_input=False,
                n_buffer = 0,
                n_history=0,
                use_sens_adj=False, 
        )
    
    model_promptmr_plus = PromptMR(            
                num_cascades=12,
                num_adj_slices=n_adj_slc,
                n_feat0=48,
                feature_dim = [72, 96, 120],
                prompt_dim = [24, 48, 72],
                sens_n_feat0=24,
                sens_feature_dim = [36, 48, 60],
                sens_prompt_dim = [12, 24, 36],
                len_prompt = [5, 5, 5],
                prompt_size = [64, 32, 16],
                n_enc_cab = [2, 3, 3],
                n_dec_cab = [2, 2, 3],
                n_skip_cab = [1, 1, 1],
                n_bottleneck_cab = 3,
                no_use_ca = False,
                adaptive_input=True,        # adaptive input
                n_buffer = 4,               # buffer size in adaptive input, fixed to 4
                n_history=11,               # historical feature aggregation
                use_sens_adj=True,          # adjacent sensitivity map estimation
        )
        
    model_promptmr.to(device)
    model_promptmr_plus.to(device)

    #* use random input
    rand_input = torch.randn(1, n_adj_slc*n_coil, 218, 170, 2)
    rand_mask = torch.randn(1, 1, 218, 170, 1).bool()
    num_low_frequencies = torch.tensor([18])

    output = model_promptmr(rand_input.to(device), rand_mask.to(device), num_low_frequencies.to(device))
    output_plus = model_promptmr_plus(rand_input.to(device), rand_mask.to(device), num_low_frequencies.to(device))
    
    print('model_promptmr param: ', count_parameters(model_promptmr))
    print('model_promptmr+ param: ', count_parameters(model_promptmr_plus))
    
    print('output_promptmr shape: ', output['img_pred'].shape)
    print('output_promptmr+ shape: ', output_plus['img_pred'].shape)

        



