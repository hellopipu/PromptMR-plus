
% Specify File location
%file_string = '/home/chenj6/HPC_common/cmrchallenge/data/CMR2025/Processed/MultiCoil/Center001_UIH_30T_umr780_P001_T2w.h5';
file_string = '/home/shenc2/Desktop/sshfs/chushu/PromptMR-plus/CMR2025_debug_T2w/reconstructions/Center001_UIH_30T_umr780_P049_T2w.h5'
%filestring='/home/shenc2/Desktop/sshfs/chushu/PromptMR-plus/CMR2025_debug_T1w/reconstructions'

% Display H5 file info
h5disp(file_string)

% Extract Recon
reconstruction_rss = h5read(file_string, '/reconstruction');

% Extract Kspace
kspace_struct = h5read(file_string, '/kspace');
kspace_r = kspace_struct.r;
kspace_i = kspace_struct.i;
kspace = complex(kspace_r, kspace_i);

% Self Recon

image = ifftshift(ifftshift(ifft2(fftshift(fftshift(kspace,1),2)),1),2);
image_rss = squeeze(sqrt(sum(image.*conj(image),3)));
%%
%folder_name='/home/shenc2/Desktop/sshfs/chushu/PromptMR-plus_modified/CMR2025_debug_T2w/reconstructions';
folder_name = '/home/shenc2/Desktop/sshfs/chushu/PromptMR-plus/output_chushu/debug_T3w/reconstructions'
cd(folder_name)
for i =3:size(dir(folder_name))
    %i
    i=3
    filename_all=dir(folder_name);
    file_name=filename_all(i).name;
    h5disp(file_name)
    reconstruction_rss = h5read(file_name, '/reconstruction');
    findUniqueSlices(reconstruction_rss,4)
end



