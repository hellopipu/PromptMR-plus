#!/usr/bin/env bash
set -x

# knee

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_singlecoil_train.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=VI6nHpRu4%2BfnUzeH1MMrWCSmfmo%3D&Expires=1751847184" --output knee_singlecoil_train.tar.xz

echo "First finished."

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_singlecoil_val.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=xHhH4dBsuLl3kpbl2u4%2F5u4jh50%3D&Expires=1751847184" --output knee_singlecoil_val.tar.xz

echo "Second finished."

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_singlecoil_test.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=Nv9WZDA%2FE7OjLsxkcwhsmEY4UkE%3D&Expires=1751847184" --output knee_singlecoil_test_v2.tar.xz

echo "third."

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_multicoil_train_batch_0.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=fCerrJ%2BGnJBN6HY3ErFW3EOjwCc%3D&Expires=1751847184" --output knee_multicoil_train_batch_0.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_multicoil_train_batch_1.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=uILQeMH%2F%2BHRNKRs1eJZ97fk3LEk%3D&Expires=1751847184" --output knee_multicoil_train_batch_1.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_multicoil_train_batch_2.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=DnukfJ%2FkdBEqVcMA94%2BatYH%2B7%2FI%3D&Expires=1751847184" --output knee_multicoil_train_batch_2.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_multicoil_train_batch_3.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=wU9dWIEVnwHntdZH55RzAYejGIY%3D&Expires=1751847184" --output knee_multicoil_train_batch_3.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_multicoil_train_batch_4.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=Oy3T07MEyZ29O96E%2Fgz516mxP%2Bc%3D&Expires=1751847184" --output knee_multicoil_train_batch_4.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_multicoil_val.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=oPVrrBHj6n0E7yp9fYB3QBp2cx0%3D&Expires=1751847184" --output knee_multicoil_val.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_multicoil_test.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=g3KKhIHhu6prLrgmV2AgPW5h5MU%3D&Expires=1751847184" --output knee_multicoil_test.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_DICOMs_batch1.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=SWOPANi%2F1%2FIdlutVKP6YCi%2FmC6w%3D&Expires=1751847184" --output knee_DICOMs_batch1.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/knee_DICOMs_batch2.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=mkS1QT5jjYVePfPlKtoLFK1bZe8%3D&Expires=1751847184" --output knee_DICOMs_batch2.tar.xz

# brain 

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_0.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=hjE9%2F1lGL12MrL1h2KaIRv%2Fw3mw%3D&Expires=1751847184" --output brain_multicoil_train_batch_0.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_1.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=2kLdqTYUjxVoDtiJLX47a9%2BW2uk%3D&Expires=1751847184" --output brain_multicoil_train_batch_1.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_2.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=1TtYltRjht2UeqQlN%2Fioyv3RtVk%3D&Expires=1751847184" --output brain_multicoil_train_batch_2.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_3.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=ELypAEbo194BLwOgil17PFyIlFw%3D&Expires=1751847184" --output brain_multicoil_train_batch_3.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_4.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=o6mvBZQzjQdPxse04XpHhun3Ub0%3D&Expires=1751847184" --output brain_multicoil_train_batch_4.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_5.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=Vf9fWpVgmBjLK9Z%2BT4zJy33Lbwc%3D&Expires=1751847184" --output brain_multicoil_train_batch_5.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_6.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=UABw7XPYoyi7zIi3vKmdJidOgSc%3D&Expires=1751847184" --output brain_multicoil_train_batch_6.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_7.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=3m05J%2FUc0Zc7NFU72OqJSHA57%2BU%3D&Expires=1751847184" --output brain_multicoil_train_batch_7.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_8.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=dttiSfPMzpSxcpPyO9VCi3ADBMM%3D&Expires=1751847184" --output brain_multicoil_train_batch_8.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_train_batch_9.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=KTZxD7gXLX%2BNBDhvNKlJiTTllYs%3D&Expires=1751847184" --output brain_multicoil_train_batch_9.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_val_batch_0.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=n88TVQWDFasH37oTjA8ltXBAqCE%3D&Expires=1751847184" --output brain_multicoil_val_batch_0.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_val_batch_1.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=a3iCJ7d4Mux6ZU3P5MC898SlnlE%3D&Expires=1751847184" --output brain_multicoil_val_batch_1.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_val_batch_2.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=2Xt8YGZ53hXuH%2FHhadVDZ5a%2Bxrk%3D&Expires=1751847184" --output brain_multicoil_val_batch_2.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_test_batch_0.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=vk3MLVFkcJcU2yL%2FN7c0QlTGMHw%3D&Expires=1751847184" --output brain_multicoil_test_batch_0.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_test_batch_1.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=DrpodE08ogV92vXFFSAdRLpT9R0%3D&Expires=1751847184" --output brain_multicoil_test_batch_1.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_test_batch_2.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=jRve%2BmUzXDJpaS%2B24oAzqXSjBvA%3D&Expires=1751847184" --output brain_multicoil_test_batch_2.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_test_full_batch_0.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=dx30s7o0Wh5qqYYJn2WAk8Nhl5E%3D&Expires=1751847184" --output brain_multicoil_test_full_batch_0.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_test_full_batch_1.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=ZzgSl34TSAu2loUPvJHk9qD6cQI%3D&Expires=1751847184" --output brain_multicoil_test_full_batch_1.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_multicoil_test_full_batch_2.tar.xz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=XI2xCJWzJIMg7urMgXR3Bkbz5Bk%3D&Expires=1751847184" --output brain_multicoil_test_full_batch_2.tar.xz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v2.0/brain_fastMRI_DICOM.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=s7S6Ns%2Bx0LJ%2BjFLnqn2g1xt%2FJu4%3D&Expires=1751847184" --output brain_fastMRI_DICOM.tar.gz



# prostate

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/labels.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=OueoG8WBD14eLzMwWaSQaXWXFuw%3D&Expires=1751847184" --output labels.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastmri_prostate_DICOMS_IDS_001_312.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=9Ry13PgWh21GOdhFzK%2BucJ2ke0o%3D&Expires=1751847184" --output fastmri_prostate_DICOMS_IDS_001_312.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_001_011.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=MXZd8CqXMcU%2Bl9bOMzb%2BVw0%2FWA8%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_001_011.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_012_022.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=uQeoRS8cRJLW3ZZgOGd9colxQLI%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_012_022.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_023_033.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=ohsAr6%2FpQpUE5o9y6Wb%2BEF3%2FItI%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_023_033.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_034_044.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=p3phNdECINci01jsFksIZJ4zbfk%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_034_044.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_045_055.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=7hmE%2BFNV7N9vyrLSwA5ikxOlPac%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_045_055.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_056_066.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=AaJHxUT8TKDGeSttYASTIr63PH4%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_056_066.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_067_077.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=jaBc%2FwR23WD%2BoXjgzf498WhfedU%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_067_077.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_078_088.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=6JgVGbAOprC6E1LMH%2BnJbaNdW9U%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_078_088.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_089_099.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=ecvRT1rlL%2F806nPwxBatQD7kmyw%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_089_099.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_100_110.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=Ndpc%2FuG99g6kihCdl%2Fjr9fLEEbs%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_100_110.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_111_121.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=CCnAJcQanA5esj6Ytra49R8K8E0%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_111_121.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_122_132.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=bSuDVeBvM0iILVwPal4VhP1r2C8%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_122_132.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_133_142.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=8yf4Ne02BJkprV0r3DJ4vOOODcE%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_133_142.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_143_152.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=C1Su7AwzM54viz5m4hO9uCuSkF4%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_143_152.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_153_162.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=rK5V5Kistsug98rgRwech2NCLlc%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_153_162.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_163_172.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=wIhX5Ib6SpVuKQN0iStr18cTAm8%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_163_172.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_173_182.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=vx3QWn2JAhrXu4pO327UHCPcNao%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_173_182.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_183_192.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=HboLGKl1w3vJaKygvZtgc3tqPRk%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_183_192.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_193_202.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=KuLSwqxjQmT9R0Bxf3vmJNIjEuo%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_193_202.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_203_212.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=m%2BTpZHKSobcgjR9BnO14dAV9mXI%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_203_212.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_213_222.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=M%2FfReFD8MLFV9m%2BXeZlSY8Bl1jo%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_213_222.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_223_232.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=uBaWsDtZmaL119rvQYXCmVCjWpE%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_223_232.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_233_242.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=oB%2FkFCQpB%2FsEKheRJNx7Al1d5tg%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_233_242.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_243_252.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=HBHFJRc2p2yYfeF5ZbT8Tnk0nig%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_243_252.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_253_262.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=WUBu77C7zWQf1DPEzJkvahSU3zk%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_253_262.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_263_272.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=6xP8nMD9J%2BCKbasBuLLelcYup3g%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_263_272.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_273_282.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=57B8149CDpcmn28Rcz9EIChN1zk%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_273_282.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_283_292.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=276g2LGJw0VoapXR9GngGC76eR0%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_283_292.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_293_302.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=9RCOLx3UCI5wHbd%2F9xD2jvQkma0%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_293_302.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_DIFF_IDS_303_312.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=USeqHzUnODYPAsFR%2BkIn91cjxfw%3D&Expires=1751847184" --output fastMRI_prostate_DIFF_IDS_303_312.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_T2_IDS_001_020.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=Q0zSa0cxlAcoy%2BN8CkZ%2FSzf9P0g%3D&Expires=1751847184" --output fastMRI_prostate_T2_IDS_001_020.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_T2_IDS_021_040.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=1xc7dqgy4Q7Jay2FM3PeI%2FNoS6M%3D&Expires=1751847184" --output fastMRI_prostate_T2_IDS_021_040.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_T2_IDS_041_060.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=ggeh1sy8x%2BALBy9IQ1mnHpjbARQ%3D&Expires=1751847184" --output fastMRI_prostate_T2_IDS_041_060.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_T2_IDS_061_080.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=kn61nxq%2Bu%2BrJIUAiBhBPmvMKBxE%3D&Expires=1751847184" --output fastMRI_prostate_T2_IDS_061_080.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_T2_IDS_081_100.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=3g31nLGOYc8RWPyxzqIYNeyRDac%3D&Expires=1751847184" --output fastMRI_prostate_T2_IDS_081_100.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_T2_IDS_101_120.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=FCWlT8UjqR0tSocQ8oHKUpV3eYI%3D&Expires=1751847184" --output fastMRI_prostate_T2_IDS_101_120.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_T2_IDS_121_140.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=ocmdueVAMmz5nKGe%2FvD9NjdVyts%3D&Expires=1751847184" --output fastMRI_prostate_T2_IDS_121_140.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_T2_IDS_141_160.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=VGmqaKEW5B2SvnbQwp%2BrHNd9Rss%3D&Expires=1751847184" --output fastMRI_prostate_T2_IDS_141_160.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_T2_IDS_161_179.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=V3xj1cq%2BpEpG5qndFKNL6HdiXWg%3D&Expires=1751847184" --output fastMRI_prostate_T2_IDS_161_179.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_T2_IDS_180_198.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=YokSoc0lQZzxh37oWuMMnHILhRk%3D&Expires=1751847184" --output fastMRI_prostate_T2_IDS_180_198.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_T2_IDS_199_217.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=HhVzLX%2FAm4IEt5y0BRYw%2FGVOCZg%3D&Expires=1751847184" --output fastMRI_prostate_T2_IDS_199_217.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_T2_IDS_218_236.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=diaq8RJmoUF6dZ1NIq4%2FS4OZH5A%3D&Expires=1751847184" --output fastMRI_prostate_T2_IDS_218_236.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_T2_IDS_237_255.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=Kh50FnXK%2B8tUVwHGPbIK%2FZP1U74%3D&Expires=1751847184" --output fastMRI_prostate_T2_IDS_237_255.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_T2_IDS_256_274.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=HLDXkhtmepSnmvlBFXjcqcxy%2F0U%3D&Expires=1751847184" --output fastMRI_prostate_T2_IDS_256_274.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_T2_IDS_275_293.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=%2F%2BsW22SHb4BJm%2BGAWn7nigDoUiE%3D&Expires=1751847184" --output fastMRI_prostate_T2_IDS_275_293.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/fastMRI_prostate_T2_IDS_294_312.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=m%2BK2tSdR8cvgyG6HnRrwI1tyNXM%3D&Expires=1751847184" --output fastMRI_prostate_T2_IDS_294_312.tar.gz


# breast 

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_001_150_DCM.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=XfCdnHvKKSQrEbsWQvLL0XdgnKE%3D&Expires=1751847184" --output fastMRI_breast_IDS_001_150_DCM.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_150_300_DCM.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=NoehpEd%2Fw%2FwBp2LwE8DWtlje%2BNM%3D&Expires=1751847184" --output fastMRI_breast_IDS_150_300_DCM.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_001_010.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=qSdlZwOkvvHLmGrDSpdOup%2BsYlM%3D&Expires=1751847184" --output fastMRI_breast_IDS_001_010.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_011_020.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=cVLzUtXTZ10X995diuYCRraB7%2BQ%3D&Expires=1751847184" --output fastMRI_breast_IDS_011_020.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_021_030.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=%2B4yjNiKN%2BY%2FxNf0LP4qFs4B85TA%3D&Expires=1751847184" --output fastMRI_breast_IDS_021_030.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_031_040.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=avENwrUoHN%2BFSERxVoZhuXJst3Q%3D&Expires=1751847184" --output fastMRI_breast_IDS_031_040.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_041_050.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=l%2FMMfKWFy4XAqBuWQukw2yTN800%3D&Expires=1751847184" --output fastMRI_breast_IDS_041_050.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_051_060.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=tZCGt8mmJG6sw2zAGcjnrV%2F%2Fi0s%3D&Expires=1751847184" --output fastMRI_breast_IDS_051_060.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_061_070.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=eYrnvHnGCj%2FyQoumrL6yBkWmfYA%3D&Expires=1751847184" --output fastMRI_breast_IDS_061_070.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_071_080.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=uxS7OZi8FH5DGh3Rn0Yco3wARJc%3D&Expires=1751847184" --output fastMRI_breast_IDS_071_080.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_081_090.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=nCdj7SWylZdrrUC137584G9oBM0%3D&Expires=1751847184" --output fastMRI_breast_IDS_081_090.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_091_100.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=B3qRqVUbOPWdqaUqNlnaHN63SO8%3D&Expires=1751847184" --output fastMRI_breast_IDS_091_100.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_101_110.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=AF%2FsECbAoZp0vkT16U3bu7zAdkA%3D&Expires=1751847184" --output fastMRI_breast_IDS_101_110.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_111_120.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=bVX5zfTwGtE3wDVUwKcC4%2BBj7wo%3D&Expires=1751847184" --output fastMRI_breast_IDS_111_120.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_121_130.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=0HsS%2FXChhBTL1ZNknwUfpcOpVM4%3D&Expires=1751847184" --output fastMRI_breast_IDS_121_130.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_131_140.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=6b5wKOB2mU1hWb12cTTBQKkdQa0%3D&Expires=1751847184" --output fastMRI_breast_IDS_131_140.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_141_150.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=c2KMNFq1zoLiYSxNXfrWtxk1G5E%3D&Expires=1751847184" --output fastMRI_breast_IDS_141_150.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_151_160.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=98eOuFf0A%2BgeTPlg1hqq6CX85sQ%3D&Expires=1751847184" --output fastMRI_breast_IDS_151_160.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_161_170.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=WS48UUU6HXwOCvtPApf1AGKqobI%3D&Expires=1751847184" --output fastMRI_breast_IDS_161_170.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_171_180.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=tGCy9%2BY%2FlkCSeU%2BuiGqqN%2BuzzYY%3D&Expires=1751847184" --output fastMRI_breast_IDS_171_180.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_181_190.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=%2BHpjPh5pLj7kEeRfTnX5S4upFJI%3D&Expires=1751847184" --output fastMRI_breast_IDS_181_190.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_191_200.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=8YhEyXtERKvZ3W0SHG%2FmJBwScGQ%3D&Expires=1751847184" --output fastMRI_breast_IDS_191_200.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_201_210.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=WOyVByUEs8MZ2Zi%2B4rmrXVmJBjM%3D&Expires=1751847184" --output fastMRI_breast_IDS_201_210.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_211_220.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=jYsI8kM2riAXjjObz7WAKFxVdOE%3D&Expires=1751847184" --output fastMRI_breast_IDS_211_220.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_221_230.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=I4Y%2FBKHoeX%2FiR%2FDtycRA9zdI%2F2c%3D&Expires=1751847184" --output fastMRI_breast_IDS_221_230.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_231_240.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=5uwsiLsZCuCniCMkSwmNFx8%2FCwY%3D&Expires=1751847184" --output fastMRI_breast_IDS_231_240.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_241_250.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=BMoaFmhfD5XyDZnhawY72QdOtcc%3D&Expires=1751847184" --output fastMRI_breast_IDS_241_250.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_251_260.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=UGZvtcSBufqUyaCaDB7BhWBCShg%3D&Expires=1751847184" --output fastMRI_breast_IDS_251_260.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_261_270.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=WSSM0sPPmSezqXIFoSRXesoCTsU%3D&Expires=1751847184" --output fastMRI_breast_IDS_261_270.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_271_280.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=rM2DFBcyB9i7iLUrcfXQYgBduSc%3D&Expires=1751847184" --output fastMRI_breast_IDS_271_280.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_281_290.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=EPApQnsW4rkJr%2BKYUdrpDmpwcBI%3D&Expires=1751847184" --output fastMRI_breast_IDS_281_290.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_IDS_291_300.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=bmQpFbf5XjGk9QUPmTGk4MRE4kQ%3D&Expires=1751847184" --output fastMRI_breast_IDS_291_300.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v4.0/fastMRI_breast_labels.tar.gz?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=mgj%2BKJTiG0MwIoBhstkF8%2FzwnQ4%3D&Expires=1751847184" --output fastMRI_breast_labels.tar.gz

curl -C - "https://fastmri-dataset.s3.amazonaws.com/v3.0/SHA256?AWSAccessKeyId=AKIAJM2LEZ67Y2JL3KRA&Signature=OwpQbqUqIZH2ZGmPYuFD3LVQoUM%3D&Expires=1751847184" --output SHA256



bash ~/mailsender.sh

