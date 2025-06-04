from cdfvd import fvd
import torch

ckpt_path = '/path/to/vit_g_hybrid_pt_1200e_ssv2_ft.pth'
assert ckpt_path != '/path/to/vit_g_hybrid_pt_1200e_ssv2_ft.pth', 'You need to change the detector_path to the real path where the vit_g_hybrid_pt_1200e_ssv2_ft.pth is.'
torch.cuda.set_device(0)
evaluator = fvd.cdfvd('videomae', ckpt_path=ckpt_path)
evaluator.compute_real_stats(evaluator.load_videos('/path/to/video_dataset', data_type='video_folder'))
evaluator.compute_fake_stats(evaluator.load_videos('/path/to/generated_video', data_type='video_folder'))
score = evaluator.compute_fvd_from_stats()
