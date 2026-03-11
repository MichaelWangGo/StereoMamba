import os
import argparse
# from PIL import Image

from tqdm import tqdm
# import numpy as np
import wandb
import torch
import torch.optim as optim
# import torchvision.transforms as tv_transforms
from torch.utils.data import DataLoader

from utils import Params, set_random_seed, RunningAvg
from utils.error_metrics import disparity_epe, disparity_bad3
from losses.disparity import model_loss
from models.StereoMamba import StereoMamba
from datasets import __datasets__
from utils.iotools import *


parser = argparse.ArgumentParser(description= 'fully supervised disparity training')
parser.add_argument('--default_config', help= '.json file containing hyperparameter configuration')
parser.add_argument('--cuda_id', help='gpu number you want to use.', default=0, type=int)


if __name__ == "__main__":

    args, unknown = parser.parse_known_args()
    lower_eval_loss=None
    force_log=False
    config = Params(args.default_config)
    wandb.init(config=config.dict, project=config.project, name=config.experiment_tag)
    config = wandb.config

    if config.device !='cpu':
        print(config.device)
        if args.cuda_id is None:
            exit()
        os.environ["CUDA_VISIBLE_DEVICES"]=str(args.cuda_id)
    
    if config.seed:
        set_random_seed(config.seed)
   
    # load data 
    StereoDataset = __datasets__[config.dataset]
    train_dataset = StereoDataset(config.data_path, config.trainlist, True)
    test_dataset = StereoDataset(config.data_path, config.testlist, False, eval_pad_to=16)
    TrainImgLoader = DataLoader(train_dataset, config.batch_size, shuffle=True, num_workers=8, drop_last=True)
    TestImgLoader = DataLoader(test_dataset, config.test_batch_size, shuffle=False, num_workers=4, drop_last=False)
    
    model = StereoMamba(patch_size=config.patch_size, in_chans=config.in_chans, depths=config.depths, 
                       dims=config.dims, ssm_d_state=config.ssm_d_state, ssm_ratio=config.ssm_ratio,
                       ssm_dt_rank=config.ssm_dt_rank, ssm_act_layer=config.ssm_act_layer, ssm_conv=config.ssm_conv,
                       ssm_conv_bias=config.ssm_conv_bias, ssm_drop_rate=config.ssm_drop_rate, ssm_init=config.ssm_init,
                       forward_type=config.forward_type, mlp_ratio=config.mlp_ratio, mlp_act_layer=config.mlp_act_layer,
                       mlp_drop_rate=config.mlp_drop_rate, gmlp=config.gmlp, drop_path_rate=config.drop_path_rate,
                       patch_norm=config.patch_norm, norm_layer=config.norm_layer, downsample_version=config.downsample_version,
                       patchembed_version=config.patchembed_version, use_checkpoint=config.use_checkpoint, posembed=config.posembed,
                       imgsize=config.imgsize, max_disparity=config.max_disparity, use_concat_volume=config.use_concat_volume,
                       cross_attn=config.cross_attn, d_model=config.d_model, d_state=config.d_state, d_conv=config.d_conv,
                       expand=config.expand, headdim=config.headdim, ngroups=config.ngroups)
    print("Parameter Count: %d" % count_parameters(model))
    if config.load_model:
        print('loading pretrained model: {}'.format(config.load_model))
        try:
            model.load_state_dict(torch.load(config.load_model))
        except:
            pass
        finally:
            print('loading only disparity weights, the rest of the model has random weights')
            model.load_state_dict(torch.load(config.load_model), strict=False)

    model = model.to(config.device)
    scaler = torch.cuda.amp.GradScaler(enabled=config.use_amp)
    optimizer = optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    if config.use_onecycle_lr:
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=config.lr, steps_per_epoch=len(TrainImgLoader), epochs=config.epochs)

    for epoch in tqdm(range(config.epochs), desc= 'training_progress'):
        
        train_epoch_loss_acc = RunningAvg()
        eval_epoch_loss_acc = RunningAvg()
        epe_error_acc = RunningAvg()
        bad3_error_acc = RunningAvg()

        model.train()
        for batch in tqdm(TrainImgLoader, total=len(TrainImgLoader), desc='training (epoch:{:04d})'.format(epoch), leave=False):
            left, right, reference_disparity = batch['left'], batch['right'], batch['disparity']
            mask = (reference_disparity > 0) & (reference_disparity < config.max_disparity)
            mask = mask.to(config.device)
            left = left.to(config.device)
            right = right.to(config.device)
            reference_disparity = reference_disparity.to(config.device)
            reference_disparity = torch.squeeze(reference_disparity, 1)
            optimizer.zero_grad()
            with torch.cuda.amp.autocast(enabled=config.use_amp):
                disparity_scales = model(left, right)

                training_loss = model_loss(disparity_scales, reference_disparity, mask)
                # print('training loss:', training_loss)
            scaler.scale(training_loss).backward()
            scaler.step(optimizer)
            scaler.update()

            train_epoch_loss_acc.append(training_loss.detach())
            del training_loss
            if config.use_onecycle_lr:
                scheduler.step()
            if config.overfit:
                break
        
        model.eval()
        with torch.no_grad():
            for batch in tqdm(TestImgLoader, total =len(TestImgLoader), desc='evaluating (epoch:{:04d})'.format(epoch),leave=False):
                left, right, reference_disparity = batch['left'], batch['right'], batch['disparity']
                mask = (reference_disparity > 0) & (reference_disparity < config.max_disparity)
                mask = mask.to(config.device)
                left = left.to(config.device)
                right = right.to(config.device)
                reference_disparity = reference_disparity.to(config.device)
                reference_disparity = torch.squeeze(reference_disparity, 1)
                with torch.cuda.amp.autocast(enabled=config.use_amp):
                    disparity_scales = model(left, right)

                    eval_loss = model_loss(disparity_scales, reference_disparity, mask)
                    # left_features3 = left_features[0][0][0:3]
                

                eval_epoch_loss_acc.append(eval_loss.detach())
                
                epe = disparity_epe(disparity_scales[-1], reference_disparity, max_disparity=config.max_disparity)
                bad3 = disparity_bad3(disparity_scales[-1], reference_disparity, max_disparity=config.max_disparity)
                epe_error_acc.append(epe.detach())
                bad3_error_acc.append(bad3.detach())

                del eval_loss, epe, bad3
                if config.overfit:
                    break

        current_lr = optimizer.param_groups[0]['lr']
        
        
        tqdm.write(' epoch: {:05d}\t lr: {:.02E}\t training_loss: {:.03f}\t evaluation_loss:{:.03f}\t epe: {:.03f}\t bad3: {:.02f}'.format(epoch,
                                                                                                        current_lr,
                                                                                                        train_epoch_loss_acc.get_val(),
                                                                                                        eval_epoch_loss_acc.get_val(),
                                                                                                        epe_error_acc.get_val(),
                                                                                                        bad3_error_acc.get_val()))
        if lower_eval_loss is None:
            lower_eval_loss = eval_epoch_loss_acc.get_val()
        elif lower_eval_loss > eval_epoch_loss_acc.get_val():
            lower_eval_loss = eval_epoch_loss_acc.get_val()
            # torch.save(model.state_dict(), str(config.experiment_tag+'_lowest_eval_loss.pt'))
            lowest_eval_save_path = os.path.join(config.ckpt_path, f"{config.experiment_tag}_lowest_eval_loss.pt")
            os.makedirs(config.ckpt_path, exist_ok=True)
            torch.save(model.state_dict(), lowest_eval_save_path)
            tqdm.write(f'lowest_eval_loss={eval_epoch_loss_acc.get_val()} at epoch {epoch}')
            force_log=True
        
        if ((epoch+1)%config.wandb_log == 0) or force_log:

            # Convert tensors to CPU numpy arrays with proper formatting before passing to wandb.Image
            gt_disp_numpy = reference_disparity[0].detach().cpu().numpy()
            pred_disp_numpy = disparity_scales[-1][0].detach().cpu().numpy()
            
            # Create log dictionary with properly formatted images
            log_dict = {
                'learning_rate': optimizer.param_groups[0]['lr'],
                'Loss/Training': train_epoch_loss_acc.get_val(),
                'Loss/Evaluation': eval_epoch_loss_acc.get_val(),
                'Error/Epe': epe_error_acc.get_val(),
                'Error/Bad3': bad3_error_acc.get_val(),
                "sample": [
                    wandb.Image(gt_disp_numpy, caption="gt"),
                    wandb.Image(pred_disp_numpy, caption="prediction_f"),
                ]
            }
            wandb.log(log_dict, step=epoch+1)
            force_log=False
            
        
        if (epoch+1)%config.model_save_period == 0:
            save_path = os.path.join(config.ckpt_path, f"{config.experiment_tag}_epoch_{epoch+1}_error_{epe_error_acc.get_val().item():.3f}.pt")
            os.makedirs(config.ckpt_path, exist_ok=True)
            # torch.save(model.state_dict(), str(config.experiment_tag+'_epoch_'+str(epoch+1) + '__error_' + str(epe_error_acc.get_val().item()) + '.pt'))
            torch.save(model.state_dict(), save_path)

    # save final model
    # torch.save(model.state_dict(), config.experiment_tag+'_lr'+str(config.lr)+'_final_error__' + str(epe_error_acc.get_val().item()) + '.pt')
    # save final model
    final_save_path = os.path.join(config.ckpt_path, f"{config.experiment_tag}_lr{config.lr}_final_error_{epe_error_acc.get_val().item():.3f}.pt")
    torch.save(model.state_dict(), final_save_path)
