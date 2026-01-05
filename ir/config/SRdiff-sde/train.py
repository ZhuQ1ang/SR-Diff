import argparse
import logging
import math
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
import sys

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import options as option
from models import create_model
import warnings
warnings.filterwarnings("ignore", category=FutureWarning, message="Importing from timm.models.layers is deprecated")
sys.path.insert(0, "../../")

import utils as util
from data import create_dataloader, create_dataset
from data.data_sampler import DistIterSampler
from PIL import Image

import time
from BLIPFeatureExtractor import BLIPFeatureExtractor
from CLSEnhancer import  CLSEnhancer


def init_dist(args,backend="nccl", **kwargs):
    """ initialization for distributed training"""
    # if mp.get_start_method(allow_none=True) is None:
    if (
        mp.get_start_method(allow_none=True) != "spawn"
    ):  # Return the name of start method used for starting processes
        mp.set_start_method("spawn", force=True)  ##'spawn' is the default on Windows
    #rank = int(os.environ["RANK"])  # system env process ranks
    rank = int(os.environ.get("RANK", args.local_rank))
    num_gpus = torch.cuda.device_count()  # Returns the number of GPUs available
    torch.cuda.set_device(rank % num_gpus)
    # dist.init_process_group(
    #     backend=backend, **kwargs
    # )  # Initializes the default distributed process group
    dist.init_process_group(backend="nccl")


def main():
    avgtime = 0
    cnt = 0
    #### setup options of three networks
    parser = argparse.ArgumentParser()
    parser.add_argument("-opt", type=str, help="Path to option YMAL file.")
    parser.add_argument(
        "--launcher", choices=["none", "pytorch"], default="none", help="job launcher"
    )
    parser.add_argument("--local-rank", type=int, default=0)
    args = parser.parse_args()
    opt = option.parse(args.opt, is_train=True)

    # convert to NoneDict, which returns None for missing keys
    opt = option.dict_to_nonedict(opt)

    # choose small opt for SFTMD test, fill path of pre-trained model_F
    #### set random seed
    seed = opt["train"]["manual_seed"]

    #### distributed training settings
    if args.launcher == "none":  # disabled distributed training
        opt["dist"] = False
        opt["dist"] = False
        rank = -1
        print("Disabled distributed training.")
    else:
        opt["dist"] = True
        opt["dist"] = True
        init_dist(args)
        world_size = (
            torch.distributed.get_world_size()
        )  # Returns the number of processes in the current process group
        rank = torch.distributed.get_rank()  # Returns the rank of current process group
        # util.set_random_seed(seed)

    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    ###### Predictor&Corrector train ######

    #### loading resume state if exists
    if opt["path"].get("resume_state", None):
        # distributed resuming: all load into default GPU
        device_id = torch.cuda.current_device()
        resume_state = torch.load(
            opt["path"]["resume_state"],
            map_location=lambda storage, loc: storage.cuda(device_id),
        )
        option.check_resume(opt, resume_state["iter"])  # check resume options
    else:
        resume_state = None

    #### mkdir and loggers
    if rank <= 0:  # normal training (rank -1) OR distributed training (rank 0-7)
        if resume_state is None:
            # Predictor path
            util.mkdir_and_rename(
                opt["path"]["experiments_root"]
            )  # rename experiment folder if exists
            util.mkdirs(
                (
                    path
                    for key, path in opt["path"].items()
                    if not key == "experiments_root"
                    and "pretrain_model" not in key
                    and "resume" not in key
                    and "daclip" not in key
                )
            )
            #os.system("rm ./log")
            #os.symlink(os.path.join(opt["path"]["experiments_root"], ".."), "./log")

        # config loggers. Before it, the log will not work
        util.setup_logger(
            "base",
            opt["path"]["log"],
            "train_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        util.setup_logger(
            "val",
            opt["path"]["log"],
            "val_" + opt["name"],
            level=logging.INFO,
            screen=False,
            tofile=True,
        )
        logger = logging.getLogger("base")
        logger.info(option.dict2str(opt))
        # tensorboard logger
        if opt["use_tb_logger"] and "debug" not in opt["name"]:
            version = float(torch.__version__[0:3])
            if version >= 1.1:  # PyTorch 1.1
                from torch.utils.tensorboard import SummaryWriter
            else:
                logger.info(
                    "You are using PyTorch {}. Tensorboard will use [tensorboardX]".format(
                        version
                    )
                )
                from tensorboardX import SummaryWriter
            tb_logger = SummaryWriter(log_dir="log/{}/tb_logger/".format(opt["name"]))
    else:
        util.setup_logger(
            "base", opt["path"]["log"], "train", level=logging.INFO, screen=False
        )
        logger = logging.getLogger("base")

    
    #### create train and val dataloader
    dataset_ratio = 200  # enlarge the size of each epoch
    for phase, dataset_opt in opt["datasets"].items():
        if phase == "train":
            train_set = create_dataset(dataset_opt)
            train_size = int(math.ceil(len(train_set) / dataset_opt["batch_size"]))
            total_iters = int(opt["train"]["niter"])
            total_epochs = int(math.ceil(total_iters / train_size))
            if opt["dist"]:
                train_sampler = DistIterSampler(
                    train_set, world_size, rank, dataset_ratio
                )
                total_epochs = int(
                    math.ceil(total_iters / (train_size * dataset_ratio))
                )
            else:
                train_sampler = None
            train_loader = create_dataloader(train_set, dataset_opt, opt, train_sampler)
            if rank <= 0:
                logger.info(
                    "Number of train images: {:,d}, iters: {:,d}".format(
                        len(train_set), train_size
                    )
                )
                logger.info(
                    "Total epochs needed: {:d} for iters {:,d}".format(
                        total_epochs, total_iters
                    )
                )
        elif phase == "val":
            val_set = create_dataset(dataset_opt)
            val_loader = create_dataloader(val_set, dataset_opt, opt, None)
            if rank <= 0:
                logger.info(
                    "Number of val images in [{:s}]: {:d}".format(
                        dataset_opt["name"], len(val_set)
                    )
                )
        else:
            raise NotImplementedError("Phase [{:s}] is not recognized.".format(phase))
    assert train_loader is not None
    assert val_loader is not None
    restormodel = CLSEnhancer(input_dim=768, hidden_dim=1024, num_layers=4, num_heads=8).cuda()
    checkpoint = torch.load(r"D:\OneDrive\Desktop\The visual computer\sr-diff\sr\best_model.pth", map_location='cuda')
    restormodel.load_state_dict(checkpoint)
    #### create model
    model = create_model(opt) 
    device = model.device
    
   

    #### resume training
    if resume_state:
       
        logger.info(
            "Resuming training from epoch: {}, iter: {}.".format(
                resume_state["epoch"], resume_state["iter"]
            )
        )

        start_epoch = resume_state["epoch"]
        current_step = resume_state["iter"]
        model.resume_training(resume_state)  # handle optimizers and schedulers
    else:
        current_step = 0
        start_epoch = 0
    print("-----")
    sde = util.IRSDE(max_sigma=opt["sde"]["max_sigma"], T=opt["sde"]["T"], schedule=opt["sde"]["schedule"], eps=opt["sde"]["eps"], device=device)
    sde.set_model(model.model)

    scale = opt['degradation']['scale']
    print("-----")
    #### training
    logger.info(
        "Start training from epoch: {:d}, iter: {:d}".format(start_epoch, current_step)
    )

    best_psnr = 0.0
    best_iter = 0
    error = mp.Value('b', False)
    epoch_loss = 0.0
    cnt = 0
    os.makedirs('image', exist_ok=True)
    
    for epoch in range(start_epoch, total_epochs * 10 + 1):
        
        if opt["dist"]:
            
            train_sampler.set_epoch(epoch)
        
        if cnt != 0:
            loss_value = epoch_loss/cnt
            print("<epoch>:{:3d}avg_epoch_loss{:.6e}".format(epoch,loss_value))
            file_path = 'epoch_loss_log.txt'  # 定义文件名

            
            with open(file_path, 'a') as f:
                f.write("<epoch>:{:3d}  avg_epoch_loss:{:.6e}\n".format(epoch, loss_value))
        epoch_loss = 0.0
        cnt = 0
        
        for _, train_data in enumerate(train_loader):
            current_step += 1

            if current_step > total_iters:
                break
        
            LQ, GT,lq_paths = train_data["LQ"], train_data["GT"], train_data["LQ_path"]
            
            with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                cls_embeddings = []
                for i,LQ_path in enumerate(lq_paths):
                    save_name = os.path.splitext(os.path.basename(LQ_path))[0]
                    save_name = save_name + '.pt'
                    
                    save_path = os.path.join(r"D:\OneDrive\Desktop\The visual computer\sr-diff\datasets\train\hazy\LQCLS", save_name) #save cls
    
                    if not os.path.exists(save_path):
                        with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                            BLIPFE = BLIPFeatureExtractor()
                            image_ = Image.open(LQ_path).convert("RGB")
                            cls_embedding = BLIPFE.get_cls_embedding(image_)
                            torch.save(cls_embedding, save_path)
                            cls_embedding = cls_embedding.to(device)
                           
                    else:
                        cls_embedding_path = save_path

                        cls_embedding = torch.load(cls_embedding_path).half().to(device)

                    
                    cls_embeddings.append(cls_embedding)   
                #cls_embeddings = cls_embeddings.to(device)
                cls_embeddings = torch.stack(cls_embeddings, dim=0).to(device)
                cls_embeddings = cls_embeddings.squeeze(1).to(device) 
                image_context = cls_embeddings.to(device)
                image_context = image_context.float()
                #image_context = restormodel(image_context)
            
            
            timesteps, states = sde.generate_random_states(x0=GT, mu=LQ)
            
            model.feed_data(states, LQ, GT,image_context=image_context) # xt, mu, x0
            model.optimize_parameters(current_step, timesteps, sde)
            model.update_learning_rate(
                current_step, warmup_iter=opt["train"]["warmup_iter"]
            )
            # 显式释放中间变量显存
            del LQ, GT, image_context, cls_embeddings, train_data, states, timesteps
            torch.cuda.empty_cache()


            if current_step % opt["logger"]["print_freq"] == 0:
                logs = model.get_current_log()
                
                message = "<epoch:{:3d}, iter:{:8,d}, lr:{:.3e}> ".format(
                    epoch, current_step, model.get_current_learning_rate()
                )
                for k, v in logs.items():
                    epoch_loss += v
                    cnt += 1
                    message += "{:s}: {:.4e} ".format(k, v)
                    # tensorboard logger
                    if opt["use_tb_logger"] and "debug" not in opt["name"]:
                        if rank <= 0:
                            tb_logger.add_scalar(k, v, current_step)
                if rank <= 0:
                    logger.info(message)
            
            # validation, to produce ker_map_list(fake)
            if current_step % opt["train"]["val_freq"] == 0 and rank <= 0: #只在rank0上验证
                model.model.eval()

            
                torch.cuda.empty_cache()
                avg_psnr = 0.0
                idx = 0
                for _, val_data in enumerate(val_loader):

                    LQ, GT,lq_paths = val_data["LQ"], val_data["GT"], val_data["LQ_path"]
                    # print(LQ.shape)
                    # H, W = LQ.shape
                    # rnd_h = random.randint(0, max(0, H - 600))
                    # rnd_w = random.randint(0, max(0, W - 600))
                    # LQ = LQ[rnd_h : rnd_h + 600, rnd_w : rnd_w + 600, :]
                    def center_crop_half(tensor):
                        """对形状 (B, C, H, W) 的张量，中心裁剪一半宽高"""
                        B, C, H, W = tensor.shape
                        new_H, new_W = H // 2, W // 2
                        start_H = (H - new_H) // 2
                        start_W = (W - new_W) // 2
                        return tensor[:, :, start_H:start_H + new_H, start_W:start_W + new_W]
                    # LQ = center_crop_half(LQ)
                    # GT = center_crop_half(GT)
                    LQ = torch.rand(1, 3, 720, 720, device=device)
                    GT = torch.rand(1, 3, 720, 720, device=device)

                    with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                        BLIPFE = BLIPFeatureExtractor()
                        cls_embeddings = []
                        start = time.time()
                        for i,lq_path in enumerate(lq_paths):
                            save_name = os.path.splitext(os.path.basename(lq_path))[0]
                            save_name = save_name + '.pt'
                            save_path = os.path.join(r"D:\OneDrive\Desktop\The visual computer\sr-diff\datasets\val\hazy\RestoreCLS", save_name)
                            usepth = 0
                            if not os.path.exists(save_path) or usepth == 0:
                                with torch.no_grad(), torch.amp.autocast(device_type='cuda', dtype=torch.float16):
                                    
                                    BLIPFE = BLIPFeatureExtractor()
                                    image_ = Image.open(lq_path).convert("RGB")
                                    time_clsbegin = time.time()
                                    cls_embedding = BLIPFE.get_cls_embedding(LQ)
                                    cls_embedding = restormodel(cls_embedding.to(device))
                                    time_clsend  =time.time()
                                    print("cls+ srm运行时间:", time_clsend - time_clsbegin, "秒")
                                    torch.save(cls_embedding, save_path)
                                    cls_embedding = cls_embedding.to(device)
                                    print("not exist")
                            else:
                                cls_embedding_path = save_path
                                cls_embedding = torch.load(cls_embedding_path).to('cuda')
                                cls_embedding = cls_embedding.to(device)
                            cls_embeddings.append(cls_embedding)   

                        #cls_embeddings = cls_embeddings.to(device)
                        cls_embeddings = torch.stack(cls_embeddings, dim=0).to(device)
                        cls_embeddings = cls_embeddings.squeeze(1).to(device)    
                        image_context = cls_embeddings.to(device)
                        image_context = image_context.float()
                        #image_context = restormodel(image_context)
                    
                    

                    # valid Predictor
                    best_psnr_run = -1.0
                    #for run in range(1):
                          # 记录 10 次测试里最好的 PSNR
                    print(LQ.shape)
                    noisy_state = sde.noise_state(LQ)
                    model.feed_data(noisy_state, LQ, GT,image_context=image_context)
                    model.test(sde)
                    visuals = model.get_current_visuals()
                    end = time.time()
                    avgtime += end-start
                    cnt += 1
                    print("运行时间:", end - start, "秒")
                    if cnt == 10:
                        print("avgtime：",avgtime/10)
                    output = util.tensor2img(visuals["Output"].squeeze())  # uint8
                    gt_img = util.tensor2img(GT.squeeze())  # uint8
                    lq_img = util.tensor2img(LQ.squeeze())
                    psnr_single = util.calculate_psnr(output, gt_img)
                    if psnr_single > best_psnr_run:
                        best_psnr_run = psnr_single
                        util.save_img(output, f'hazyuselq/hazy-{idx+1:03d}.png')
                    # util.save_img(gt_img, f'image/{idx}_GT.png')
                    # util.save_img(lq_img, f'image/{idx}_LQ.png')

                    # calculate PSNR
                        
                    print(best_psnr_run)
                    avg_psnr += best_psnr_run
                    idx += 1

                    if idx > 100:
                        break

                avg_psnr = avg_psnr / idx

                if avg_psnr > best_psnr:
                    best_psnr = avg_psnr
                    best_iter = current_step
                # if opt["dist"]:
                #     dist.barrier()

                # log
                logger.info("# Validation # PSNR: {:.6f}, Best PSNR: {:.6f}| Iter: {}".format(avg_psnr, best_psnr, best_iter))
                logger_val = logging.getLogger("val")  # validation logger
                logger_val.info(
                    "<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    )
                )
                print("<epoch:{:3d}, iter:{:8,d}, psnr: {:.6f}".format(
                        epoch, current_step, avg_psnr
                    ))
                # tensorboard logger
                if opt["use_tb_logger"] and "debug" not in opt["name"]:
                    tb_logger.add_scalar("psnr", avg_psnr, current_step)
                #if current_step % opt["train"]["val_freq"] == 0:    
                

            if error.value:
                sys.exit(0)
            if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
                    if rank <= 0:
                        logger.info("Saving models and training states.")
                        model.save(current_step)
                        model.save_training_state(epoch, current_step)
            #### save models and training states
            # if current_step % opt["logger"]["save_checkpoint_freq"] == 0:
            #     if rank <= 0:
            #         logger.info("Saving models and training states.")
            #         model.save(current_step)
            #         model.save_training_state(epoch, current_step)

    if rank <= 0:
        logger.info("Saving the final model.")
        model.save("latest")
        logger.info("End of Predictor and Corrector training.")
    tb_logger.close()


if __name__ == "__main__":
    main()
