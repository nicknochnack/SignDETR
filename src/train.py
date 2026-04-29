from data import DETRData
from model import DETR
from loss import DETRLoss, HungarianMatcher
from torch.utils.data import DataLoader 
from torch import optim, load, save
from colorama import Fore  
from utils.logger import get_logger
from utils.rich_handlers import TrainingHandler, rich_training_context
import sys 
import torch
from utils.boxes import stacker

if __name__ == '__main__': 
    # Initialize logger and handlers
    logger = get_logger("training")
    logger.print_banner()
    
    train_dataset = DETRData('data/train') 
    train_dataloader = DataLoader(train_dataset, batch_size=4, collate_fn=stacker, drop_last=True) 

    test_dataset = DETRData('data/test', train=False) 
    test_dataloader = DataLoader(test_dataset, batch_size=4, collate_fn=stacker, drop_last=True) 

    num_classes = 2 
    model = DETR(num_classes=num_classes)
    #model.load_pretrained('checkpoints/')
    #prev pretrained/4426_model.pt changed to checkpoints/49_model.pt
    model.log_model_info()
    model.train() 

    opt = optim.Adam(model.parameters(), lr=1e-5)
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(opt, len(train_dataloader)*30, T_mult=2)

    weights= {'class_weighting': 1, 'bbox_weighting': 5, 'giou_weighting': 2}
    matcher = HungarianMatcher(weights)
    criterion = DETRLoss(num_classes=num_classes, matcher=matcher, weight_dict=weights, eos_coef=0.1)

    train_batches = len(train_dataloader)
    test_batches = len(test_dataloader)
    epochs = 50
    #prev 100 changed to 40 for the demo training.
    
    # Log training configuration
    training_config = {
        "Total Epochs": epochs,
        "Batch Size": 4,
        "Train Batches": train_batches,
        "Test Batches": test_batches,
        "Learning Rate": 1e-5,
        "Optimizer": "Adam",
        "Scheduler": "CosineAnnealingWarmRestarts"
    }
    logger.print_table("🏋️ Training Configuration", list(training_config.keys()), [list(training_config.values())])
    
    # Start training with rich context
    with rich_training_context() as training_handler:
        for epoch in range(epochs): 
            with training_handler.create_training_progress() as epoch_progress:
                epoch_task = epoch_progress.add_task(f"[bold blue] Progress {epoch+1}/{epochs}", train_loss=0.0, test_loss=0.0, total=train_batches)
                # Training phase
                model.train()
                train_epoch_loss = 0.0 
            
                # Create progress bar for current epoch
                for batch_idx, batch in enumerate(train_dataloader): 
                    X, y = batch
                    try: 
                        yhat = model(X) 
                        yhat_classes = yhat['pred_logits'] 
                        yhat_bb = yhat['pred_boxes'] 
                        loss_dict = criterion(yhat, y) 
                        weight_dict = criterion.weight_dict
                        
                        # Ensure we sum exactly over the expected weighted keys, and keep tensor dtype
                        losses = loss_dict['labels']['loss_ce']*weight_dict['class_weighting'] + loss_dict['boxes']['loss_bbox']*weight_dict['bbox_weighting'] + loss_dict['boxes']['loss_giou']*weight_dict['giou_weighting']
                        
                        # Calculate loss 
                        train_epoch_loss += losses.item() 
                        
                        # Zero grads
                        opt.zero_grad()
                        
                        # Backward
                        losses.backward()
                        # Apply
                        opt.step()
                        
                        # Update progress
                        epoch_progress.update(epoch_task, advance=1, train_loss=round(train_epoch_loss/train_batches,5))
                        
                    except Exception as e: 
                        logger.error(f"Training error at epoch {epoch}, batch {batch_idx}: {str(e)}")
                        logger.error(f"Batch targets: {str(y)}")
                        sys.exit()
            
                # Progress lr 
                scheduler.step()
            
                # Test phase
                model.eval()
                test_epoch_loss = 0.0
                with torch.no_grad():
                    for batch_idx, batch in enumerate(test_dataloader):
                        X, y = batch
                        yhat = model(X)
                        loss_dict = criterion(yhat, y) 
                        weight_dict = criterion.weight_dict
                        losses = loss_dict['labels']['loss_ce']*weight_dict['class_weighting'] + loss_dict['boxes']['loss_bbox']*weight_dict['bbox_weighting'] + loss_dict['boxes']['loss_giou']*weight_dict['giou_weighting']
                        
                        # Calculate loss 
                        test_epoch_loss += losses.item() 
                        epoch_progress.update(epoch_task, advance=0, test_loss=round(test_epoch_loss/test_batches,5))
                
                # Save checkpoints
                if epoch % 10 == 0 and epoch != 0: 
                    checkpoint_path = f"checkpoints/{epoch}_model.pt"
                    save(model.state_dict(), checkpoint_path)
                    training_handler.save_checkpoint_status(checkpoint_path, epoch)
            
    # Final save
    save(model.state_dict(), f"checkpoints/{epoch}_model.pt")