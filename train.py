import types
from torch.utils.data import Dataset, DataLoader, Subset
from utils_ours import *
from loss import LossComputer

from pytorch_transformers import AdamW, WarmupLinearSchedule

def run_epoch(epoch, model, optimizer, loader, loss_computer, logger, csv_logger, args,
              is_training, show_progress=False, log_every=50, scheduler=None):
    """
    scheduler is only used inside this function if model is bert.
    """
    if args.reduce_lr_manually == 1:
        if epoch == 50:
            for g in optimizer.param_groups:
                g['lr'] = g['lr']/10
        elif epoch == 270:
            for g in optimizer.param_groups:
                g['lr'] = g['lr']/10

    if is_training:
        model.train()
        if args.model == 'bert':
            model.zero_grad()
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):

            batch = tuple(t.cuda(args.gpu_num) for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            # if args.model == 'bert':
            
            #     input_ids = x[:, :, 0]
            #     input_masks = x[:, :, 1]
            #     segment_ids = x[:, :, 2]
            #     outputs = model(
            #         input_ids=input_ids,
            #         attention_mask=input_masks,
            #         token_type_ids=segment_ids,
            #         labels=y
            #     )[1] # [1] returns logits
            # else:
            outputs = model(x)

            loss_main = loss_computer.loss(outputs, y, g, is_training)

            if is_training:
                # if args.model == 'bert':
                #     loss_main.backward()
                #     torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                #     scheduler.step()
                #     optimizer.step()
                #     model.zero_grad()
                # else:
                optimizer.zero_grad()
                loss_main.backward()
                optimizer.step()

            if is_training and (batch_idx+1) % log_every==0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()

        if (not is_training) or loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            if is_training:
                loss_computer.reset_stats()


def train(model, criterion, dataset,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args, epoch_offset):
    model = model.cuda(args.gpu_num)

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(
        criterion,
        is_robust=args.robust,
        dataset=dataset['train_data'],
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight,vs_alpha=args.vs_alpha,loss_type=args.loss)

    # BERT uses its own scheduler and optimizer
    # if args.model == 'bert':
    #     no_decay = ['bias', 'LayerNorm.weight']
    #     optimizer_grouped_parameters = [
    #         {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)], 'weight_decay': args.weight_decay},
    #         {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    #         ]
    #     optimizer = AdamW(
    #         optimizer_grouped_parameters,
    #         lr=args.lr,
    #         eps=args.adam_epsilon)
    #     t_total = len(dataset['train_loader']) * args.n_epochs
    #     print(f'\nt_total is {t_total}\n')
    #     scheduler = WarmupLinearSchedule(
    #         optimizer,
    #         warmup_steps=args.warmup_steps,
    #         t_total=t_total)
    # else:

    optimizer = torch.optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=args.weight_decay)

    # optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=args.lr,
    #     # momentum=0.9,
    #     weight_decay=args.weight_decay) ##ここはSGDに変わる可能性あり
    
    
    if args.scheduler:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            'min',
            factor=0.1,
            patience=5,
            threshold=0.0001,
            min_lr=0,
            eps=1e-08)
    else:
        scheduler = None

    best_val_acc = 0
    for epoch in range(epoch_offset, epoch_offset+args.n_epochs):
        logger.write('\nEpoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')
        run_epoch(
            epoch, model, optimizer,
            dataset['train_loader'],
            train_loss_computer,
            logger, train_csv_logger, args,
            is_training=True,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler)

        logger.write(f'\nValidation:\n')
        val_loss_computer = LossComputer(
            criterion,
            is_robust=args.robust,
            dataset=dataset['val_data'],
            step_size=args.robust_step_size,
            alpha=args.alpha,loss_type=args.loss,vs_alpha=args.vs_alpha)
        run_epoch(
            epoch, model, optimizer,
            dataset['val_loader'],
            val_loss_computer,
            logger, val_csv_logger, args,
            is_training=False)

        # Test set; don't print to avoid peeking
        logger.write(f'\nTesting:\n')
        if dataset['test_data'] is not None:
            test_loss_computer = LossComputer(
                criterion,
                is_robust=args.robust,
                dataset=dataset['test_data'],
                step_size=args.robust_step_size,
                alpha=args.alpha,loss_type=args.loss,vs_alpha=args.vs_alpha)
            run_epoch(
                epoch, model, optimizer,
                dataset['test_loader'],
                test_loss_computer,
                logger, test_csv_logger, args,
                is_training=False)

        # Inspect learning rates
        if (epoch+1) % 1 == 0:
            for param_group in optimizer.param_groups:
                curr_lr = param_group['lr']
                logger.write('Current lr: %f\n' % curr_lr)

        if args.scheduler and args.model != 'bert':
            if args.robust:
                val_loss, _ = val_loss_computer.compute_robust_loss_greedy(val_loss_computer.avg_group_loss, val_loss_computer.avg_group_loss)
            else:
                val_loss = val_loss_computer.avg_actual_loss
            scheduler.step(val_loss) #scheduler step to update lr at the end of epoch

        if epoch % args.save_step == 0:
            torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

        if args.save_last:
            torch.save(model, os.path.join(args.log_dir, 'last_model.pth'))

        if args.save_best:
            if args.robust or args.reweight_groups:
                curr_val_acc = min(val_loss_computer.avg_group_acc)
            else:
                curr_val_acc = val_loss_computer.avg_acc
            logger.write(f'Current validation accuracy: {curr_val_acc}\n')
            if curr_val_acc > best_val_acc:
                best_val_acc = curr_val_acc
                torch.save(model, os.path.join(args.log_dir, 'best_model.pth'))
                logger.write(f'Best model saved at epoch {epoch}\n')

        if args.automatic_adjustment:
            gen_gap = val_loss_computer.avg_group_loss - train_loss_computer.exp_avg_loss
            adjustments = gen_gap * torch.sqrt(train_loss_computer.group_counts)
            train_loss_computer.adj = adjustments
            logger.write('Adjustments updated\n')
            for group_idx in range(train_loss_computer.n_groups):
                logger.write(
                    f'  {train_loss_computer.get_group_name(group_idx)}:\t'
                    f'adj = {train_loss_computer.adj[group_idx]:.3f}\n')
        logger.write('\n')

### For ours
def run_epoch_OP(epoch, model, per_cls_weight ,optimizer, loader, loss_computer, logger, csv_logger, args,
              is_training, show_progress=False, log_every=50, scheduler=None):
    if is_training:
        model.train()
        # criterion = nn.CrossEntropyLoss(weight=per_cls_weights).cuda(args.gpu_num)
    else:
        model.eval()

    if show_progress:
        prog_bar_loader = tqdm(loader)
    else:
        prog_bar_loader = loader

    pred_labels = []
    with torch.set_grad_enabled(is_training):
        for batch_idx, batch in enumerate(prog_bar_loader):
            batch = tuple(t.cuda(args.gpu_num) for t in batch)
            x = batch[0]
            y = batch[1]
            g = batch[2]
            outputs = model(x)
            pred = torch.max(outputs, 1)[1]
            pred_labels.extend(pred.cpu().numpy()) ## for ours

            loss_main = loss_computer.loss(outputs, y, g, is_training)

            if is_training:
                optimizer.zero_grad()
                loss_main.backward()
                optimizer.step()

            if is_training and (batch_idx+1) % log_every==0:
                csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
                csv_logger.flush()
                loss_computer.log_stats(logger, is_training)
                loss_computer.reset_stats()

        if (not is_training) or loss_computer.batch_count > 0:
            csv_logger.log(epoch, batch_idx, loss_computer.get_stats(model, args))
            csv_logger.flush()
            loss_computer.log_stats(logger, is_training)
            stats = loss_computer.get_stats(model=model, args=args)

            if is_training:
                loss_computer.reset_stats()
        # avg_acc_group = {key: value for key, value in stats.items() if key.startswith('avg_acc_group')}


    return stats,model,pred_labels

def train_OP_init(model, criterion, dataset,per_cls_weight,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args, epoch_offset):
    model = model.cuda(args.gpu_num)

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(
        criterion,
        is_robust=args.robust,
        dataset=dataset['train_data'],
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight,vs_alpha=args.vs_alpha,loss_type=args.loss)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) ##ここはSGDに変わる可能性あり
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)
    
    scheduler = None

    for epoch in range(1):
        # logger.write('\Epoch [%d]:\n' % epoch)
        # logger.write(f'Training:\n')
        stats,weak_model,_ = run_epoch_OP(
            epoch, model, per_cls_weight, optimizer,
            dataset['train_loader'],
            train_loss_computer,
            logger, train_csv_logger, args,
            is_training=True,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler)
    print("=== OP initialize Finished ===")
        
    return True


def train_OP(model, criterion, dataset,per_cls_weight,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args, epoch_offset):
    model = model.cuda(args.gpu_num)

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    train_loss_computer = LossComputer(
        criterion,
        is_robust=args.robust,
        dataset=dataset['train_data'],
        alpha=args.alpha,
        gamma=args.gamma,
        adj=adjustments,
        step_size=args.robust_step_size,
        normalize_loss=args.use_normalized_loss,
        btl=args.btl,
        min_var_weight=args.minimum_variational_weight,vs_alpha=args.vs_alpha,loss_type=args.loss)

    # optimizer = torch.optim.SGD(
    #     filter(lambda p: p.requires_grad, model.parameters()),
    #     lr=args.lr,
    #     momentum=0.9,
    #     weight_decay=args.weight_decay)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) ##ここはSGDに変わる可能性あり
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)
    
    scheduler = None

    # best_val_acc = 0

    for epoch in range(epoch_offset, epoch_offset+args.max_epoch):
        logger.write('\Epoch [%d]:\n' % epoch)
        logger.write(f'Training:\n')
        stats,weak_model,_ = run_epoch_OP(
            epoch, model, per_cls_weight, optimizer,
            dataset['train_loader'],
            train_loss_computer,
            logger, train_csv_logger, args,
            is_training=True,
            show_progress=args.show_progress,
            log_every=args.log_every,
            scheduler=scheduler)
        
        ## check_weak learnabiity
        avg_acc_group = {key: value for key, value in stats.items() if key.startswith('avg_acc_group')}

        feed = Feedbuck.Binary_ACC
        
        weight_l = per_cls_weight.tolist()
        rt = [0]* dataset['train_data'].n_groups
        for idx in range(dataset['train_data'].n_groups):
            rt[idx] = feed(list(avg_acc_group.values())[idx],args.theta)

        ft = 0
        for idx in range( dataset['train_data'].n_groups):
            ft +=  weight_l[idx]*rt[idx]

        if ft >=  (1/2) + (args.gamma - args.epsilon):
            # f.append(np.dot(weight_l,rt))
            print(f"Satisfied with W.L Definition : {epoch}")
            return list(avg_acc_group.values()), weak_model,rt

        elif epoch == (args.max_epoch-1):
            print("Couldn't Satisfied with W.L Definition")
            sys.exit()
        ##

    # logger.write(f'\nValidation:\n')

    # val_loss_computer = LossComputer(
    #     criterion,
    #     is_robust=args.robust,
    #     dataset=dataset['val_data'],
    #     step_size=args.robust_step_size,
    #     alpha=args.alpha,loss_type=args.loss,vs_alpha=args.vs_alpha)
    
    # run_epoch_OP(
    #     epoch, model,optimizer,
    #     dataset['val_loader'],
    #     val_loss_computer,
    #     logger, val_csv_logger, args,
    #     is_training=False)

    # Test set; don't print to avoid peeking
    # logger.write(f'\nTesting:\n')
    # if dataset['test_data'] is not None:
    #     test_loss_computer = LossComputer(
    #         criterion,
    #         is_robust=args.robust,
    #         dataset=dataset['test_data'],
    #         step_size=args.robust_step_size,
    #         alpha=args.alpha,loss_type=args.loss,vs_alpha=args.vs_alpha)
    #     run_epoch_OP(
    #         epoch, model,optimizer,
    #         dataset['test_loader'],
    #         test_loss_computer,
    #         logger, test_csv_logger, args,
    #         is_training=False)

    # Inspect learning rates
    if (epoch+1) % 1 == 0:
        for param_group in optimizer.param_groups:
            curr_lr = param_group['lr']
            logger.write('Current lr: %f\n' % curr_lr)

    # if epoch % args.save_step == 0:
    #     torch.save(model, os.path.join(args.log_dir, '%d_model.pth' % epoch))

    # if args.save_last:
    #     torch.save(model, os.path.join(args.log_dir, 'last_model.pth'))
    logger.write('\n')

    return list(avg_acc_group.values()), weak_model,rt

def val_OP(model, criterion, dataset,per_cls_weight,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args, epoch_offset):
    model = model.cuda(args.gpu_num)

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) ##ここはSGDに変わる可能性あり
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)
    
    scheduler = None

    # best_val_acc = 0

    # for epoch in range(epoch_offset, epoch_offset+args.max_epoch):
    for epoch in range(1): ##最後の一回だけテストする
        logger.write(f'\nValidation:\n')

        val_loss_computer = LossComputer(
            criterion,
            is_robust=args.robust,
            dataset=dataset['val_data'],
            step_size=args.robust_step_size,
            alpha=args.alpha,loss_type=args.loss,vs_alpha=args.vs_alpha)
        
        stats,_,pred_labels = run_epoch_OP(
            epoch, model,per_cls_weight,optimizer,
            dataset['val_loader'],
            val_loss_computer,
            logger, val_csv_logger, args,
            is_training=False)

        # # Test set; don't print to avoid peeking
        # logger.write(f'\nTesting:\n')
        # if dataset['test_data'] is not None:
        #     test_loss_computer = LossComputer(
        #         criterion,
        #         is_robust=args.robust,
        #         dataset=dataset['test_data'],
        #         step_size=args.robust_step_size,
        #         alpha=args.alpha,loss_type=args.loss,vs_alpha=args.vs_alpha)
            
        #     stats,_,pred_labels = run_epoch_OP(
        #         epoch, model,per_cls_weight,optimizer,
        #         dataset['test_loader'],
        #         test_loss_computer,
        #         logger, test_csv_logger, args,
        #         is_training=False)
            
            
    avg_acc_group = {key: value for key, value in stats.items() if key.startswith('avg_acc_group')}
    logger.write('\n')

    return list(avg_acc_group.values()), pred_labels

def test_OP(model, criterion, dataset,per_cls_weight,
          logger, train_csv_logger, val_csv_logger, test_csv_logger,
          args, epoch_offset):
    model = model.cuda(args.gpu_num)

    # process generalization adjustment stuff
    adjustments = [float(c) for c in args.generalization_adjustment.split(',')]
    assert len(adjustments) in (1, dataset['train_data'].n_groups)
    if len(adjustments)==1:
        adjustments = np.array(adjustments* dataset['train_data'].n_groups)
    else:
        adjustments = np.array(adjustments)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001) ##ここはSGDに変わる可能性あり
    # optimizer = torch.optim.SGD(model.parameters(), lr=args.lr,momentum=0.9)
    
    scheduler = None

    # best_val_acc = 0

    # for epoch in range(epoch_offset, epoch_offset+args.max_epoch):
    for epoch in range(1): ##最後の一回だけテストする
    # logger.write(f'\nValidation:\n')

    # val_loss_computer = LossComputer(
    #     criterion,
    #     is_robust=args.robust,
    #     dataset=dataset['val_data'],
    #     step_size=args.robust_step_size,
    #     alpha=args.alpha,loss_type=args.loss,vs_alpha=args.vs_alpha)
    
    # run_epoch_OP(
    #     epoch, model,optimizer,
    #     dataset['val_loader'],
    #     val_loss_computer,
    #     logger, val_csv_logger, args,
    #     is_training=False)

        # Test set; don't print to avoid peeking
        logger.write(f'\nTesting:\n')
        if dataset['test_data'] is not None:
            test_loss_computer = LossComputer(
                criterion,
                is_robust=args.robust,
                dataset=dataset['test_data'],
                step_size=args.robust_step_size,
                alpha=args.alpha,loss_type=args.loss,vs_alpha=args.vs_alpha)
            
            stats,_,pred_labels = run_epoch_OP(
                epoch, model,per_cls_weight,optimizer,
                dataset['test_loader'],
                test_loss_computer,
                logger, test_csv_logger, args,
                is_training=False)
            
            
    avg_acc_group = {key: value for key, value in stats.items() if key.startswith('avg_acc_group')}
    logger.write('\n')

    return list(avg_acc_group.values()), pred_labels