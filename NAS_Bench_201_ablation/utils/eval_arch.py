import os
import tqdm
import torch
from meta_acc_predictor.get_datasets import get_datasets
from network import get_cell_based_tiny_net
from utils.nb201_fitness import get_nb201_arch_str
from utils.flop_benchmark import get_model_infos
from utils.optimizers import get_optim_scheduler
from concurrent.futures import ThreadPoolExecutor, as_completed


def get_a_network(operation_matrix, num_classes: int):
    """Get the network configuration from the NAS_Bench_201 API.

    Args:
    - api: NASBench201API object
    - operation_matrix: operation_matrix of a network, its shape should be (8, 7) with float values
    """
    arch_str = get_nb201_arch_str(operation_matrix=operation_matrix)
    network_config = {
        'name': 'infer.tiny',
        'C': 16,
        'N': 5,
        'arch_str': arch_str,
        'num_classes': num_classes,
    }
    net = get_cell_based_tiny_net(config=network_config)
    return net, arch_str


def obtain_accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        # correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(index, net, train_dataloader, lr, momentum, decay, nesterov, epochs, warmup_epoch, eta_min, device,
          early_stop):
    # Training
    net.train()
    optimizer, scheduler, criterion = get_optim_scheduler(
        parameters=net.parameters(),
        lr=lr,
        momentum=momentum,
        decay=decay,
        nesterov=nesterov,
        epochs=epochs,
        warmup_epoch=warmup_epoch,
        eta_min=eta_min
    )
    net, criterion = net.to(device), criterion.to(device)
    bar = tqdm.tqdm(range(epochs + warmup_epoch), ncols=120)
    bar.set_description(f'Net: {index}')
    for epoch in bar:
        scheduler.update(epoch, 0.0)
        avg_loss = 0.0
        avg_prec_top1 = 0.0
        avg_prec_top5 = 0.0
        for i, (inputs, targets) in enumerate(train_dataloader):
            scheduler.update(None, 1.0 * i / len(train_dataloader))
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            features, logits = net(inputs)
            loss = criterion(logits, targets)
            loss.backward()
            optimizer.step()
            prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
            avg_loss += loss.item()
            avg_prec_top1 += prec1.item()
            avg_prec_top5 += prec5.item()
        avg_loss /= len(train_dataloader)
        avg_prec_top1 /= len(train_dataloader)
        avg_prec_top5 /= len(train_dataloader)
        bar.set_postfix_str(f'Loss={avg_loss:.4f}, Train Prec@1={avg_prec_top1:.2f}, Train Prec@5={avg_prec_top5:.2f}')
        if early_stop:
            if avg_loss < 0.001 and avg_prec_top1 > 99.9:
                break  # early stop
    return net


def test(net, test_dataloader, device):
    # Testing
    print('>>> Test network...')
    net.eval()
    net = net.to(device)
    avg_prec_top1 = 0.0
    avg_prec_top5 = 0.0
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_dataloader):
            inputs, targets = inputs.to(device), targets.to(device)
            features, logits = net(inputs)
            prec1, prec5 = obtain_accuracy(logits.data, targets.data, topk=(1, 5))
            avg_prec_top1 += prec1.item()
            avg_prec_top5 += prec5.item()
    avg_prec_top1 /= len(test_dataloader)
    avg_prec_top5 /= len(test_dataloader)
    return avg_prec_top1, avg_prec_top5


def eval_an_arch(
        index: int,
        api,
        operation_matrix: torch.Tensor,
        dataset_name: str,
        batch_size: int,
        device: str,
        lr: float,
        momentum: float,
        decay: float,
        nesterov: bool,
        epochs: int,
        warmup_epoch: int,
        eta_min: float,
        train_data,
        test_data,
        xshape,
        class_num: int,
        early_stop: bool
):
    try:
        _ = api.query_index_by_arch(get_nb201_arch_str(operation_matrix=operation_matrix))
    except:
        print(f'Architecture {operation_matrix} is not in the NAS_Bench_201 dataset')
        return 0.0

    train_dataloader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    test_dataloader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    net, arch_str = get_a_network(operation_matrix=operation_matrix, num_classes=class_num)
    print(f'>>> Evaluate architecture {index}: {arch_str} with dataset {dataset_name}...')

    flop, param = get_model_infos(model=net, shape=xshape)
    print(f'For arch {index}, FLOPs = {flop} MB, Param = {param} MB')

    # 只保留原逻辑：FLOPs 过滤
    # if flop < 100:
    #     return 0.0

    net = train(
        index=index,
        net=net,
        train_dataloader=train_dataloader,
        lr=lr,
        momentum=momentum,
        decay=decay,
        nesterov=nesterov,
        epochs=epochs,
        warmup_epoch=warmup_epoch,
        eta_min=eta_min,
        device=device,
        early_stop=early_stop
    )

    prec1, prec5 = test(net=net, test_dataloader=test_dataloader, device=device)
    print(f'For arch {index}, Test Prec@1 = {prec1:.2f}, Test Prec@5 = {prec5:.2f}')

    return prec1

from concurrent.futures import ThreadPoolExecutor, as_completed
import torch
import os

def eval_architectures(
        x: torch.Tensor,
        api,
        dataset_name: str,
        image_cutout: int,
        batch_size: int,
        device: str,
        lr: float,
        momentum: float,
        decay: float,
        nesterov: bool,
        train_epochs: int,
        warmup_epoch: int,
        eta_min: float,
        multi_thread: bool = False,
        early_stop: bool = False,
        repeat_times: int = 3,
):
    arch_num = x.shape[0]
    arch_matrices = x.view(arch_num, 6)
    acc_list = [None] * arch_num

    train_data, test_data, xshape, class_num = get_datasets(name=dataset_name, cutout=image_cutout)
    torch.cuda.empty_cache()

    cache_dir = "results/eval"
    cache_path = f"{cache_dir}/eval_{dataset_name}_avg.pth"

    # 读取缓存（如果存在）
    saved_dict = {}
    if os.path.exists(cache_path):
        saved_dict = torch.load(cache_path)

    def eval_one_arch(i: int):
        arch_matrix = arch_matrices[i, :]

        # 仅为了拿到 arch_str 用来查缓存（不训练）
        _, arch_str = get_a_network(operation_matrix=arch_matrix, num_classes=class_num)

        # 1) 有缓存：一次都不训练
        if arch_str in saved_dict:
            print(f'For arch {i}, Test Prec@1 = {saved_dict[arch_str]:.2f} (cached)')
            return saved_dict[arch_str]

        # 2) 无缓存：训练 repeat_times 次取平均
        prec1_list = []
        for _ in range(repeat_times):
            prec1 = eval_an_arch(
                index=i,
                api=api,
                operation_matrix=arch_matrix,
                dataset_name=dataset_name,
                batch_size=batch_size,
                device=device,
                lr=lr,
                momentum=momentum,
                decay=decay,
                nesterov=nesterov,
                epochs=train_epochs,
                warmup_epoch=warmup_epoch,
                eta_min=eta_min,
                train_data=train_data,
                test_data=test_data,
                xshape=xshape,
                class_num=class_num,
                early_stop=early_stop
            )
            prec1_list.append(prec1)

        avg_prec1 = sum(prec1_list) / len(prec1_list)

        # 3) 写缓存：只写一次（写 avg）
        # if not os.path.exists(cache_dir):
        #     os.makedirs(cache_dir)
        #
        # saved_dict[arch_str] = avg_prec1
        # torch.save(saved_dict, cache_path)

        return avg_prec1

    if multi_thread:
        def eval_task(i):
            return i, eval_one_arch(i)

        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(eval_task, i) for i in range(arch_num)]
            for future in as_completed(futures):
                i, avg_prec1 = future.result()
                acc_list[i] = avg_prec1
    else:
        for i in range(arch_num):
            acc_list[i] = eval_one_arch(i)

    max_acc = max(acc_list)
    print(f'Max accuracy among {arch_num} searched architectures: {max_acc}')
    return max_acc, acc_list
