import torch

class ActList():
    '''
    Doubles as gradient recording when used for backward hooks
    Args:
      forward: Bool, whether to forward or backward hook
    Returns:
    '''
    def __init__(self, forward=True):
        self.acts = []
        self.forward = forward
        self.dimensions = None

    def record_output(self, module, input, output):
        '''
        Doubles as gradient recording when used for backward hook. Appends activations into self.acts.
        '''
        if self.forward and self.dimensions is None: self.dimensions = output.data.cpu().numpy().shape
        if self.forward: self.acts.append(output.data.cpu().reshape(-1).numpy())
        else: self.acts.append(output[0].cpu().reshape(-1).numpy())


def get_acts(model, layer_name, dataset, concept_labels,):
    '''
    returns list of [act,concept_name] pairs
    '''
    act_log = ActList()
    hook = None
    for name, mod in model.named_modules():
        if name==layer_name: hook = mod.register_forward_hook(act_log.record_output)
    if hook is None:
        raise NameError(layer_name+'Not found')
    model.eval()
    
    sample = dataset[0]
    if type(sample)==dict:
        batch_shape = [1]+list(dataset[0]['image'].shape)
    else:
        batch_shape = [1]+list(dataset[0][0].shape)
    for ind in range(len(dataset)):
        with torch.no_grad():
            im = dataset.__getitem__(ind)
            if type(im)==dict:
                im = im['image'].view(*batch_shape).cuda()
            else:
                im = im[0].view(*batch_shape).cuda()
            model(im)
            
    acts = zip(act_log.acts, concept_labels)
    acts = list(map(list,acts))

    hook.remove()
    return acts, act_log.dimensions


def get_grads(model, layer_name, dataset, true_class_nums, grad_class_num, num_classes):
    '''
    Returns list of gradients for all samples in dataset of class class_num ordered by index
    '''
    grad_log = ActList(forward=False)
    for name, mod in model.named_modules():
        if name==layer_name: hook = mod.register_backward_hook(grad_log.record_output)
    model.eval()

    target = torch.zeros([num_classes])
    target[grad_class_num]=1
    target = target.cuda()
    sample = dataset[0]
    if type(sample)==dict:
        batch_shape = [1]+list(dataset[0]['image'].shape)
    else:
        batch_shape = [1]+list(dataset[0][0].shape)
    
    for ind in range(len(dataset)):
        model.zero_grad()
        item = dataset.__getitem__(ind)
        if type(item)==dict:
            if not item['labels'] in true_class_nums: continue
            im = item['image'].view(*batch_shape).cuda()
        else:
            if not item[1] in true_class_nums: continue
            im = item[0].view(*batch_shape).cuda()
        out = model(im)
        masked_out = torch.sum(out*target)
        masked_out.backward()

    grads = grad_log.acts
    hook.remove()
    return grads