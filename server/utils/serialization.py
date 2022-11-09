# Save and Load Functions
import torch
import logging

logger = logging.getLogger('global')


def save_checkpoint(save_path, model):
    if save_path == None:
        return
    state_dic = {'model': model.state_dict()}
    torch.save(state_dic, save_path)
    logger.info('Model saved to ==> {}'.format(save_path))


def load_checkpoint(load_path, model):
    if load_path == None:
        return
    state_dict = torch.load(load_path)
    logger.info('Model loaded from <== {}'.format(load_path))

    model.load_state_dict(state_dict['model'])

    return model


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    if save_path == None:
        return

    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    logger.info('Model saved to ==> {}'.format(save_path))


def load_metrics(load_path):
    if load_path == None:
        return

    state_dict = torch.load(load_path, map_location=cfg.DEVICES)
    logger.info(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']
