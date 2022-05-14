import numpy as np
from tqdm import tqdm
from multiprocessing import Manager
from _utils import run_as_Ps, ROC_PR, dump_pickle, load_pickle, save_PR_plot, save_ROC_plot, email_when_done

tqdm.pandas()

#https://stackoverflow.com/questions/26515595/how-does-one-ignore-unexpected-keyword-arguments-passed-to-a-function
def static_flag(hzrds, deltas, **kwargs):
    ths = np.concatenate(([-np.inf], np.sort(np.unique(np.around(hzrds, decimals=3)), kind='mergesort'), [np.inf]))
    out = {th:(hzrds>th).astype('int') for th in tqdm(ths)}

    out['label'] = deltas
    return out


def single_ep_flag(hzrds, deltas, Y, ths, idxs):
    delta   = deltas[idxs[-1]]
    idxs    = idxs[Y[idxs]==1]
    assert len(idxs)>0
    return [(hzrds[idxs] > th).sum() > 0 for th in ths] + [bool(delta)]


def batch_ep_flag(comm_dict, batch_idx):
    hzrds, deltas, Y, ths, ids_bndry, id_batches = \
        [comm_dict[key] for key in ['hzrds', 'deltas', 'Y', 'ths', 'ids_bndry', 'id_batches']]

    comm_dict[f'ep_batch_{batch_idx}'] = \
        np.asarray([single_ep_flag(hzrds, deltas, Y, ths, np.arange(ids_bndry[idx], ids_bndry[idx+1])) 
            for idx in tqdm(id_batches[batch_idx])]).transpose()


def flag(ep_ids, deltas, Y, hzrds, njobs, **kwargs):
    ids_bndry  = np.concatenate(([0],1+np.where(ep_ids[:-1] != ep_ids[1:])[0], [len(ep_ids)]))
    id_batches = np.array_split(range(len(ids_bndry)-1), njobs)
    ths        = np.concatenate(([hzrds.min()-1], 
                 np.sort(np.unique(np.around(hzrds, decimals=3)), kind='mergesort'), 
                 [hzrds.max()+1]))

    manager    = Manager()
    comm_dict  = manager.dict()
    
    comm_dict.update({
        'deltas':     deltas,
        'Y':          Y,
        'hzrds':      hzrds,
        'ths':        ths,
        'ids_bndry':  ids_bndry,
        'id_batches': id_batches
    })

    run_as_Ps([batch_ep_flag], [{"comm_dict": comm_dict, "batch_idx":batch_idx}
        for batch_idx in range(len(id_batches))])
    
    out = np.hstack([comm_dict[f'ep_batch_{batch_idx}'] for batch_idx in range(len(id_batches))]).astype('int')
    return {key:out[key_idx,:] for key_idx, key in enumerate(ths)} | {'label': out[-1,:]}
        


model_flaggings = [
    {"model": "BoXHED",  "flagging": flag},
    {"model": "XGB:Cox", "flagging": static_flag},
    {"model": "Cox",     "flagging": flag},
]

# @email_when_done
def flagging_main():
    rslts = {}
    
    for model_flagging in model_flaggings:
        rslts[model_flagging['model']] = \
            model_flagging["flagging"](**(load_pickle(f"./rslts/{model_flagging['model']}_rslts.pkl") | {"njobs":60}))

    
    models = [model_flagging['model'] for model_flagging in model_flaggings]
    roc, pr = ROC_PR(rslts, models)

    save_ROC_plot(roc, f"./rslts/{'_'.join(models)}_ROC.png", fontsize=25, figsize=(15, 15), dpi=100)
    save_PR_plot (pr,  f"./rslts/{'_'.join(models)}_PR.png" , fontsize=25, figsize=(15, 15), dpi=100)

if __name__=="__main__":
    flagging_main()
