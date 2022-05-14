import os
import ssl
import pickle
import smtplib
import argparse
import traceback
import functools
import numpy as np
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
from multiprocessing import Process
from pandarallel import pandarallel

pandarallel.initialize(progress_bar=True, verbose=0)
tqdm.pandas()


def compute_AUC(x, y):
    return 0.5*np.inner(y[1:]+y[:-1], x[1:]-x[:-1])


def fpr_tpr_precision_recall_pr_baseline(D):

    ths   = sorted([key for key in D.keys() if key!='label'], reverse=True)
    label = D['label']

    tp  = np.array([np.inner(  D[th],   label) for th in ths], dtype=np.float32)
    fp  = np.array([np.inner(  D[th], 1-label) for th in ths], dtype=np.float32)
    fn  = np.array([np.inner(1-D[th],   label) for th in ths], dtype=np.float32)
    p   =    label.sum()
    n   = (1-label).sum()

    fpr = fp/n
    tpr = tp/p

    precision = np.divide(tp, tp+fp, out=np.zeros_like(tp), where=tp+fp>0)
    recall    = np.divide(tp, tp+fn, out=np.zeros_like(tp), where=tp+fn>0)


    return fpr, tpr, precision, recall, sum(label)/len(label)


def save_ROC_plot(fpr_tpr_label, figname, fontsize=25, figsize=(10, 10), dpi=400):
    plt.figure(figsize=figsize, dpi=dpi)
    #plt.rcParams["figure.figsize"] = figsize
    plt.xlabel("False Positive Rate", fontsize = fontsize)
    plt.ylabel("True Positive Rate", fontsize = fontsize)
    plt.title("Receiver Operating Characteristic", fontsize = fontsize)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    for fpr_tpr_label_ in fpr_tpr_label:
        plt.plot(fpr_tpr_label_['fpr'], fpr_tpr_label_['tpr'], label = fpr_tpr_label_['label'], linestyle = '--', linewidth=5.0, marker=fpr_tpr_label_['marker']) 
    plt.plot([1,0], [1,0], linestyle = 'dotted', color = 'black', label = 'Baseline')
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.legend(fontsize = fontsize)
    plt.grid(linestyle = '--',linewidth=2)
    plt.savefig(figname, format=figname.split('.')[-1])


def save_PR_plot(r_p_label, figname, fontsize=25, figsize=(10, 10), dpi=400):
    plt.figure(figsize=figsize, dpi=dpi)
    #plt.rcParams["figure.figsize"] = figsize
    plt.xlabel("Recall", fontsize = fontsize)
    plt.ylabel("Precision", fontsize = fontsize)
    plt.title("Precision Recall", fontsize = fontsize)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    pr_baseline = r_p_label[0]['baseline']
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)
    plt.grid(linestyle = '--',linewidth=2)
    #plt.plot([0,1], [0.06,0.06], color = 'black', label = 'Baseline', linestyle = 'dotted')
    for r_p_label_ in r_p_label:
        #for pr, re in zip(r_p_label_['precision'], r_p_label_['recall']):
        #    print (pr, re)
        #print ("-")
        plt.plot(r_p_label_['recall'], r_p_label_['precision'], label = r_p_label_['label'], linestyle = '--', linewidth=5.0, marker=r_p_label_['marker'])     
    plt.axhline(y=pr_baseline, linestyle = 'dotted', color = 'crimson', label = f'Baseline ({pr_baseline:.2f})')
    plt.legend(fontsize = fontsize)
    plt.savefig(figname, format=figname.split('.')[-1])   


def ROC_PR(comm_dict, models):
    roc, pr = [], []
    marker={
        "Cox":    "*",
        "BoXHED": "s",
        'XGB:Cox': ".",
        "DDH": "d",
    }    

    print(models)
    for model in models:
        print(model)
        fpr, tpr, precision, recall, pr_baseline = fpr_tpr_precision_recall_pr_baseline(comm_dict[model])
            
        roc.append({
            'fpr': fpr,
            'tpr': tpr,
            'label': f"{model} (AUROC={compute_AUC(fpr, tpr):.2f})",
            'marker': marker[model]
        })

        pr.append({
            'precision': precision,
            'recall':    recall,
            'baseline':  pr_baseline,
            'label': f"{model} (AUC-PR={compute_AUC(recall, precision):.2f})",
            'marker': marker[model]
        })

    return roc, pr


def run_as_Ps(f_list, args_dict_list):
    
    P = []
    for f in f_list:
        for args_dict in args_dict_list:
            p=Process(target = f, kwargs = args_dict)
            p.start()
            P.append(p)

    for p in P:
        p.join()


def send_email_notif(success=True, msg=""):
    port = 465  # For SSL
    smtp_server = "smtp.gmail.com"
    sender_email = "ari.paki.1994@gmail.com"  # Enter your address
    receiver_email = "a.pakbin@tamu.edu"  # Enter receiver address
    password = "Pa123456!!"
    text     = msg
    subject = f"Run{'' if success else ' NOT'} successful!"

    message = 'Subject: {}\n\n{}'.format(subject, text)

    context = ssl.create_default_context()
    with smtplib.SMTP_SSL(smtp_server, port, context=context) as server:
        server.login(sender_email, password)
        server.sendmail(sender_email, receiver_email, message)


def email_when_done(func):
    @functools.wraps(func)
    def email_when_done(*args, **kwargs):

        try:
            output = func(*args, **kwargs)
        except KeyboardInterrupt:
            raise
        except:
            send_email_notif(success=False, msg=traceback.format_exc())
            raise

        send_email_notif(success=True, msg=f"{func.__name__}() completed successfully!")
        return output
    return email_when_done


def exec_if_not_cached(func):
    @functools.wraps(func)

    def _exec_if_not_cached(file_name, func, *args, **kwargs):
        file_path = os.path.join('./tmp/', file_name+'.pkl')
        if Path(file_path).is_file():
            return load_pickle(file_path)
        else:
            create_dir_if_not_exist(os.path.dirname(file_path))
            output = func(*args, **kwargs)
            dump_pickle(output, file_path)
            return output

    def _func_args_to_str(func, *args, **kwargs):
        return func.__name__ + ''.join([f"__{str(arg)}" for arg in args]) + ''.join([f"__{str(kw)}_{str(arg)}" for kw, arg in kwargs.items()])

    def exec_if_not_cached(*args, **kwargs):
        return _exec_if_not_cached(_func_args_to_str(func, *args, **kwargs), func, *args, **kwargs)

    return exec_if_not_cached


def dump_pickle(obj, addr):
    with open(addr, 'wb') as handle:
        pickle.dump(obj, handle) 


def load_pickle(addr):
    with open(addr, 'rb') as handle:
        obj = pickle.load(handle)
    return obj 


def plot_save_var_imps(vars, imps, fig_addr, show_larger_than=0.1):
    def plot_var_imps(vars, imps):
        font_size = 40
        fig, axis = plt.subplots(figsize=(20,12), dpi=100)
        plt.xticks(fontsize= font_size)
        plt.yticks(fontsize= font_size)
        plt.title(f"{model} Normalized VarImp", fontsize=font_size)
        plt.bar(vars, imps, color='yellow')
        plt.xticks(rotation = -90)
        labels = axis.set_xticklabels(vars)
        for label in labels:
            label.set_y(label.get_position()[1]+0.9)
        plt.savefig(fig_addr)

    srtd_imp_idxs = imps.argsort()
    vars          = vars[srtd_imp_idxs[::-1]]
    imps          = imps[srtd_imp_idxs[::-1]]

    model         = fig_addr.split('/')[-1].split('_')[0]
    dump_pickle([vars, imps], f"./tmp/{model}_var_imps.pkl")

    norm_imps     = imps/imps.sum()
    vars          = vars      [norm_imps>show_larger_than]
    norm_imps     = norm_imps [norm_imps>show_larger_than]

    plot_var_imps(vars, norm_imps)


def cli_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--monitor-time', default = 8,          required=False, type   = int                           )
    parser.add_argument('--recurr',       default = True,       required=False, action = argparse.BooleanOptionalAction)
    parser.add_argument('--cv',           default = False,      required=False, action = argparse.BooleanOptionalAction)
    parser.add_argument('--njobs',        default = 1,          required=False, type   = int                           )
    parser.add_argument('--rslt-dir',     default = './rslts/', required=False, type   = str                           )
    parser.add_argument('--tmp-dir',      default = './tmp/',   required=False, type   = str                           )
    return parser.parse_args()


def create_dir_if_not_exist(path):
    if not os.path.exists(path):
        os.makedirs(path)