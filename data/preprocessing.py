"""Preprocessing: DB query, resampling, masking, tensor conversion."""
import sqlite3
import numpy as np
import torch
from tqdm import tqdm


def call_EQ_motion(DBfilename, callEQ):
    EQtslist, EQdtlist, EQnPtslist = [], [], []
    conn = sqlite3.connect(DBfilename, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    for EQid in tqdm(callEQ, desc='[EQ Motion]', unit='eq'):
        cursor.execute('SELECT "acc_ts [m/s^2]", dt, nPts FROM eq_dt WHERE id = ?', (int(EQid),))
        ts, dt, nPts = cursor.fetchall()[0]
        EQtslist.append(ts)
        EQdtlist.append(dt)
        EQnPtslist.append(nPts)
    conn.commit()
    conn.close()
    return EQtslist, EQdtlist, EQnPtslist


def call_EQ_response(DBfilename, callEQ, callnode, rdof, acc_dsp_rct):
    EQresptslist, EQrespdtlist, EQrespnPtslist, EQGMfactlist = [], [], [], []
    conn = sqlite3.connect(DBfilename, detect_types=sqlite3.PARSE_DECLTYPES)
    cursor = conn.cursor()
    for EQid in tqdm(callEQ, desc=f'[Response {acc_dsp_rct}]', unit='eq'):
        for nodes in callnode:
            cursor.execute(
                'SELECT response_value, analysisdt, nPts, GMfact FROM node_resp '
                'WHERE (eq_id, node_id, response_dof, response_type) = (?,?,?,?)',
                (int(EQid), nodes, rdof, acc_dsp_rct))
            ts, dt, nPts, GMFact = cursor.fetchall()[0]
            EQresptslist.append(ts)
            EQrespdtlist.append(dt)
            EQrespnPtslist.append(nPts)
            EQGMfactlist.append(GMFact)
    conn.commit()
    conn.close()
    return EQresptslist, EQrespdtlist, EQrespnPtslist, EQGMfactlist


def resample_TS(orgtslist, orgdtlist, orgnPtslist, resampledt):
    tslist_res = []
    for i, original_dt in enumerate(tqdm(orgdtlist, desc='[Resample]', unit='ts')):
        n_samples = orgnPtslist[i]
        ts_org = orgtslist[i]
        tt_org = np.arange(0, n_samples * original_dt, original_dt)
        ts_org = ts_org[:len(tt_org)]
        tt_org = tt_org[:len(ts_org)]
        tt_new = np.arange(0, tt_org[-1] + 0.5 * resampledt, resampledt)
        ts_res = np.interp(tt_new, tt_org, ts_org)
        tslist_res.append(ts_res)
    maxlen = max(len(x) for x in tslist_res)
    REStsDATA = np.zeros((len(orgdtlist), maxlen))
    REStsMASK = np.zeros((len(orgdtlist), maxlen))
    for i in range(len(tslist_res)):
        REStsDATA[i, :len(tslist_res[i])] = tslist_res[i]
        REStsMASK[i, :len(tslist_res[i])] = 1.0
    REStsDATA = np.expand_dims(REStsDATA, 1)
    REStsMASK = np.expand_dims(REStsMASK, 1)
    REStsDATA = torch.from_numpy(REStsDATA).float()
    REStsMASK = torch.from_numpy(REStsMASK).float()
    return REStsDATA, REStsMASK


def _wrong_response_shape(EQtsMASK, EQresptsMASK):
    somethingwrongID = []
    need2adjustID = []
    nn = EQresptsMASK.shape[1] - 1
    for EQi in range(EQtsMASK.shape[0]):
        if np.sum(EQtsMASK[EQi, 0, :]) != np.sum(EQresptsMASK[EQi, nn, :]):
            delta = np.sum(EQtsMASK[EQi, 0, :]) - np.sum(EQresptsMASK[EQi, nn, :])
            need2adjustID.append(EQi)
            if abs(delta) > 2:
                somethingwrongID.append(EQi)
    return somethingwrongID, need2adjustID


def modify_EQ_response(DBfilename, callEQ, callnode, rdof, acc_dsp_rct, EQtsDATA, EQtsMASK):
    EQresptslist, _, _, GMfactlist = call_EQ_response(DBfilename, callEQ, callnode, rdof, acc_dsp_rct)
    GM_factors = np.array(GMfactlist[0::len(callnode)])
    maxlen = max(len(x) for x in EQresptslist)
    nEQ = len(EQresptslist) // len(callnode)
    nNode = len(callnode)
    EQresptsDATA = np.zeros((nEQ, nNode, maxlen))
    EQresptsMASK = np.zeros((nEQ, nNode, maxlen))

    for i in range(len(EQresptslist)):
        eq_idx = i // nNode
        nd_idx = i % nNode
        expected_len = int(EQtsMASK[eq_idx, 0, :].sum())
        actual_len = len(EQresptslist[i])
        delayD = expected_len - actual_len
        if delayD != 0 and abs(delayD) < 3:
            if delayD > 0:
                EQresptslist[i] = np.concatenate((np.full(int(delayD), 1e-6), EQresptslist[i]))
            elif delayD < 0:
                EQresptslist[i] = EQresptslist[i][:int(delayD)]
        L = min(len(EQresptslist[i]), maxlen)
        EQresptsDATA[eq_idx, nd_idx, :L] = EQresptslist[i][:L]
        EQresptsMASK[eq_idx, nd_idx, :L] = 1.0

    wrongEQid, _ = _wrong_response_shape(EQtsMASK, EQresptsMASK)
    goodDATAidx = sorted(set(range(EQtsMASK.shape[0])) - set(wrongEQid))
    GM_Factors = GM_factors[goodDATAidx]
    EQtsDATA_ = EQtsDATA[goodDATAidx]
    EQtsDATA_factored = EQtsDATA_ * GM_Factors[:, None, None]
    EQtsMASK = EQtsMASK[goodDATAidx]
    EQresptsDATA = EQresptsDATA[goodDATAidx]
    EQresptsMASK = EQresptsMASK[goodDATAidx]
    return EQresptsDATA, EQresptsMASK, EQtsDATA_factored, EQtsMASK
