"""FE-based seismic response data generation."""
import numpy as np
from tqdm import tqdm
from structure.fem_model import ZeroLengthMDoFDynamicAnalysis, ReadRecord


def DataGeneration(nDOF_Prop, at2filepaths, numberofEQrange, dirfolder,
                   GMfact, recordnodes, acc_dsp_rct, dt_analysis=0.01):
    if numberofEQrange is None:
        numberofEQrange = range(len(at2filepaths))
    failed = []
    pbar = tqdm(at2filepaths, desc=f'[{acc_dsp_rct}]', unit='rec')
    for filepath in pbar:
        result = ZeroLengthMDoFDynamicAnalysis(
            nDOF_Prop,
            GMfilename=filepath,
            dirfolder=dirfolder,
            GMfact=GMfact,
            recordnodelist=recordnodes,
            recorddoflist=[1],
            acc_dsp_rct=acc_dsp_rct,
            dt_analysis=dt_analysis,
        )
        if result != 0:
            failed.append(filepath)
            tqdm.write(f'  >> CONVERGENCE FAILED (code={result}): {filepath}')
    pbar.close()
    if failed:
        print(f'[{acc_dsp_rct}] {len(failed)}/{len(at2filepaths)} records failed to converge.')
    return failed


def _format_value(value):
    if value < 0:
        return f"{value:.6e}  "
    else:
        return f" {value:.6e}  "


def AddZeropad2Input(testat2, zeropadtime=5.0):
    """Append zero-padding to earthquake records for free vibration decay."""
    ns = len(testat2)
    InputGMdatalist = []
    pbar = tqdm(range(ns), desc='[ZeroPad]', unit='rec')
    for tt in pbar:
        EQname = testat2[tt].split('/')[-2]
        EQdir = testat2[tt].split('/')[-1]
        pbar.set_postfix_str(f'{EQname}-{EQdir}')
        dt, Npts = ReadRecord(testat2[tt] + '.AT2', testat2[tt] + '.dat')
        with open(testat2[tt] + '.dat', 'r') as f:
            lines = f.readlines()
        def _safe_float(s):
            try:
                return float(s)
            except ValueError:
                return 0.0
        rows = [list(map(_safe_float, line.split())) for line in lines]
        max_len = max(len(row) for row in rows)
        padded_rows = [row + [0.0] * (max_len - len(row)) for row in rows]
        data = np.array(padded_rows).reshape(-1)[:Npts]

        zeropadN = int(zeropadtime / dt)
        dataAddZeropad = np.append(data, np.zeros(zeropadN))
        Npts += zeropadN
        InputGMdatalist.append(dataAddZeropad)

        if len(dataAddZeropad) % 5 == 0:
            reshaped_data = dataAddZeropad.reshape(-1, 5)
        else:
            reshaped_data = dataAddZeropad[:len(dataAddZeropad) // 5 * 5].reshape(-1, 5)
        remaining_data = data[len(data) // 5 * 5:]

        with open(testat2[tt] + '.AT2', 'r') as f:
            lines = f.readlines()
        lines[3] = f'{Npts}    {dt}    NPTS, DT\n'

        with open(testat2[tt] + '_ZeroPad.AT2', 'w') as f:
            for i in range(4):
                f.write(lines[i])
            for row in reshaped_data:
                f.write(''.join(_format_value(x) for x in row).rstrip() + '\n')
            if len(remaining_data) > 0:
                f.write(''.join(_format_value(x) for x in remaining_data).rstrip() + '\n')
    pbar.close()
    return Npts, dt, InputGMdatalist
