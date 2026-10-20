"""Pure v2.2 decision from paired seed means; test never selects settings."""
import math

def judge(baseline,candidate):
    delta={s:{k:candidate[s][k]-baseline[s][k] for k in baseline[s]} for s in baseline}
    ratio=candidate['FR']['OBS_RMSE_DN']/baseline['FR']['OBS_RMSE_DN'] if baseline['FR']['OBS_RMSE_DN']>0 else (1. if candidate['FR']['OBS_RMSE_DN']==0 else None)
    primary=delta['RR']['PSNR']>0 and delta['RR']['SAM']<0 and delta['RR']['ERGAS']<0
    guards={'RR2':delta['RR2']['PSNR']>=-.05 and delta['RR2']['SAM']<=.06,
            'FR_WALD':delta['FR_WALD']['PSNR']>=-.07 and delta['FR_WALD']['SAM']<=.04,
            'FR_OBS':ratio is not None and ratio<=1.10}
    some=delta['RR']['PSNR']>0 or delta['RR']['SAM']<0 or delta['RR']['ERGAS']<0
    return {'criteria_version':'v2.2','delta':delta,'observation_ratio':ratio,'primary':primary,'guards':guards,
            'label':'개선' if primary and all(guards.values()) else ('부분 개선' if some else '개선 없음'),
            'QNR_used_for_decision':False,'FR_ground_truth_available':False}
