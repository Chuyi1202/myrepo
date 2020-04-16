# estimate c, p, delta, sigma, dt from data

import os

# os.chdir("/Users/zoehu/Desktop/Python_Learning/act_unequal_dt")
os.chdir("/Users/chuyiyu/Desktop/Python_Learning/HFM_IT_Python_Program")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

import time
import itertools
from Parameters import *
import DataPreProcess
import math
from RLsetups import State, choose_action, act, mytiles, q_func, RL_reward
import re
# import importlib
# importlib.reload(BT)
import glob
import copy
import Plots

import Benchmark_trajectory as BT

# =============================================================================
# Import Data
# =============================================================================

# 07.17-08.17 data
f_book = sorted(glob.glob('Data.nosync/MSFT/MSFT_book_0[4-5]*19.csv'), key=lambda x: (x[-6:-4], x[-10:-6]),
                reverse=True)  # Datafiles name, Reverse sorted according to dates
f_msg = sorted(glob.glob('Data.nosync/MSFT/MSFT_orders_0[4-5]*19.csv'), key=lambda x: (x[-6:-4], x[-10:-6]), reverse=True)

# f_book = sorted(glob.glob('Data/MSFT/MSFT_book_*.csv'),key = lambda x: (x[-6:-4],x[-10:-6]),reverse=True) # Datafiles name, Reverse sorted according to dates
# f_msg = sorted(glob.glob('Data/MSFT/MSFT_orders_*.csv'),key = lambda x: (x[-6:-4],x[-10:-6]),reverse=True)

acttime_est = list(range(34200+1800, 34200 + 23400 - 1800, 3))

mismatch = []  # Record data with mismathed book and message
dateprocessed = []  # Record date been successfully processed

day_count = 0

# -------------------------------------------

pi_rec = []
sim_arr_rec = []

acttime_rec = []

pask_rec = []
pbid_rec = []
pos_pp_rec = []
pos_pm_rec = []

plus_p_rec = []
plus_c_rec = []
plus_cp_rec = []
plus_cc_rec = []
plus_ccp_rec = []
plus_ccpp_rec = []

minus_p_rec = []
minus_c_rec = []
minus_cp_rec = []
minus_cc_rec = []
minus_ccp_rec = []
minus_ccpp_rec = []

dltSp_rec = []
dltSm_rec = []
dltStld_rec = []
dltSnone_rec = []

sigmap_rec = []
sigmam_rec = []
sigmatld_rec = []
sigmanone_rec = []

bk = 'Data.nosync/MSFT/MSFT_book_041819.csv'
msg = 'Data.nosync/MSFT/MSFT_orders_041819.csv'
# f_book = f_book [:1]
# f_msg = f_msg [:1]

start_time = time.time()

for bk, msg in zip(f_book,f_msg): # each day is an episode

    if bk[-10:] != msg[-10:]: # Check if the dates of book and message match
        print(bk+" and "+ msg +" don't match")
        mismatch.append([bk,msg])

    else:
        day_count += 1
        dateprocessed.append(msg[-10:-4])

        message_dat = pd.read_csv(msg, header=1)
        book_dat = pd.read_csv(bk, header=1)
        message_dat['Time'] = message_dat['Seconds'] + message_dat['Nanoseconds since last second'] * 10 ** -9
        book_dat['Time'] = book_dat['Seconds'] + book_dat['Nanoseconds'] * 10 ** -9
        message_dat = message_dat.loc[
            (34200 + 23400 - 1800 >= message_dat.Time) & (message_dat.Time >= 34200 + 1800)] # Drop the first and last half hour data
        book_dat = book_dat.loc[
            (34200 + 23400 - 1800 >= book_dat.Time) & (book_dat.Time >= 34200 + 1800)]

        message_dat.Price = message_dat.Price/100
        book_dat.loc[:,["BidPrice1","AskPrice1","BidPrice2","AskPrice2","BidPrice3","AskPrice3","BidPrice4","AskPrice4",
                        "BidPrice5","AskPrice5","BidPrice6","AskPrice6","BidPrice7","AskPrice7","BidPrice8","AskPrice8",
                        "BidPrice9","AskPrice9","BidPrice10","AskPrice10","BidPrice11","AskPrice11","BidPrice12","AskPrice12",
                        "BidPrice13","AskPrice13","BidPrice14","AskPrice14","BidPrice15","AskPrice15","BidPrice16","AskPrice16",
                        "BidPrice17","AskPrice17","BidPrice18","AskPrice18","BidPrice19","AskPrice19","BidPrice20","AskPrice20"]]\
            =book_dat.loc[:,["BidPrice1","AskPrice1","BidPrice2","AskPrice2","BidPrice3","AskPrice3","BidPrice4","AskPrice4",
                        "BidPrice5","AskPrice5","BidPrice6","AskPrice6","BidPrice7","AskPrice7","BidPrice8","AskPrice8",
                        "BidPrice9","AskPrice9","BidPrice10","AskPrice10","BidPrice11","AskPrice11","BidPrice12","AskPrice12",
                        "BidPrice13","AskPrice13","BidPrice14","AskPrice14","BidPrice15","AskPrice15","BidPrice16","AskPrice16",
                        "BidPrice17","AskPrice17","BidPrice18","AskPrice18","BidPrice19","AskPrice19","BidPrice20","AskPrice20"]]/100

        # # =============================================================================
        # # Combine all message orders arriving at the same time
        # # And split buy and sell MO
        # # =============================================================================

        message_new_dat, book_new_dat= DataPreProcess.mergemessage(message_dat,book_dat)

        # Pick out buy and sell market orders
        MO_Plus = message_new_dat.loc[(message_new_dat['Order type']=='MO')&(message_new_dat['Direction']==83)]
        MO_Minus = message_new_dat.loc[(message_new_dat['Order type']=='MO')&(message_new_dat['Direction']==66)]

#         # # =============================================================================
#         # # preset unequally spaced action times based on the following principles:
#         # # 1. pi^+=pi^-=0.4
#         # # 2. The number of arrivals between two consecutive action times is around 1
#         # # Note: we don't set pi(1,1) right now. pi(1,1) is computed after fixing action times
#         # # =============================================================================
#
#         # Split the time interval into 5 equally spaced parts between two consecutive even indexed MOs
#         MO_Plus_act = []
#         MO_Minus_act = []
#
#         for i in range((len(MO_Plus)-1)//2):
#             MO_Plus_act.extend(list(np.linspace(MO_Plus.Time.iloc[2*i],MO_Plus.Time.iloc[2*i+2],5,endpoint=False)))
#         for i in range((len(MO_Minus)-1)//2):
#             MO_Minus_act.extend(list(np.linspace(MO_Minus.Time.iloc[2*i],MO_Minus.Time.iloc[2*i+2],5,endpoint=False)))
#
#         comb = MO_Plus_act+MO_Minus_act
#         comb.sort()
#         acttime = comb[::2]
#         acttime_rec.extend(acttime)

# acttime_rec.sort()
# acttime_est = acttime_rec[::len(f_book)]

        # # =============================================================================
        # # Calculate pi^+ and pi^- and pi(1,1)
        # # =============================================================================

        # How many MOs falling in each action intervals
        count_plus = np.histogram(MO_Plus.Time, bins=acttime_est)[0]  # left closed right open
        count_minus = np.histogram(MO_Minus.Time, bins=acttime_est)[0]
        # calculate pi^+,-
        pip = len(count_plus.nonzero()[0]) / len(count_plus)
        pim = len(count_minus.nonzero()[0]) / len(count_minus)
        pioo = len(set(count_plus.nonzero()[0]) & set(count_minus.nonzero()[0]))/len(count_minus)

        # Moving Average
        rolling_mean_plus = pd.Series(count_plus!=0).rolling(window=100).mean()
        rolling_mean_minus = pd.Series(count_minus!=0).rolling(window=100).mean()
        plt.plot(rolling_mean_plus, label='pi_plus')
        plt.plot(rolling_mean_minus, label='pi_minus')
        plt.plot(rolling_mean_plus-rolling_mean_minus, label='pi_minus')
        plt.legend()
        plt.show()

        pi_rec.append([pip,pim,pioo])

        # See how many orders can arrive between two consecutive actions
        sim_arr_plus = count_plus[np.nonzero(count_plus)]
        sim_arr_minus = count_minus[np.nonzero(count_minus)]
        sim_arr_rec.append([np.mean(sim_arr_plus),np.mean(sim_arr_minus)])

        # # =============================================================================
        # # For each interval [tk,tkk), record size and prices of MOs, Stk, Stkk, MO types P,M or Both
        # # =============================================================================
        # For each day, create a dataframe to record size and prices of MOs, Stk, Stkk, MO types P,M or Both
        df = pd.DataFrame(acttime_est,columns=['Time'])
        df["PlusMO_maxp"] = ""
        df["PlusMO_size"] = ""
        df["MinusMO_minp"] = ""
        df["MinusMO_size"] = ""
        df["Stk"] = ""
        df["Stkk"] = ""

        # Partition the message and book into subintervals according to the action times
        part_msg_p = pd.Series(np.histogram(MO_Plus.Time, bins=acttime_est)[0])
        cum_msg_p = part_msg_p.cumsum()
        cum_msg_p = pd.concat([pd.Series([0]),cum_msg_p,pd.Series([len(MO_Plus)])])

        part_msg_m = pd.Series(np.histogram(MO_Minus.Time, bins=acttime_est)[0])
        cum_msg_m = part_msg_m.cumsum()
        cum_msg_m = pd.concat([pd.Series([0]), cum_msg_m, pd.Series([len(MO_Minus)])])

        part_book = pd.Series(np.histogram(book_dat.Time, bins=acttime_est)[0])
        cum_book = part_book.cumsum()
        cum_book = pd.concat([pd.Series([0]), cum_book, pd.Series([len(book_dat)])])

        for i in range(len(cum_book)-1):

            df.loc[i, 'Stk'] = (book_dat.AskPrice1.iloc[max(0,cum_book.iloc[i] - 1)] + book_dat.BidPrice1.iloc[max(0,cum_book.iloc[i] - 1)]) / 2 # stk is in fact stk-
            df.loc[i, 'Stkk'] = (book_dat.AskPrice1.iloc[cum_book.iloc[i+1] - 1] + book_dat.BidPrice1.iloc[
                cum_book.iloc[i+1] - 1]) / 2

            temp = MO_Plus.iloc[cum_msg_p.iloc[i]:cum_msg_p.iloc[i+1]]
            if len(temp):
                df.loc[i, 'PlusMO_maxp'] = max(temp['Price'])
                # if df.loc[i, 'PlusMO_maxp']<=df.loc[i, 'Stkk']:
                if df.loc[i, 'PlusMO_maxp'] <= df.loc[i, 'Stk']: # new p: max-stk instead of max-stkk
                    df.loc[i, 'PlusMO_size'] = 0
                else:
                    # df.loc[i, 'PlusMO_size'] = -sum(temp.loc[temp.Price > df.loc[i, 'Stkk']].Volume)
                    df.loc[i, 'PlusMO_size'] = -sum(temp.loc[temp.Price > df.loc[i, 'Stk']].Volume)#  new p

            temp = MO_Minus.iloc[cum_msg_m.iloc[i]:cum_msg_m.iloc[i+1]]
            if len(temp):
                df.loc[i, 'MinusMO_minp'] = min(temp['Price'])
                # if df.loc[i, 'MinusMO_minp']>=df.loc[i, 'Stkk']: #  new p: p-=stk-min instead of p-=stkk-min
                if df.loc[i, 'MinusMO_minp'] >= df.loc[i, 'Stk']:
                    df.loc[i, 'MinusMO_size'] = 0
                else:
                    # df.loc[i, 'MinusMO_size'] = -sum(temp.loc[temp.Price < df.loc[i, 'Stkk']].Volume)
                    df.loc[i, 'MinusMO_size'] = -sum(temp.loc[temp.Price < df.loc[i, 'Stk']].Volume)#  new p: p-=stk-min instead of p-=stkk-min

        # Check pi^+-
        pask_rec.append(sum(df.PlusMO_size != '')/len(df)) # pi+
        pbid_rec.append(sum(df.MinusMO_size != '') / len(df)) # pi-

        # Check how many p is positive
        pos_pp_rec.append(sum(df.loc[df.PlusMO_size != ''].PlusMO_size <=0) / sum(df.PlusMO_size != ''))
        pos_pm_rec.append(sum(df.loc[df.MinusMO_size != ''].MinusMO_size <= 0) / sum(df.MinusMO_size != ''))


        for i in range(len(df)):
            if df.loc[i, 'PlusMO_maxp']!='' and df.loc[i, 'MinusMO_minp']=='':
                df.loc[i, 'TypeMO'] = 'P'
            elif df.loc[i, 'PlusMO_maxp']=='' and df.loc[i, 'MinusMO_minp']!='':
                df.loc[i, 'TypeMO'] = 'M'
            elif df.loc[i, 'PlusMO_maxp'] != '' and df.loc[i, 'MinusMO_minp'] != '':
                df.loc[i, 'TypeMO'] = 'Both'

        # # =============================================================================
        # # Estimate the expectation of each parameters
        # # =============================================================================
        # Estimation of c,p:
        plus_df = df[df["TypeMO"].isin(["P","Both"])]

        # Compute p^+. Write negative p^+ as 0
        # p_plus = plus_df.PlusMO_maxp-plus_df.Stkk
        p_plus = plus_df.PlusMO_maxp - plus_df.Stk #new p
        # p_plus[p_plus < 1] = 0
        p_plus.loc[p_plus <= 0] = 0
        df['p_plus']=p_plus


        cp_plus = plus_df.PlusMO_size
        # cp_plus.loc[p_plus == 0] = 0
        df['cp_plus'] = cp_plus

        c_plus=cp_plus.copy()
        c_plus.loc[c_plus != 0] = c_plus.loc[c_plus != 0]/p_plus.loc[p_plus != 0]
        df['c_plus'] = c_plus

        minus_df = df[df["TypeMO"].isin(["M","Both"])]

        # p_minus = minus_df.Stkk-minus_df.MinusMO_minp # new p
        p_minus = minus_df.Stk - minus_df.MinusMO_minp
        # p_minus[p_minus < 1] = 0
        p_minus.loc[p_minus <= 0] = 0
        df['p_minus']=p_minus

        cp_minus = minus_df.MinusMO_size
        # cp_minus.loc[p_minus == 0] = 0
        df['cp_minus'] = cp_minus

        c_minus=cp_minus.copy()
        c_minus.loc[c_minus != 0] = c_minus.loc[c_minus != 0]/p_minus.loc[p_minus != 0]
        df['c_minus'] = c_minus

        # # Estimation of Delta and sigma:
        # P_df = df[df["TypeMO"].isin(["P"])]
        # delta_p = P_df.Stkk - P_df.Stk
        # df['DeltaP'] = delta_p
        #
        # M_df = df[df["TypeMO"].isin(["M"])]
        # delta_m = M_df.Stkk - M_df.Stk
        # df['DeltaM'] = delta_m
        #
        # B_df = df[df["TypeMO"].isin(["Both"])]
        # delta_tild = B_df.Stkk - B_df.Stk
        # df['DeltaTilde'] = delta_tild
        #
        # None_df = df[~df["TypeMO"].isin(["Both",'M',"P"])]
        # delta_none = None_df.Stkk - None_df.Stk
        # df['DeltaNone'] = delta_none

        # Compute mean of parameter_t and store
        plus_p_rec.append((df.p_plus[df.p_plus>0]).mean())
        plus_c_rec.append((df.c_plus[df.c_plus>0]).mean())
        plus_cp_rec.append((df.cp_plus[df.p_plus>0]).mean())
        plus_cc_rec.append(((df.c_plus**2)[df.p_plus>0]).mean())
        plus_ccp_rec.append(((df.c_plus**2*df.p_plus)[df.p_plus>0]).mean())
        plus_ccpp_rec.append(((df.c_plus**2*df.p_plus**2)[df.p_plus>0]).mean())

        minus_p_rec.append((df.p_minus[df.p_minus>0]).mean())
        minus_c_rec.append((df.c_minus[df.p_minus>0]).mean())
        minus_cp_rec.append((df.cp_minus[df.p_minus>0]).mean())
        minus_cc_rec.append(((df.c_minus**2)[df.p_minus>0]).mean())
        minus_ccp_rec.append(((df.c_minus**2*df.p_minus)[df.p_minus>0]).mean())
        minus_ccpp_rec.append(((df.c_minus**2*df.p_minus**2)[df.p_minus>0]).mean())

        # dltSp_rec.append(df.DeltaP.mean())
        # dltSm_rec.append(df.DeltaM.mean())
        # dltStld_rec.append(df.DeltaTilde.mean())
        # dltSnone_rec.append(df.DeltaNone.mean())
        #
        # sigmap_rec.append((df.DeltaP**2).mean())
        # sigmam_rec.append((df.DeltaM**2).mean())
        # sigmatld_rec.append((df.DeltaTilde**2).mean())
        # sigmanone_rec.append((df.DeltaNone**2).mean())

        print('Day Count: ', day_count)

t1=time.time() - start_time
print("--- %s seconds ---" % t1)

pip = np.mean([x[0] for x in pi_rec])
pim = np.mean([x[1] for x in pi_rec])
pioo = np.mean([x[2] for x in pi_rec])

np.mean([x[0] for x in sim_arr_rec])
np.mean([x[1] for x in sim_arr_rec])

# neg_p_ratio = []
# for x in pbid_rec:
#     neg_p_ratio.append(len([k for k in x if k<0])/len(x))
# np.mean(neg_p_ratio)

print("E(pi+): %.2e" % pip)
print("E(pi-): %.2e" % pim)
print("E(pioo): %.2e" % pioo)

print("E(p+): %.2e" % np.mean(plus_p_rec))
print("E(p-): %.2e" % np.mean(minus_p_rec))
print("E(p): %.2e" % ((np.mean(plus_p_rec)+np.mean(minus_p_rec))/2))

print("E(c+): %.2e" % np.mean(plus_c_rec))
print("E(c-): %.2e" % np.mean(minus_c_rec))
print("E(c): %.2e" % ((np.mean(plus_c_rec)+np.mean(minus_c_rec))/2))


print("E(cp+): %.2e" % np.mean(plus_cp_rec))
print("E(cp-): %.2e" % np.mean(minus_cp_rec))
print("E(cp): %.2e" % ((np.mean(plus_cp_rec)+np.mean(minus_cp_rec))/2))

print("E(cc+): %.2e" % np.mean(plus_cc_rec))
print("E(cc-): %.2e" % np.mean(minus_cc_rec))
print("E(cc): %.2e" % ((np.mean(plus_cc_rec)+np.mean(minus_cc_rec))/2))

print("E(ccp+): %.2e" % np.mean(plus_ccp_rec))
print("E(ccp-): %.2e" % np.mean(minus_ccp_rec))
print("E(ccp): %.2e" % ((np.mean(plus_ccp_rec)+np.mean(minus_ccp_rec))/2))

print("E(ccpp+): %.2e" % np.mean(plus_ccpp_rec))
print("E(ccpp-): %.2e" % np.mean(minus_ccpp_rec))
print("E(ccpp): %.2e" % ((np.mean(plus_ccpp_rec)+np.mean(minus_ccpp_rec))/2))

# print("Delta+: %.2f" % np.mean(dltSp_rec))
# print("Delta-: %.2f" % np.mean(dltSm_rec))
# print("Delta_Tilde: %.2f" % np.mean(dltStld_rec))
# print("Delta_None: %.2f" % np.mean(dltSnone_rec))
#
# dt = 3
# print("Sigma^2+: %.2f" % (np.mean(sigmap_rec)/dt))
# print("Sigma^2-: %.2f" % (np.mean(sigmam_rec)/dt))
# print("Sigma^2_Tilde: %.2f" % (np.mean(sigmatld_rec)/dt))
# print("Sigma^2_None: %.2f" % (np.mean(sigmanone_rec)/dt))

# =============================================================================
# Implement on real data
# =============================================================================

# # set parameters as estimated above in cents
# order_size = 1000 # Fixed limit order size by the agent
#
# lmbda = 10 # penalty for the inventory
#
# sigmap = sigmam = 0.53
# tldsigma = 1.07  # (second to second) #(in cents)
#
# muop = muom = 540
# mutp = mutm = 6736706  # mut>=muo^2
# Ep = 2.01
# muoop = muoom = muop * Ep
# mutop = mutom = mutp * Ep
# muttp = muttm = 12930886
#
# I0 = 0 # Initial inventory
# W0 = 0 # Initial cashflow
#
# alphaT = -lmbda
# gT = 0
# hT = 0
# # pip+pim-1V0<=pioo<=pip/\pim
# pip = 0.42
# pim = 0.42
# pioo = 0.24
#
# # Deltap>0, Deltam<0
# Deltap = 0.9
# Deltam = -0.9
#
# S = 0

# # set parameters as estimated above in cents
# # new p
# order_size = 1000  # Fixed limit order size by the agent
#
# #lmbda = 10  # penalty for the inventory
# lmbda = 0.0001  # penalty for the inventory
#
# # sigmap = sigmam = 0.53
# # tldsigma = 1.07  # (second to second) #(in cents)
#
# muop = muom = 628
# mutp = mutm = 1.28 * 10 ** 6  # mut>=muo^2
# Ep = 0.997
# muoop = muoom = muop * Ep
# mutop = mutom = mutp * Ep
# muttp = muttm = 4.15 * 10 ** 6
#
# I0 = 0  # Initial inventory
# W0 = 0  # Initial cashflow
#
# alphaT = -lmbda
# gT = 0
# hT = 0
# # pip+pim-1V0<=pioo<=pip/\pim
# pip = 0.42
# pim = 0.42
# pioo = 0.24

##############################################################
# set parameters as estimated above in cents
# new p # new lmbda # new T # No independent assumption between c and p
order_size = 1000  # Fixed limit order size by the agent

lmbda = 0.001  # penalty for the inventory
# lmbda = 0  # penalty for the inventory
# lmbda = 0.0001  # penalty for the inventory
# sigmap = sigmam = 0.53
# tldsigma = 1.07  # (second to second) #(in cents)

muop = muom = 600
mutp = mutm = 1.08*10**6  # mut>=muo^2
Ep = 0.866
muoop = muoom = muop * Ep
mutop = mutom = mutp * Ep
muttp = muttm = 1.93*10**6

I0 = 0  # Initial inventory
W0 = 0  # Initial cashflow

alphaT = -lmbda
gT = 0
hT = 0
## pip+pim-1V0<=pioo<=pip/\pim
# pip = 0.38
# pim = 0.385
# pioo = 0.205

# # Deltap>0, Deltam<0
# Deltap = 0.9
# Deltam = -0.9


# def trajectory(alphaT, pip, pim, pioo, Deltap, Deltam, sigmap, sigmam, tldsigma, acttime_res):
#     alpha_res = []
#     g_res = []
#     h_res = []
#
#     a1p_res = []
#     a2p_res = []
#     a3p_res = []
#     a1m_res = []
#     a2m_res = []
#     a3m_res = []
#
#     alpha = alphaT
#     g = gT
#     h = hT
#     for k in range(len(time)-1):
#         a1p = BT.A1p(alpha, pioo, pip, pim, muop, muom, mutp, mutm)
#         a1m = BT.A1m(alpha, pioo, pip, pim, muop, muom, mutp, mutm)
#         a2p = BT.A2p(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom)
#         a2m = BT.A2m(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom)
#         a3p = BT.A3p(alpha, pioo, pip, pim, muop, muom, mutp, mutm, Deltap, Deltam, h)
#         a3m = BT.A3m(alpha, pioo, pip, pim, muop, muom, mutp, mutm, Deltap, Deltam, h)
#
#         a1p_res.append(a1p)
#         a2p_res.append(a2p)
#         a3p_res.append(a3p)
#         a1m_res.append(a1m)
#         a2m_res.append(a2m)
#         a3m_res.append(a3m)
#
#         alpha = BT.alpha_updt(alpha, pioo, pip, pim, muop, muom, mutp, mutm, a1p, a1m)
#         h = BT.h_updt(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, Deltap, Deltam, h, a1p,
#                    a1m,
#                    a2p, a2m, a3p, a3m)
#
#         g = BT.g_updt(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, Deltap,
#                    Deltam,
#                    h, g, sigmap, sigmam, tldsigma, acttime_res[k], acttime_res[k + 1], a2p, a2m, a3p, a3m)
#
#         alpha_res.append(alpha)
#         h_res.append(h)
#         g_res.append(g)
#
#     alpha_res.reverse()
#     h_res.reverse()
#     g_res.reverse()
#
#     a1p_res.reverse()
#     a2p_res.reverse()
#     a3p_res.reverse()
#     a1m_res.reverse()
#     a2m_res.reverse()
#     a3m_res.reverse()
#     return alpha_res, h_res, g_res, a1p_res, a2p_res, a3p_res, a1m_res, a2m_res, a3m_res

# def trajectory_new(alphaT, gT, hT, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp,
#                    muttm,time):  # new p
#     alpha_res = []
#     g_res = []
#     h_res = []
#
#     a1p_res = []
#     a2p_res = []
#     a3p_res = []
#     a1m_res = []
#     a2m_res = []
#     a3m_res = []
#
#     alpha = alphaT
#     g = gT
#     h = hT
#     for k in range(len(time) - 1):
#         a1p = BT.A1p(alpha, pioo, pip, pim, muop, muom, mutp, mutm)
#         a1m = BT.A1m(alpha, pioo, pip, pim, muop, muom, mutp, mutm)
#         a2p = BT.A2p(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom)
#         a2m = BT.A2m(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom)
#         # a3p = BT.A3p(alpha, pioo, pip, pim, muop, muom, mutp, mutm, Deltap, Deltam, h)
#         # a3m = BT.A3m(alpha, pioo, pip, pim, muop, muom, mutp, mutm, Deltap, Deltam, h)
#         a3p = BT.A3p_new(alpha, pioo, pip, pim, muop, muom, mutp, mutm, h)
#         a3m = BT.A3m_new(alpha, pioo, pip, pim, muop, muom, mutp, mutm, h)
#
#         a1p_res.append(a1p)
#         a2p_res.append(a2p)
#         a3p_res.append(a3p)
#         a1m_res.append(a1m)
#         a2m_res.append(a2m)
#         a3m_res.append(a3m)
#
#         alpha = BT.alpha_updt(alpha, pioo, pip, pim, muop, muom, mutp, mutm, a1p, a1m)
#         # h = BT.h_updt(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, Deltap, Deltam, h, a1p,
#         #            a1m,
#         #            a2p, a2m, a3p, a3m)
#
#         h = BT.h_updt_new(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop,mutom, h, a1p, a1m, a2p, a2m,
#                           a3p, a3m)
#
#         # g = BT.g_updt(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, Deltap,
#         #            Deltam,
#         #            h, g, sigmap, sigmam, tldsigma, acttime_res[k], acttime_res[k + 1], a2p, a2m, a3p, a3m)
#
#         g = BT.g_updt_new(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm,
#                           h, g, a2p, a2m, a3p, a3m)
#
#         alpha_res.append(alpha)
#         h_res.append(h)
#         g_res.append(g)
#
#     alpha_res.reverse()
#     h_res.reverse()
#     g_res.reverse()
#
#     a1p_res.reverse()
#     a2p_res.reverse()
#     a3p_res.reverse()
#     a1m_res.reverse()
#     a2m_res.reverse()
#     a3m_res.reverse()
#     return alpha_res, h_res, g_res, a1p_res, a2p_res, a3p_res, a1m_res, a2m_res, a3m_res


# time = acttime_est.copy()
# # time.append(57600)
# time.append(57600-1800)
# time = time[::-1]
# # alpha_res, h_res, g_res, a1p_res, a2p_res, a3p_res, a1m_res, a2m_res, a3m_res = trajectory(alphaT, pip, pim, pioo, Deltap, Deltam, sigmap, sigmam, tldsigma, time)
# alpha_res, h_res, g_res, a1p_res, a2p_res, a3p_res, a1m_res, a2m_res, a3m_res = trajectory_new(alphaT, gT, hT, pioo,
#                                                                                                pip, pim, muop, muom,
#                                                                                                mutp, mutm, muoop, muoom,
#                                                                                                mutop, mutom, muttp,
#                                                                                                muttm,time)

def executed(p_a, p_b, message, order_size):
    # accumulative number of executions during dt time slot given the price of limit orders placed at the beginning of dt
    # p_a, p_b are the price of agent's placement
    # book (df) is a subset of the whole book, starting from the time when the action is taken and ending at the time of the next action
    # Similarly, message (df) is a subset of the whole message data,
    # order_size is the fixed order size by the agent

    # Note: Assume the agent's LO is always ahead of queue, because it's placed at the beginning of dt
    executed_a = 0
    executed_b = 0

    messageMO = message.loc[message['Order type'] == 'MO']  # pickout MO in message set

    for index, row in messageMO.iterrows():
        if row.Direction == 83 and row.Price >= p_a:  # on ask side # Should we make it row.Price > p_a to make it consistent with the Q function?
            executed_a -= row.Volume  # the number of executions
        elif row.Direction == 66 and row.Price <= p_b:  # Same on the bid side
            executed_b -= row.Volume  # the number of executions

    executed_a = min(order_size, executed_a)
    executed_b = min(order_size, executed_b)

    return executed_a, executed_b

def Liquid_cost(book_dat,IT):
    # This function compute the liquidating cost of the terminal inventory at the end of trading horizon

    cost = 0
    book_timeT = book_dat.iloc[-1]

    if IT>0:
        i = 1
        while book_timeT["BidSize"+str(i)]<IT and i<20: # Data only has 20 levels
            cost += book_timeT["BidSize"+str(i)]*book_timeT["BidPrice"+str(i)]
            IT -= book_timeT["BidSize"+str(i)]
            i += 1
        cost += book_timeT["BidPrice"+str(i)] * IT

    if IT<0:
        tempIT = -IT
        i = 1
        while book_timeT["AskSize"+str(i)]<tempIT and i<20: # Data only has 20 levels
            cost -= book_timeT["AskSize"+str(i)]*book_timeT["AskPrice"+str(i)]
            tempIT -= book_timeT["AskSize"+str(i)]
            i += 1
        cost -= book_timeT["AskPrice"+str(i)] * tempIT

    return cost

def Benchmark_reward_intraday(l, W0, I0, message_new_dat, book_dat, acttime_est, order_size, a1p_res, a2p_res, a3p_res,
                              a1m_res, a2m_res, a3m_res):
    # Reward of optimal control
    # compared with strategy that places limit orders at level l
    part_msg = pd.Series(np.histogram(message_new_dat.Time, bins=acttime_est)[0])
    cum_msg = list(part_msg.cumsum())
    cum_msg.insert(0, 0)
    cum_msg.append(len(message_new_dat) - 1)

    part_book = pd.Series(np.histogram(book_dat.Time, bins=acttime_est)[0])
    cum_book = list(part_book.cumsum())
    cum_book.insert(0, 0)
    cum_book.append(len(book_new_dat) - 1)

    W = W0
    Wl = [W0] * l

    I = I0
    Il = [I0] * l

    # store W and I at each arriving time
    W_list = []
    Wl_list = [[] for _ in range(l)]

    I_list = []
    Il_list = [[] for _ in range(l)]

    price_a = [] # Price trajectory for optimal
    price_al_list = [[] for _ in range(l)] # Price trajectory for 6 benchmarks (level1-level6)

    price_b = []
    price_bl_list = [[] for _ in range(l)]

    S = [] # trajectory of asset price

    for k in range(len(cum_msg) - 1):
        message = message_new_dat.iloc[cum_msg[k]:cum_msg[k + 1]]
        book = book_dat.iloc[cum_book[k]:cum_book[k + 1]]

        stk = (book_dat.AskPrice1.iloc[max(cum_book[k] - 1, 0)] + book_dat.BidPrice1.iloc[
            max(cum_book[k] - 1, 0)]) / 2
        S.append(stk)
        stkk = (book_dat.AskPrice1.iloc[cum_book[k + 1] - 1] + book_dat.BidPrice1.iloc[
            cum_book[k + 1] - 1]) / 2
        [a, b] = BT.optimalstrategy(stk, I, a1p_res[k], a1m_res[k], a2p_res[k], a2m_res[k], a3p_res[k], a3m_res[k]) # The placement is based on Stk-

        # Compare with the strategy placing orders at first and second levels
        abl = []
        for i in range(1, l + 1):
            abl.append([book_dat['AskPrice' + str(i)].iloc[max(cum_book[k] - 1, 0)], # The placement is based on Stk-
                        book_dat['BidPrice' + str(i)].iloc[max(cum_book[k] - 1, 0)]])

        if len(book) != 0:

            i = I
            il = Il.copy()

            delta_m = stkk - stk
            executed_a, executed_b = executed(a, b, message, order_size)
            I = i + executed_b - executed_a
            W = W + executed_a * a - executed_b * b

            excuted_abl = []
            for i in range(l):
                executed_a, executed_b = executed(abl[i][0], abl[i][1], message, order_size)
                Il[i] = il[i] + executed_b - executed_a
                Wl[i] = Wl[i] + executed_a * abl[i][0] - executed_b * abl[i][1]

        W_list.append(W)
        I_list.append(I)
        price_a.append(a)
        price_b.append(b)

        for i in range(l):
            Wl_list[i].append(Wl[i])
            Il_list[i].append(Il[i])
            price_al_list[i].append(abl[i][0])
            price_bl_list[i].append(abl[i][1])

    res = []

    df = pd.DataFrame({'CashFlow': W_list, 'Inventory': I_list})
    df_price = pd.DataFrame({'Ask_Price': price_a, 'Bid_Price': price_b})
    S.append(stkk)

    res.append([df, df_price])

    for i in range(l):
        res.append([pd.DataFrame({'CashFlow': Wl_list[i], 'Inventory': Il_list[i]}),
                    pd.DataFrame({'Ask_Price': price_al_list[i], 'Bid_Price': price_bl_list[i]})])

    return res, S


l = 6

IT = [[] for _ in range(l + 1)]
WT = [[] for _ in range(l + 1)]
liquid_cost = [[] for _ in range(l + 1)]

ST = []
day_count = 0
#0401,0425,0430
bk = 'Data.nosync/MSFT/MSFT_book_041219.csv'
msg = 'Data.nosync/MSFT/MSFT_orders_041219.csv'
#
# bk ='Data/MSFT/MSFT_book_040119.csv'
# msg = 'Data/MSFT/MSFT_orders_040119.csv'

# f_book = f_book[:5]
# f_msg = f_msg[:5]

for bk, msg in zip(f_book, f_msg):  # each day is an episode

    if bk[-10:] != msg[-10:]:  # Check if the dates of book and message match
        print(bk + " and " + msg + " don't match")
        mismatch.append([bk, msg])

    else:
        day_count += 1
        dateprocessed.append(msg[-10:-4])

        message_dat = pd.read_csv(msg, header=1)
        book_dat = pd.read_csv(bk, header=1)
        message_dat['Time'] = message_dat['Seconds'] + message_dat['Nanoseconds since last second'] * 10 ** -9
        book_dat['Time'] = book_dat['Seconds'] + book_dat['Nanoseconds'] * 10 ** -9


        MA_pi = 100 # number of time slots to calculate pi. (MA)
        message_dat = message_dat.loc[
            (34200 + 23400 - 1800 >= message_dat.Time) & (message_dat.Time >= 34200 + 1800 - 3*MA_pi)] # Drop the first and last half hour data, but keep 300s before 10:00 in order to calculate moving average of pi
        book_dat = book_dat.loc[
            (34200 + 23400 - 1800 >= book_dat.Time) & (book_dat.Time >= 34200 + 1800- 3*MA_pi)]

        message_dat.Price = message_dat.Price / 100
        book_dat.loc[:,
        ["BidPrice1", "AskPrice1", "BidPrice2", "AskPrice2", "BidPrice3", "AskPrice3", "BidPrice4", "AskPrice4",
         "BidPrice5", "AskPrice5", "BidPrice6", "AskPrice6", "BidPrice7", "AskPrice7", "BidPrice8", "AskPrice8",
         "BidPrice9", "AskPrice9", "BidPrice10", "AskPrice10", "BidPrice11", "AskPrice11", "BidPrice12", "AskPrice12",
         "BidPrice13", "AskPrice13", "BidPrice14", "AskPrice14", "BidPrice15", "AskPrice15", "BidPrice16", "AskPrice16",
         "BidPrice17", "AskPrice17", "BidPrice18", "AskPrice18", "BidPrice19", "AskPrice19", "BidPrice20",
         "AskPrice20"]] \
            = book_dat.loc[:,
              ["BidPrice1", "AskPrice1", "BidPrice2", "AskPrice2", "BidPrice3", "AskPrice3", "BidPrice4", "AskPrice4",
               "BidPrice5", "AskPrice5", "BidPrice6", "AskPrice6", "BidPrice7", "AskPrice7", "BidPrice8", "AskPrice8",
               "BidPrice9", "AskPrice9", "BidPrice10", "AskPrice10", "BidPrice11", "AskPrice11", "BidPrice12",
               "AskPrice12",
               "BidPrice13", "AskPrice13", "BidPrice14", "AskPrice14", "BidPrice15", "AskPrice15", "BidPrice16",
               "AskPrice16",
               "BidPrice17", "AskPrice17", "BidPrice18", "AskPrice18", "BidPrice19", "AskPrice19", "BidPrice20",
               "AskPrice20"]] / 100

        # # =============================================================================
        # # Combine all message orders arriving at the same time
        # # And split buy and sell MO
        # # =============================================================================

        message_new_dat, book_new_dat = DataPreProcess.mergemessage(message_dat, book_dat)

        # Pick out buy and sell market orders
        MO_Plus = message_new_dat.loc[(message_new_dat['Order type'] == 'MO') & (message_new_dat['Direction'] == 83)]
        MO_Minus = message_new_dat.loc[(message_new_dat['Order type'] == 'MO') & (message_new_dat['Direction'] == 66)]

        # Calculate pi_t (MA)
        ##################
        # How many MOs falling in each action intervals
        acttime_temp = list(range(34200+1800-3*MA_pi, 34200 + 23400 - 1800 + 3, 3))
        count_plus = np.histogram(MO_Plus.Time, bins=acttime_temp)[0]  # left closed right open
        count_minus = np.histogram(MO_Minus.Time, bins=acttime_temp)[0]

        # Moving Average
        rolling_mean_plus = pd.Series(count_plus != 0).rolling(window=100).mean().iloc[100:]
        rolling_mean_minus = pd.Series(count_minus != 0).rolling(window=100).mean().iloc[100:]
        rolling_mean_oo = pd.Series((count_minus != 0)&(count_plus != 0)).rolling(window=100).mean().iloc[100:]

        # Delete the first 300s after obtaining pi_t
        message_dat = message_dat.loc[
            (34200 + 23400 - 1800 >= message_dat.Time) & (
                        message_dat.Time >= 34200 + 1800)]  # Drop the first and last half hour data, but keep 300s before 10:00 in order to calculate moving average of pi
        book_dat = book_dat.loc[
            (34200 + 23400 - 1800 >= book_dat.Time) & (book_dat.Time >= 34200 + 1800)]

        message_new_dat = message_new_dat.loc[(34200 + 23400 - 1800 >= message_new_dat.Time) & (message_new_dat.Time >= 34200 + 1800)]
        book_new_dat = book_new_dat.loc[
            (34200 + 23400 - 1800 >= book_new_dat.Time) & (book_new_dat.Time >= 34200 + 1800)]

        MO_Plus =  MO_Plus.loc[(34200 + 23400 - 1800 >=  MO_Plus.Time) & ( MO_Plus.Time >= 34200 + 1800)]
        MO_Minus = MO_Minus.loc[(34200 + 23400 - 1800 >= MO_Minus.Time) & (MO_Minus.Time >= 34200 + 1800)]
        ##################
        alpha_res, h_res, g_res, a1p_res, a2p_res, a3p_res, a1m_res, a2m_res, a3m_res = BT.trajectory_new_v2(alphaT, gT, hT, rolling_mean_plus, rolling_mean_minus, rolling_mean_oo, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, acttime_est)

        output = Benchmark_reward_intraday(l, W0, I0, message_new_dat, book_dat, acttime_est, order_size, a1p_res,
                                           a2p_res, a3p_res, a1m_res, a2m_res, a3m_res)
        for i in range(l + 1):
            IT[i].append(output[0][i][0].Inventory.iloc[-1])
            WT[i].append(output[0][i][0].CashFlow.iloc[-1])
            liquid_cost[i].append(Liquid_cost(book_dat,IT[i][-1])) # liquidating cost

        ST.append(output[1][-1])

        print('Day Count: ', day_count)

print('lmbda 0')


# ax = fig.add_subplot(gs[:, 0])
# for i in range(len(alphaT)):
#     ax.plot(time_plot[:-1], ab_res[i][0], color=c[i])
#     ax.plot(time_plot[:-1], ab_res[i][1], color=c[i])
# ax.set_ylabel('Bid-Ask Price')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])


# plt.plot(res[1].Ask_Price-res[2],label = 'Ask')
# plt.plot(res[0].Inventory,label = 'Inventory')
# plt.legend()
#
# stkk = (book_dat.AskPrice1.iloc[- 1] + book_dat.BidPrice1.iloc[- 1]) / 2
#
# plt.plot(res[1].Bid_Price-res[2],label = 'Bid')
# # plt.plot(res[2],label = 'St')
# plt.legend()
#

# -sum(MO_Plus.Volume[(MO_Plus.Time<36250)&(MO_Plus.Time>=36000)])
# -sum(MO_Minus.Volume[(MO_Minus.Time<36250)&(MO_Minus.Time>=36000)])

-sum(MO_Plus.Volume[(MO_Plus.Time>=54000)])
-sum(MO_Minus.Volume[(MO_Minus.Time>=54000)])

-sum(MO_Plus.Volume)
-sum(MO_Minus.Volume)

from labellines import labelLine, labelLines
import datetime
# Inventory trajectory within one specific day
plt.figure(figsize=(20,8))
for i in reversed(range(1,l+1)):
    plt.plot(acttime_est, output[0][i][0].Inventory, label='Level'+str(i),color=str(i/(l+1)))
plt.plot(acttime_est,output[0][0][0].Inventory,label = 'Optimal Control',color='red')
# labelLines(plt.gca().get_lines(),xvals=(2000,6000),fontsize=10,align=False)
plt.legend()
locs, labels = plt.xticks()
plt.xticks(locs,[str(datetime.timedelta(seconds=x)) for x in locs], rotation=20)
plt.title('Intraday Inventory')
plt.show()

# optimal control trajectory within one specific day
# c = plt.rcParams['axes.prop_cycle'].by_key()['color']  # get default colormap in matlibplot such that each ask bid pair shares same color
# plt.ylim((-10,10))
plt.figure(figsize=(20,8))
plt.plot(acttime_est,output[0][0][1].Ask_Price - output[1][:-1], label='Optimal Ask',color='red')
plt.plot(acttime_est,output[0][0][1].Bid_Price - output[1][:-1], label='Optimal Bid',color='red')
# for i in range(1,l+1):
for i in range(1,3):
    plt.plot(acttime_est,output[0][i][1].Ask_Price - output[1][:-1], label='Level'+str(i) +'Ask',color=str(i/(l+1)))
    plt.plot(acttime_est,output[0][i][1].Bid_Price - output[1][:-1], label='Level'+str(i) +'Bid',color=str(i/(l+1)))
lines = plt.gca().get_lines()
labelLines([lines[0],lines[2],lines[4]], xvals=(acttime_est[0], acttime_est[6000]), zorder=1,  align=False, ha='left', va='top')
labelLines([lines[1],lines[3],lines[5]], xvals=(acttime_est[0], acttime_est[6000]), zorder=1,  align=False, ha='left', va='bottom')
# plt.legend(loc = 3)
locs, labels = plt.xticks()
plt.xticks(locs,[str(datetime.timedelta(seconds=x)) for x in locs], rotation=20)
plt.show()


# # plot difference between optimal and first level strategy in inventory and price
# fig, ax1 = plt.subplots()
#
# color = 'tab:red'
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('Inventory difference', color=color)
# # ax1.plot(output[0][1][0].Inventory-output[0][0][0].Inventory, color=color)
# ax1.plot((output[0][1][0].Inventory-output[0][0][0].Inventory).diff(), color=color)
# ax1.tick_params(axis='y', labelcolor=color)
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# color = 'tab:green'
# ax2.set_ylabel('Price Difference', color=color)  # we already handled the x-label with ax1
# ax2.plot(output[0][1][1].Ask_Price-output[0][0][1].Ask_Price, color=color,label = 'ask')
# ax2.plot(output[0][1][1].Bid_Price-output[0][0][1].Bid_Price, color='blue',label = 'bid')
# ax2.tick_params(axis='y', labelcolor=color)
# ax2.legend()
#
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()

# temp_bk = book_dat.loc[(book_dat.Time>=acttime_est[2883])&(book_dat.Time<acttime_est[2885])]
# temp_MOp = MO_Plus.loc[(MO_Plus.Time>=acttime_est[2883])&(MO_Plus.Time<acttime_est[2885])]
# temp_MOm = MO_Minus.loc[(MO_Minus.Time>=acttime_est[2883])&(MO_Minus.Time<acttime_est[2885])]

#
# plt.plot(res[0].CashFlow, label = 'Intraday Cashflow')
# plt.plot(res[3].CashFlow, label = 'Intraday Cashflow1')
# plt.plot(res[6].CashFlow, label = 'Intraday Cashflow2')
# plt.legend()
#
# res[0].Inventory.iloc[-1]*stkk+ res[0].CashFlow.iloc[-1]

# Average reward over 27 days
Vl = []
for j in range(l+1):
    Vl.append([IT[j][i] * ST[i] + WT[j][i] - lmbda*IT[j][i]**2 for i in range(len(IT[0]))]) # IT*ST+WT-lambda*IT^2
    # Vl.append([IT[j][i] * ST[i] + WT[j][i] for i in range(len(IT[0]))])  # IT*ST+WT-lambda*IT^2
    # Vl.append([WT[j][i]+liquid_cost[j][i] for i in range(len(IT[0]))])  # IT*ST+WT-lambda*IT^2
    # Vl.append([liquid_cost[j][i] for i in range(len(IT[0]))])  # IT*ST+WT-lambda*IT^2
    # Vl.append([IT[j][i] * ST[i] - lmbda*IT[j][i]**2 for i in range(len(IT[0]))]) # IT*ST+WT-lambda*IT^2
    # Vl.append([IT[j][i] * ST[i] for i in range(len(IT[0]))]) # IT*ST+WT-lambda*IT^2
    # Vl.append([WT[j][i] for i in range(len(IT[0]))])
    # Vl.append([IT[j][i] for i in range(len(IT[0]))]) # IT
    # Vl.append([IT[j][i]**2 for i in range(len(IT[0]))])  # IT^2
mean = [np.mean(x) for x in Vl]
print('Mean: '+'\t'.join(['{:.2e}'.format(x) for x in mean]))
std = [np.std(x) for x in Vl]
print('Std: '+'\t'.join(['{:.2e}'.format(x) for x in std]))
ratio = []
for i in range(l):
    ratio.append(sum([x>=y for x,y in zip(Vl[0],Vl[i+1])])/len(Vl[0]))
    # ratio.append(sum([abs(x)<=abs(y) for x, y in zip(Vl[0], Vl[i + 1])]) / len(Vl[0]))
print('Ratio: '+'\t'.join(['{:.2e}'.format(x) for x in ratio]))

plt.plot(Vl[0],label = 'Optimal Control',color='red')
for j in range(1,l+1):
    plt.plot(Vl[j], label='Level'+str(j),color=str(j/(l+1)))
plt.legend()
plt.title(r'$W_T+I_T*S_T-\lambda*I_T^2$')
plt.plot()

# Plot W_T vs (S_T-lambda*I_T)*IT over 27 days
tempW = [WT[0][i] for i in range(len(IT[0]))]
tempLiqIT = [IT[0][i] * ST[i] - lmbda*IT[0][i]**2 for i in range(len(IT[0]))]
tempsum = [WT[0][i]+IT[0][i] * ST[i] - lmbda*IT[0][i]**2 for i in range(len(IT[0]))]
plt.plot(tempW,label = r'$W_T$')
plt.plot(tempLiqIT,label = r'$(S_T-\lambda I_T)*I_T$')
plt.plot(tempsum,label = r'$W_T+(S_T-\lambda I_T)*I_T$')
plt.hlines(0,0,len(tempW))
plt.legend()

# V_IT = [IT[i] * ST[i] for i in range(len(IT))]
# V = [V_IT[i] + WT[i] for i in range(len(IT))]

# V1_IT = [IT1[i]*ST[i] for i in range(len(IT))]
# V1 = [V1_IT[i]+WT1[i] for i in range(len(IT))]
#
# V2_IT = [IT2[i]*ST[i] for i in range(len(IT))]
# V2 = [V2_IT[i]+WT2[i] for i in range(len(IT))]
#
# plt.plot(V,label = 'CashFlow+ST*IT')
# plt.plot(V1,label = 'CashFlow1+ST*IT1')
# plt.plot(V2,label = 'CashFlow2+ST*IT2')
# plt.legend()
#
# V_penalty = [V_IT[i]+WT[i]-lmbda*IT[i]**2 for i in range(len(IT))]
# plt.plot(IT)
# plt.plot(V,label = 'CashFlow+ST*IT')
# plt.plot(WT,label='CashFlow')
# plt.plot(V_penalty,label='CashFlow+ST*IT-penalty')
# plt.legend()

# #---------- Plot Mid-Price and Inventory of optimal control in one graph ---------------------------------------
# fig, ax1 = plt.subplots()
#
# color = 'tab:red'
# ax1.set_xlabel('time (s)')
# ax1.set_ylabel('Mid-Price', color=color)
# ax1.plot(book_new_dat.Time,(book_new_dat.AskPrice1+book_new_dat.BidPrice1)/2, color=color)
# ax1.tick_params(axis='y', labelcolor=color)
#
# ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
#
# color = 'tab:blue'
# ax2.set_ylabel('Inventory of Optimal Control', color=color)  # we already handled the x-label with ax1
# ax2.plot(acttime_est,output[0][0][0].Inventory, color=color)
# ax2.tick_params(axis='y', labelcolor=color)
#
# fig.tight_layout()  # otherwise the right y-label is slightly clipped
# plt.show()

#---------- Plot Mid-Price and Inventory of first level strategy and optimal control in one graph -------------------------------------------------------------------------

fig, ax1 = plt.subplots()

ax1.set_xlabel('time (s)')
ax1.set_ylabel('Mid-Price', color='green')
ax1.plot(book_new_dat.Time,(book_new_dat.AskPrice1+book_new_dat.BidPrice1)/2, color='green',label = r'$S_T$')
ax1.tick_params(axis='y', labelcolor='green')

ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis

ax2.set_ylabel('Inventory', color='red')  # we already handled the x-label with ax1
for i in reversed(range(1,l+1)):
    ax2.plot(acttime_est, output[0][i][0].Inventory, label='Level'+str(i),color=str(i/(l+1)))
ax2.plot(acttime_est, output[0][0][0].Inventory,label = 'Optimal Control',color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.legend(framealpha=0.3,loc = 4)

fig.tight_layout()  # otherwise the right y-label is slightly clipped
plt.show()


# # See within 3 second, what does S_MO-S_0 looks like
# part_S = pd.Series(np.histogram(book_new_dat.Time, bins=acttime_est)[0])
# cum_S = list(part_S.cumsum())
# cum_S.insert(0, 0)
# cum_S.append(len(book_new_dat) - 1)
#
# delta_S = []
# for k in range(len(cum_S) - 1):
#     book_dt = book_new_dat.iloc[cum_S[k]:cum_S[k + 1]]
#     message_part = message_new_dat.iloc[cum_S[k]:cum_S[k + 1]]
#     MO_dt = book_dat.loc[message_part.loc[(message_part['Order type']=='MO')].index-1]
#     S_dt = (MO_dt.AskPrice1+MO_dt.BidPrice1)/2
#     delta_S.extend([x-(book_new_dat.iloc[cum_S[k]-1].AskPrice1+book_new_dat.iloc[cum_S[k]-1].BidPrice1)/2 for x in S_dt])
#
# sum([x>0 for x in delta_S])/len(delta_S)
plt.plot(output[0][0][0].CashFlow)
plt.hlines(0,0,len(output[0][0][0].CashFlow))