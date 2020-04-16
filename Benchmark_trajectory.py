import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
from labellines import labelLine, labelLines
from matplotlib.ticker import FormatStrFormatter

from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


def Denominator(alpha, pioo, pip, pim, muop, muom, mutp, mutm):
    return (pioo * alpha * muop * muom) ** 2 - pip * pim * (alpha * mutp - muop) * (alpha * mutm - muom)


def A1p(alpha, pioo, pip, pim, muop, muom, mutp, mutm):
    return (pip * pim * alpha * muop * (
            alpha * mutm - muom) - pim * pioo * alpha ** 2 * muop * muom ** 2) / Denominator(alpha, pioo, pip, pim,
                                                                                             muop, muom, mutp, mutm)


def A1m(alpha, pioo, pip, pim, muop, muom, mutp, mutm):
    return (pip * pim * alpha * muom * (
            alpha * mutp - muop) - pip * pioo * alpha ** 2 * muom * muop ** 2) / Denominator(alpha, pioo, pip, pim,
                                                                                             muop, muom, mutp, mutm)


def A2p(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom):
    return (pim * (alpha * mutm - muom) * (
            pip * (muoop - 2 * alpha * mutop) + 2 * pioo * alpha * muop * muoom) + pioo * alpha * muop * muom * (
                    pim * (muoom - 2 * alpha * mutom) + 2 * alpha * pioo * muoop * muom)) / (
                   2 * Denominator(alpha, pioo, pip, pim, muop, muom, mutp, mutm))


def A2m(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom):
    return (pip * (alpha * mutp - muop) * (
            pim * (muoom - 2 * alpha * mutom) + 2 * pioo * alpha * muom * muoop) + pioo * alpha * muom * muop * (
                    pip * (muoop - 2 * alpha * mutop) + 2 * alpha * pioo * muoom * muop)) / (
                   2 * Denominator(alpha, pioo, pip, pim, muop, muom, mutp, mutm))


def A3p(alpha, pioo, pip, pim, muop, muom, mutp, mutm, Deltap, Deltam, h):
    return (pim * pip * (alpha * mutm - muom) * (
            2 * muop * Deltap + h * muop - 2 * alpha * mutp * Deltap) - pioo * alpha * muop * muom * pim * (
                    2 * muom * Deltam + h * muom - 2 * alpha * mutm * Deltam)) / (
                   2 * Denominator(alpha, pioo, pip, pim, muop, muom, mutp, mutm))

def A3p_new(alpha, pioo, pip, pim, muop, muom, mutp, mutm, h): # new p
    return (pip * pim * (alpha * mutm - muom) * (h * muop) - pioo * alpha * muop * muom * pim * (+ h * muom)) / (
            2 * Denominator(alpha, pioo, pip, pim, muop, muom, mutp, mutm))


def A3m(alpha, pioo, pip, pim, muop, muom, mutp, mutm, Deltap, Deltam, h):
    return (pim * pip * (alpha * mutp - muop) * (
            2 * muom * Deltam + h * muom - 2 * alpha * mutm * Deltam) - pioo * alpha * muom * muop * pip * (
                    2 * muop * Deltap + h * muop - 2 * alpha * mutp * Deltap)) / (
                   2 * Denominator(alpha, pioo, pip, pim, muop, muom, mutp, mutm))

def A3m_new(alpha, pioo, pip, pim, muop, muom, mutp, mutm, h): # new p
    return (pim * pip * (alpha * mutp - muop) * (h * muom) - pioo * alpha * muom * muop * pip * (+ h * muop)) / (
            2 * Denominator(alpha, pioo, pip, pim, muop, muom, mutp, mutm))


def alpha_updt(alpha, pioo, pip, pim, muop, muom, mutp, mutm, a1p, a1m):
    return alpha + pip * ((alpha * mutp - muop) * a1p ** 2 + 2 * alpha * muop * a1p) + pim * (
            (alpha * mutm - muom) * a1m ** 2 + 2 * alpha * muom * a1m) + 2 * alpha * pioo * muop * muom * a1p * a1m


def h_updt(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, Deltap, Deltam, h, a1p, a1m, a2p,
           a2m, a3p, a3m):
    return h + pip * ((alpha * mutp - muop) * 2 * a1p * (a2p + a3p) + 2 * alpha * muop * (a2p + a3p) + a1p * (
            muoop + 2 * muop * Deltap + h * muop + alpha * (-2 * mutp * Deltap - 2 * mutop)) - alpha * (
                              2 * muop * Deltap + 2 * muoop)) + pim * ((
                                                                               alpha * mutm - muom) * (-2) * a1m * (
                                                                               a2m - a3m) - 2 * alpha * muom * (
                                                                               a2m - a3m) - a1m * (
                                                                               muoom - 2 * muom * Deltam - h * muom + alpha * (
                                                                               2 * mutm * Deltam - 2 * mutom)) - alpha * (
                                                                               2 * muom * Deltam - 2 * muoom)) + 2 * alpha * pioo * (
                   -muop * muom * (a1p * (a2m - a3m) - a1m * (a2p + a3p)) - muoop * muom * a1m + muop * muoom * a1p)

def h_updt_new(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, h, a1p, a1m, a2p,
           a2m, a3p, a3m): # new p
    return h + pip * ((alpha * mutp - muop) * 2 * a1p * (a2p + a3p) + 2 * alpha * muop * (a2p + a3p) + a1p * (
            muoop  + h * muop + alpha * (- 2 * mutop)) - alpha * (
                               + 2 * muoop)) + pim * ((alpha * mutm - muom) * (-2) * a1m * (
                                                                               a2m - a3m) - 2 * alpha * muom * (
                                                                               a2m - a3m) - a1m * (
                                                                               muoom  - h * muom + alpha * (
                                                                                - 2 * mutom)) - alpha * (
                                                                                - 2 * muoom)) + 2 * alpha * pioo * (
                   -muop * muom * (a1p * (a2m - a3m) - a1m * (a2p + a3p)) - muoop * muom * a1m + muop * muoom * a1p)

def g_updt(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, Deltap, Deltam, h,
           g, sigmap, sigmam, tldsigma, tkk, tk, a2p, a2m, a3p, a3m):
    return g + pip * ((alpha * mutp - muop) * (a2p + a3p) ** 2 + (
            muoop + 2 * muop * Deltap + h * muop + alpha * (-2 * mutp * Deltap - 2 * mutop)) * (
                              a2p + a3p) + alpha * (
                              mutp * sigmap ** 2 * (tkk - tk) + muttp + 2 * mutop * Deltap) - muop * sigmap ** 2 * (
                              tkk - tk) - muoop * Deltap - h * muop * Deltap - h * muoop) + pim * (
                   (alpha * mutm - muom) * (a2m - a3m) ** 2 + (
                   muoom - 2 * muom * Deltam - h * muom + alpha * (2 * mutm * Deltam - 2 * mutom)) * (
                           a2m - a3m) + alpha * (mutm * sigmam ** 2 * (
                   tkk - tk) + muttm - 2 * mutom * Deltam) - muom * sigmam ** 2 * (
                           tkk - tk) + muoom * Deltam - h * muom * Deltam + h * muoom) + 2 * alpha * pioo * (
                   -muop * muom * (a2p + a3p) * (a2m - a3m) + muoop * muom * (a2m - a3m) + muoom * muop * (
                   a2p + a3p) + muop * muom * tldsigma ** 2 * (tkk - tk) - muoop * muoom)

def g_updt_new(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, h,
           g,a2p, a2m, a3p, a3m): # new p
    return g + pip * ((alpha * mutp - muop) * (a2p + a3p) ** 2 + (
            muoop  + h * muop + alpha * (- 2 * mutop)) * (a2p + a3p) + alpha * (muttp )- h * muoop) + pim * (
                   (alpha * mutm - muom) * (a2m - a3m) ** 2 + (
                   muoom  - h * muom + alpha * ( - 2 * mutom)) * (
                           a2m - a3m) + alpha * (muttm ) + h * muoom) + 2 * alpha * pioo * (
                   -muop * muom * (a2p + a3p) * (a2m - a3m) + muoop * muom * (a2m - a3m) + muoom * muop * (
                   a2p + a3p) - muoop * muoom)


def optimalstrategy(S, I, a1p, a1m, a2p, a2m, a3p, a3m):
    return S + a1p * I + a2p + a3p, S + a1m * I - a2m + a3m


def trajectory(alphaT,gT,hT, I,S, pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom,muttp, muttm, Deltap, Deltam, sigmap, sigmam, tldsigma, time):
    alpha_res = []
    g_res = []
    h_res = []

    a_res = []
    b_res = []

    a1p_res = []
    a2p_res = []
    a3p_res = []
    a1m_res = []
    a2m_res = []
    a3m_res = []

    alpha = alphaT
    g = gT
    h = hT
    for k in range(len(time) - 1):
        a1p = A1p(alpha, pioo, pip, pim, muop, muom, mutp, mutm)
        a1m = A1m(alpha, pioo, pip, pim, muop, muom, mutp, mutm)
        a2p = A2p(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom)
        a2m = A2m(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom)
        a3p = A3p(alpha, pioo, pip, pim, muop, muom, mutp, mutm, Deltap, Deltam, h)
        a3m = A3m(alpha, pioo, pip, pim, muop, muom, mutp, mutm, Deltap, Deltam, h)

        a1p_res.append(a1p)
        a2p_res.append(a2p)
        a3p_res.append(a3p)
        a1m_res.append(a1m)
        a2m_res.append(a2m)
        a3m_res.append(a3m)

        alpha = alpha_updt(alpha, pioo, pip, pim, muop, muom, mutp, mutm, a1p, a1m)
        h = h_updt(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, Deltap, Deltam, h, a1p,
                   a1m,
                   a2p, a2m, a3p, a3m)

        g = g_updt(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, Deltap,
                   Deltam,
                   h, g, sigmap, sigmam, tldsigma, time[k], time[k + 1], a2p, a2m, a3p, a3m)

        a, b = optimalstrategy(S, I, a1p, a1m, a2p, a2m, a3p, a3m)

        alpha_res.append(alpha)
        h_res.append(h)
        g_res.append(g)

        a_res.append(a)
        b_res.append(b)

    alpha_res.reverse()
    h_res.reverse()
    g_res.reverse()

    a_res.reverse()
    b_res.reverse()

    a1p_res.reverse()
    a2p_res.reverse()
    a3p_res.reverse()
    a1m_res.reverse()
    a2m_res.reverse()
    a3m_res.reverse()
    return alpha_res, h_res, g_res, a_res, b_res, a1p_res, a2p_res, a3p_res, a1m_res, a2m_res, a3m_res

def trajectory_new(S, alphaT,gT,hT, I, pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time): # new p
    alpha_res = []
    g_res = []
    h_res = []

    a_res = []
    b_res = []

    a1p_res = []
    a2p_res = []
    a3p_res = []
    a1m_res = []
    a2m_res = []
    a3m_res = []

    alpha = alphaT
    g = gT
    h = hT
    for k in range(len(time) - 1):
        a1p = A1p(alpha, pioo, pip, pim, muop, muom, mutp, mutm)
        a1m = A1m(alpha, pioo, pip, pim, muop, muom, mutp, mutm)
        a2p = A2p(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom)
        a2m = A2m(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom)
        # a3p = A3p(alpha, pioo, pip, pim, muop, muom, mutp, mutm, Deltap, Deltam, h)
        # a3m = A3m(alpha, pioo, pip, pim, muop, muom, mutp, mutm, Deltap, Deltam, h)
        a3p = A3p_new(alpha, pioo, pip, pim, muop, muom, mutp, mutm,h)
        a3m = A3m_new(alpha, pioo, pip, pim, muop, muom, mutp, mutm, h)

        a1p_res.append(a1p)
        a2p_res.append(a2p)
        a3p_res.append(a3p)
        a1m_res.append(a1m)
        a2m_res.append(a2m)
        a3m_res.append(a3m)

        alpha = alpha_updt(alpha, pioo, pip, pim, muop, muom, mutp, mutm, a1p, a1m)
        # h = h_updt(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, Deltap, Deltam, h, a1p,
        #            a1m,
        #            a2p, a2m, a3p, a3m)
        h = h_updt_new(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, h, a1p,
                   a1m,a2p, a2m, a3p, a3m)# new p

        # g = g_updt(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, Deltap,
        #            Deltam,
        #            h, g, sigmap, sigmam, tldsigma, time[k], time[k + 1], a2p, a2m, a3p, a3m)
        g = g_updt_new(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm,
                   h, g,  a2p, a2m, a3p, a3m) # new p

        a, b = optimalstrategy(S, I, a1p, a1m, a2p, a2m, a3p, a3m)

        alpha_res.append(alpha)
        h_res.append(h)
        g_res.append(g)

        a_res.append(a)
        b_res.append(b)

    alpha_res.reverse()
    h_res.reverse()
    g_res.reverse()

    a_res.reverse()
    b_res.reverse()

    a1p_res.reverse()
    a2p_res.reverse()
    a3p_res.reverse()
    a1m_res.reverse()
    a2m_res.reverse()
    a3m_res.reverse()
    return alpha_res, h_res, g_res, a_res, b_res, a1p_res, a2p_res, a3p_res, a1m_res, a2m_res, a3m_res

# def trajectory_new_v2(S, alphaT,gT,hT, I, pip_MA, pim_MA, pioo_MA,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, acttime_est):
def trajectory_new_v2(alphaT,gT,hT, pip_MA, pim_MA, pioo_MA,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, acttime_est): # new p # pi_t
    # pip_MA, pim_MA, pioo_MA are moving average of pi^+- and pioo (MA)
    alpha_res = []
    g_res = []
    h_res = []

    a_res = []
    b_res = []

    a1p_res = []
    a2p_res = []
    a3p_res = []
    a1m_res = []
    a2m_res = []
    a3m_res = []

    alpha = alphaT
    g = gT
    h = hT
    for k in range(1,len(acttime_est) + 1):
        a1p = A1p(alpha, pioo_MA.iloc[-k], pip_MA.iloc[-k], pim_MA.iloc[-k], muop, muom, mutp, mutm)
        a1m = A1m(alpha, pioo_MA.iloc[-k], pip_MA.iloc[-k], pim_MA.iloc[-k], muop, muom, mutp, mutm)
        a2p = A2p(alpha, pioo_MA.iloc[-k], pip_MA.iloc[-k], pim_MA.iloc[-k], muop, muom, mutp, mutm, muoop, muoom, mutop, mutom)
        a2m = A2m(alpha, pioo_MA.iloc[-k], pip_MA.iloc[-k], pim_MA.iloc[-k], muop, muom, mutp, mutm, muoop, muoom, mutop, mutom)
        # a3p = A3p(alpha, pioo, pip, pim, muop, muom, mutp, mutm, Deltap, Deltam, h)
        # a3m = A3m(alpha, pioo, pip, pim, muop, muom, mutp, mutm, Deltap, Deltam, h)
        a3p = A3p_new(alpha, pioo_MA.iloc[-k], pip_MA.iloc[-k], pim_MA.iloc[-k], muop, muom, mutp, mutm,h)
        a3m = A3m_new(alpha, pioo_MA.iloc[-k], pip_MA.iloc[-k], pim_MA.iloc[-k], muop, muom, mutp, mutm, h)

        a1p_res.append(a1p)
        a2p_res.append(a2p)
        a3p_res.append(a3p)
        a1m_res.append(a1m)
        a2m_res.append(a2m)
        a3m_res.append(a3m)

        alpha = alpha_updt(alpha, pioo_MA.iloc[-k], pip_MA.iloc[-k], pim_MA.iloc[-k], muop, muom, mutp, mutm, a1p, a1m)
        # h = h_updt(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, Deltap, Deltam, h, a1p,
        #            a1m,
        #            a2p, a2m, a3p, a3m)
        h = h_updt_new(alpha, pioo_MA.iloc[-k], pip_MA.iloc[-k], pim_MA.iloc[-k], muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, h, a1p,
                   a1m,a2p, a2m, a3p, a3m)# new p

        # g = g_updt(alpha, pioo, pip, pim, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, Deltap,
        #            Deltam,
        #            h, g, sigmap, sigmam, tldsigma, time[k], time[k + 1], a2p, a2m, a3p, a3m)
        g = g_updt_new(alpha, pioo_MA.iloc[-k], pip_MA.iloc[-k], pim_MA.iloc[-k], muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm,
                   h, g,  a2p, a2m, a3p, a3m) # new p

        # a, b = optimalstrategy(S, I, a1p, a1m, a2p, a2m, a3p, a3m)

        alpha_res.append(alpha)
        h_res.append(h)
        g_res.append(g)

        # a_res.append(a)
        # b_res.append(b)

    alpha_res.reverse()
    h_res.reverse()
    g_res.reverse()

    a_res.reverse()
    b_res.reverse()

    a1p_res.reverse()
    a2p_res.reverse()
    a3p_res.reverse()
    a1m_res.reverse()
    a2m_res.reverse()
    a3m_res.reverse()
    # return alpha_res, h_res, g_res, a_res, b_res, a1p_res, a2p_res, a3p_res, a1m_res, a2m_res, a3m_res
    return alpha_res, h_res, g_res, a1p_res, a2p_res, a3p_res, a1m_res, a2m_res, a3m_res


# ###################################################################################################
# # Pi(1,1)=0. Different I
# # new p
#
# T = 19800
# lmbda = 0.001
# alphaT = -lmbda
# gT = 0
# hT = 0
# # pip+pim-1V0<=pioo<=pip/\pim
# pip = 0.4
# pim = 0.4
# pioo = 0
#
# muop = muom = 500
# mutp = mutm = 10**6  # mut>=muo^2
# Ep = 1
# muoop = muoom = muop * Ep
# mutop = mutom = mutp * Ep
# muttp = muttm = 2*10**6
#
# # # Deltap>0, Deltam<0
# # Deltap = 5
# # Deltam = -5
#
# # sigmap = sigmam = tldsigma = 0.00013  # (second to second)
#
# time = np.linspace(0, T, T / 3 + 1)[::-1]
# S = 0
# I = [-2000,-1500,-1000, -500, 0, 500, 1000,1500,2000]
# # I = [-15000,-10000, -5000,-1000, 0,1000, 5000, 10000,15000]
#
# time_plot = list(time)
# time_plot.reverse()
# ab_res = []
# for i in range(len(I)):
#     # res = trajectory(alphaT, I[i], pip, pim, pioo, Deltap, Deltam, sigmap, sigmam, tldsigma, time)
#     # new p
#     res = trajectory_new(S, alphaT, gT, hT, I[i], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
#     ab_res.append([res[3], res[4]])
#
# # fig = plt.figure(tight_layout=True, figsize=(20, 8))
# # gs = gridspec.GridSpec(2, 2,width_ratios=[1, 2])
# #
# # c = plt.rcParams['axes.prop_cycle'].by_key()[
# #     'color']  # get default colormap in matlibplot such that each ask bid pair shares same color
# #
# # ax = fig.add_subplot(gs[:, 0])
# # for i in range(len(I)):
# #     ax.plot(time_plot[:-1], ab_res[i][0], color=c[i])
# #     ax.plot(time_plot[:-1], ab_res[i][1], color=c[i])
# # ax.set_ylabel('Bid-Ask Price')
# # ax.set_xlabel('Time')
# # xticks = ax.get_xticks()
# # # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks]) # revise x ticks
# #
# # ax = fig.add_subplot(gs[0, 1])
# # for j in range(len(I)):
# #     ax.plot(time_plot[-200:], ab_res[j][0][-200:],label = 'I = %d' % I[j])
# # labelLines(ax.get_lines(),xvals=(T-25,T-20),zorder=2.5,fontsize=8)
# # ax.set_ylabel('Ask Price')
# # ax.set_xlabel('Time')
# # xticks = ax.get_xticks()
# # # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # ax = fig.add_subplot(gs[1, 1])
# # for j in range(len(I)):
# #     ax.plot(time_plot[-200:], ab_res[j][1][-200:],label = 'I = %d' % I[j])
# # labelLines(ax.get_lines(),xvals=(T-25,T-20),zorder=2.5,fontsize=8)
# # ax.set_ylabel('Bid Price')
# # ax.set_xlabel('Time')
# # xticks = ax.get_xticks()
# # # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# # fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
# # plt.show()
#
# fig = plt.figure(tight_layout=True, figsize=(15,6))
# ax1 = fig.add_axes([0.05, 0.5, .95, 0.4],
#                    xticklabels=[])
# ax2 = fig.add_axes([0.05, 0.1, .95, 0.4])
#
# for j in range(len(I)):
#     ax1.plot(time_plot[-100:], ab_res[j][0][-100:],label = 'I = %d' % I[j])
# labelLines(ax1.get_lines(),xvals=(T-25,T-20),zorder=2.5,fontsize=8)
# ax1.set_ylabel('Ask Spread')
#
# for j in range(len(I)):
#     ax2.plot(time_plot[-100:], ab_res[j][1][-100:],label = 'I = %d' % I[j])
# labelLines(ax2.get_lines(),xvals=(T-25,T-20),zorder=2.5,fontsize=8)
# ax2.set_ylabel('Bid Spread')
# ax2.set_xlabel('Time')
# xticks = ax2.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
# ax2.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# # fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
# plt.show()
#
# fig, ax = plt.subplots(tight_layout=True, figsize=(15,3))
# for j in range(len(I)):
#     ax.plot(time_plot[1:], [x-y for x,y in zip(ab_res[j][0],ab_res[j][1])])
# ax.set_ylabel('Bid-Ask Spread')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# # fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
# plt.show()
#

# ######################################################################################
# # Pi(1,1)=0. Different penalty
#
# T = 23400
# lmbda = [0,0.01,1,10, 100]
# alphaT = [-x for x in lmbda]
# gT = 0
# hT = 0
# # pip+pim-1V0<=pioo<=pip/\pim
# pip = 0.4
# pim = 0.4
# pioo = 0
#
# muop = muom = 500
# mutp = mutm = 10**6  # mut>=muo^2
# Ep = 1
# muoop = muoom = muop * Ep
# mutop = mutom = mutp * Ep
# muttp = muttm = 4*10**6
#
# # # Deltap>0, Deltam<0
# # Deltap = 5
# # Deltam = -5
# #
# # sigmap = sigmam = tldsigma = 0.00013  # (second to second)
#
# time = np.linspace(0, T, T / 3 + 1)[::-1]
# S = 0
# I =550
#
# time_plot = list(time)
# time_plot.reverse()
# ab_res = []
# for i in range(len(alphaT)):
#     res = trajectory(alphaT[i], I, pip, pim, pioo, Deltap, Deltam, sigmap, sigmam, tldsigma, time)
#     ab_res.append([res[3], res[4]])
#
# fig = plt.figure(tight_layout=True, figsize=(20, 8))
# gs = gridspec.GridSpec(2, 2,width_ratios=[1, 2])
#
# c = plt.rcParams['axes.prop_cycle'].by_key()[
#     'color']  # get default colormap in matlibplot such that each ask bid pair shares same color
#
# ax = fig.add_subplot(gs[:, 0])
# for i in range(len(alphaT)):
#     ax.plot(time_plot[:-1], ab_res[i][0], color=c[i])
#     ax.plot(time_plot[:-1], ab_res[i][1], color=c[i])
# ax.set_ylabel('Bid-Ask Price')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# ax = fig.add_subplot(gs[0, 1])
# for j in range(len(alphaT)):
#     ax.plot(time_plot[-100:], ab_res[j][0][-100:])
#     # ax.annotate('$\lambda$ = %.2f' % (-alphaT[j]), (time_plot[-8]+3, ab_res[j][0][-8]),
#     #             xytext=(time_plot[-8] - 70, ab_res[j][0][-8]+0.1), arrowprops=dict(arrowstyle='->'))
# ax.annotate('$\lambda$ = 0, 0.01, 1, 10, 100', (time_plot[-8]+3, ab_res[j][0][-8]),
#             xytext=(time_plot[-8] - 70, ab_res[j][0][-8]+0.1), arrowprops=dict(arrowstyle='->'))
#
# ax.set_ylabel('Ask Price')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# ax = fig.add_subplot(gs[1, 1])
# for j in range(len(alphaT)):
#     ax.plot(time_plot[-100:], ab_res[j][1][-100:])
#     ax.annotate('$\lambda$ = %.2f' % (-alphaT[j]), (time_plot[-8]+3, ab_res[j][1][-8]),
#                 xytext=(time_plot[-8] - 70, ab_res[j][1][-8]-2), arrowprops=dict(arrowstyle='->'))
# ax.set_ylabel('Bid Price')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
#
# plt.show()

# #####################################################################################
# # Pi(1,1)=0. Different penalty. New Plot
#
# T = 19800
# lmbda = [0,0.001,0.01]
# alphaT = [-x for x in lmbda]
# I= [500,-500,1000,-1000,1500,-1500]
# gT = 0
# hT = 0
# # pip+pim-1V0<=pioo<=pip/\pim
# pip = 0.4
# pim = 0.4
# pioo = 0
#
# muop = muom = 500
# mutp = mutm = 10**6  # mut>=muo^2
# Ep = 1
# muoop = muoom = muop * Ep
# mutop = mutom = mutp * Ep
# muttp = muttm = 2*10**6
#
# # # Deltap>0, Deltam<0
# # Deltap = 5
# # Deltam = -5
# #
# # sigmap = sigmam = tldsigma = 0.00013  # (second to second)
#
# time = np.linspace(0, T, T / 3 + 1)[::-1]
# S = 0
#
# time_plot = list(time)
# time_plot.reverse()
# #
# # fig = plt.figure(tight_layout=True,figsize=(20, 8))
# # gs = gridspec.GridSpec(3, 2)
# #
# # c = plt.rcParams['axes.prop_cycle'].by_key()[
# #     'color']  # get default colormap in matlibplot such that each ask bid pair shares same color
# # # I = -1500
# # ax = fig.add_subplot(gs[0, 0])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[0], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(),xvals=(T-50,T-50),zorder=2)
# # ax.set_ylabel('Ask Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Ask, Inventory = %d' % I[0])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # # for i in reversed(range(len(alphaT))):
# # #     temp = trajectory_new(S, alphaT[i],gT,hT, I[0], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# # #     plt.plot(time_plot[:-1][-100:], [x-y for x,y in zip(temp[3][-100:],temp[4][-100:])], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # # plt.show()
# #
# # ax = fig.add_subplot(gs[0, 1])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[0], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(),xvals=(T-50,T-50),zorder=2)
# # ax.set_ylabel('Bid Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Bid, Inventory = %d' % I[0])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # # I = -1000
# # ax = fig.add_subplot(gs[1, 0])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[1], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i],label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(), xvals=(T-50,T-50), zorder=2)
# # ax.set_ylabel('Ask Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Ask, Inventory = %d' % I[1])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # ax = fig.add_subplot(gs[1, 1])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[1], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i],label = '$\lambda$ = 0, 0.001, 0.01')
# # labelLines(ax.get_lines(), xvals=(T-100,T-100), zorder=2)
# # ax.set_ylabel('Bid Price')
# # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Bid, Inventory = %d' % I[1])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # # for i in reversed(range(len(alphaT))):
# # #     temp = trajectory_new(S, alphaT[i],gT,hT, I[1], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# # #     plt.plot(time_plot[:-1][-100:], [x-y for x,y in zip(temp[3][-100:],temp[4][-100:])], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # # plt.show()
# #
# # # ask I = -500
# # ax = fig.add_subplot(gs[2, 0])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[2], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(),xvals=(T-50,T-50),zorder=2)
# # ax.set_ylabel('Ask Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Ask, Inventory = %d' % I[2])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # ax = fig.add_subplot(gs[2, 1])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[2], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(),xvals=(T-50,T-50),zorder=2)
# # ax.set_ylabel('Bid Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Bid, Inventory = %d' % I[2])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
# #
# # plt.show()
# #
# #
# # fig = plt.figure(tight_layout=True,figsize=(20,8))
# # gs = gridspec.GridSpec(3, 2)
# #
# # c = plt.rcParams['axes.prop_cycle'].by_key()[
# #     'color']  # get default colormap in matlibplot such that each ask bid pair shares same color
# #
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[2], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     plt.plot(time_plot[:-1][-100:], [x-y for x,y in zip(temp[3][-100:],temp[4][-100:])], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # plt.show()
# #
# # # I = 500
# # ax = fig.add_subplot(gs[0, 0])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[3], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(),xvals=(T-50,T-50),zorder=2)
# # ax.set_ylabel('Ask Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Ask, Inventory = %d' % I[3])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # ax = fig.add_subplot(gs[0, 1])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[3], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(),xvals=(T-50,T-50),zorder=2)
# # ax.set_ylabel('Bid Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Bid, Inventory = %d' % I[3])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # # I = 1000
# # ax = fig.add_subplot(gs[1,0])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[4], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i],label = '$\lambda$ = 0, 0.001, 0.01')
# # labelLines(ax.get_lines(), xvals=(T-50,T-50), zorder=2)
# # ax.set_ylabel('Ask Price')
# # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Ask, Inventory = %d' % I[4])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # ax = fig.add_subplot(gs[1,1])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[4], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i],label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(), xvals=(T-50,T-50), zorder=2)
# # ax.set_ylabel('Bid Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Bid, Inventory = %d' % I[4])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # # ask I = 1500
# # ax = fig.add_subplot(gs[2,0])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[5], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(),xvals=(T-50,T-50),zorder=2)
# # ax.set_ylabel('Ask Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Ask, Inventory = %d' % I[5])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # ax = fig.add_subplot(gs[2,1])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[5], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(),xvals=(T-50,T-50),zorder=2)
# # ax.set_ylabel('Bid Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Bid, Inventory = %d' % I[5])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# #
# #
# # fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
# #
# # plt.show()
# #
#
#
# fig = plt.figure(figsize=(20,12))
# outer = gridspec.GridSpec(3, 2, wspace=0.2, hspace=0.3)
#
# # I = 500
# inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[0])
#
# ax1 = plt.Subplot(fig, inner[0])
# for i in range(len(alphaT)):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[0], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax1.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax1.get_lines(), xvals=(T - 20, T - 20), zorder=2,fontsize = 5)
# # ax1.legend()
# ax1.set_ylabel('Ask Spread')
# ax1.get_xaxis().set_visible(False)
# ax1.set_title('Optimal Ask and Bid, Inventory = %d' % I[0])
# ax1.set_ylim(0,1.9)
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# fig.add_subplot(ax1)
#
# ax2 = plt.Subplot(fig, inner[1],sharex = ax1)
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[0], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax2.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax2.get_lines(), xvals=(T - 50, T - 50), zorder=2)
# ax2.legend()
# ax2.set_ylabel('Bid Spread')
# ax2.set_xlabel('Time')
# # ax2.set_title('Optimal Ask and Bid, Inventory = %d' % I[j])
# xticks = ax2.get_xticks()
# ax2.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# ax2.set_ylim(-1.9, -0)
# fig.add_subplot(ax2)
# plt.subplots_adjust(hspace=.0)
#
# # I = -500
# inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[1])
#
# ax1 = plt.Subplot(fig, inner[0])
# for i in range(len(alphaT)):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[1], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax1.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax1.get_lines(), xvals=(T - 20, T - 20), zorder=2,fontsize = 5)
# # ax1.legend()
# ax1.set_ylabel('Ask Spread')
# ax1.get_xaxis().set_visible(False)
# ax1.set_title('Optimal Ask and Bid, Inventory = %d' % I[1])
# ax1.set_ylim(0,1.9)
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# fig.add_subplot(ax1)
#
# ax2 = plt.Subplot(fig, inner[1],sharex = ax1)
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[1], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax2.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax2.get_lines(), xvals=(T - 50, T - 50), zorder=2)
# ax2.legend()
# ax2.set_ylabel('Bid Spread')
# ax2.set_xlabel('Time')
# # ax2.set_title('Optimal Ask and Bid, Inventory = %d' % I[j])
# xticks = ax2.get_xticks()
# ax2.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# ax2.set_ylim(-1.9, -0)
# fig.add_subplot(ax2)
# plt.subplots_adjust(hspace=.0)
#
# # I = 1000
# inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[2])
#
# ax1 = plt.Subplot(fig, inner[0])
# for i in range(len(alphaT)):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[2], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax1.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[3], label = '$\lambda$ = 0, 0.001, 0.01')
# labelLines(ax1.get_lines(), xvals=(T - 70, T - 70), zorder=2)
# # ax1.legend()
# ax1.set_ylabel('Ask Spread')
# ax1.get_xaxis().set_visible(False)
# ax1.set_title('Optimal Ask and Bid, Inventory = %d' % I[2])
# ax1.set_ylim(0,1.9)
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# fig.add_subplot(ax1)
#
# ax2 = plt.Subplot(fig, inner[1],sharex = ax1)
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[2], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax2.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax2.get_lines(), xvals=(T - 50, T - 50), zorder=2)
# ax2.legend()
# ax2.set_ylabel('Bid Spread')
# ax2.set_xlabel('Time')
# # ax2.set_title('Optimal Ask and Bid, Inventory = %d' % I[j])
# xticks = ax2.get_xticks()
# ax2.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# ax2.set_ylim(-1.9, -0)
# fig.add_subplot(ax2)
# plt.subplots_adjust(hspace=.0)
#
# # I = -1000
# inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[3])
#
# ax1 = plt.Subplot(fig, inner[0])
# for i in range(len(alphaT)):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[3], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax1.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax1.get_lines(), xvals=(T - 70, T - 70), zorder=2)
# ax1.legend()
# ax1.set_ylabel('Ask Spread')
# ax1.get_xaxis().set_visible(False)
# ax1.set_title('Optimal Ask and Bid, Inventory = %d' % I[3])
# ax1.set_ylim(0,1.9)
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# fig.add_subplot(ax1)
#
# ax2 = plt.Subplot(fig, inner[1],sharex = ax1)
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[3], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax2.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[3], label='$\lambda$ = 0, 0.001, 0.01')
# labelLines(ax2.get_lines(), xvals=(T - 70, T - 70), zorder=2)
# # ax2.legend()
# ax2.set_ylabel('Bid Spread')
# ax2.set_xlabel('Time')
# # ax2.set_title('Optimal Ask and Bid, Inventory = %d' % I[j])
# xticks = ax2.get_xticks()
# ax2.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# ax2.set_ylim(-1.9, -0)
# fig.add_subplot(ax2)
# plt.subplots_adjust(hspace=.0)
#
# # I = 1500
# inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[4])
#
# ax1 = plt.Subplot(fig, inner[0])
# for i in range(len(alphaT)):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[4], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax1.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax1.get_lines(), xvals=(T - 20, T - 20), zorder=2,fontsize = 5)
# # ax1.legend()
# ax1.set_ylabel('Ask Spread')
# ax1.get_xaxis().set_visible(False)
# ax1.set_title('Optimal Ask and Bid, Inventory = %d' % I[4])
# ax1.set_ylim(0,1.9)
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# fig.add_subplot(ax1)
#
# ax2 = plt.Subplot(fig, inner[1],sharex = ax1)
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[4], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax2.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax2.get_lines(), xvals=(T - 50, T - 50), zorder=2)
# ax2.legend()
# ax2.set_ylabel('Bid Spread')
# ax2.set_xlabel('Time')
# # ax2.set_title('Optimal Ask and Bid, Inventory = %d' % I[j])
# xticks = ax2.get_xticks()
# ax2.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# ax2.set_ylim(-1.9, 0)
# fig.add_subplot(ax2)
# plt.subplots_adjust(hspace=.0)
#
#
# # I = -1500
# inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[5])
#
# ax1 = plt.Subplot(fig, inner[0])
# for i in range(len(alphaT)):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[5], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax1.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax1.get_lines(), xvals=(T - 20, T - 20), zorder=2,fontsize = 5)
# # ax1.legend()
# ax1.set_ylabel('Ask Spread')
# ax1.get_xaxis().set_visible(False)
# ax1.set_title('Optimal Ask and Bid, Inventory = %d' % I[5])
# ax1.set_ylim(0,1.9)
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# fig.add_subplot(ax1)
#
# ax2 = plt.Subplot(fig, inner[1],sharex = ax1)
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[5], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax2.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax2.get_lines(), xvals=(T - 50, T - 50), zorder=2)
# ax2.legend()
# ax2.set_ylabel('Bid Spread')
# ax2.set_xlabel('Time')
# # ax2.set_title('Optimal Ask and Bid, Inventory = %d' % I[j])
# xticks = ax2.get_xticks()
# ax2.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# ax2.set_ylim(-1.9, 0)
# fig.add_subplot(ax2)
# plt.subplots_adjust(hspace=.0)
#
# # fig.align_labels()
# plt.savefig('fooa.png', bbox_inches='tight')
#
#
# fig, ax = plt.subplots(tight_layout=True, figsize=(15,3))
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i],gT,hT, I[1], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
#     ax.plot(time_plot[:-1][-100:], [x-y for x,y in zip(temp[3][-100:],temp[4][-100:])], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# ax.set_ylabel('Bid-Ask Spread')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# labelLines(ax.get_lines(), xvals=(T - 50, T - 50), zorder=2)
# plt.savefig('foob.png', bbox_inches='tight')
# ####################################################################################################
# # Pi(1,1)=0. Asymetric (pi+,pi-), different I
#
# T = 23400
# lmbda = 10
# alphaT = -lmbda
# gT = 0
# hT = 0
# # pip+pim-1V0<=pioo<=pip/\pim
# pip = 0.9
# pim = 0.3
# pioo = 0
#
# muop = muom = 100
# mutp = mutm = 11000  # mut>=muo^2
# Ep = 10
# muoop = muoom = muop * Ep
# mutop = mutom = mutp * Ep
# muttp = muttm = 11000*110
#
# # Deltap>0, Deltam<0
# Deltap = 5
# Deltam = -5
#
# sigmap = sigmam = tldsigma = 0.00013  # (second to second)
#
# time = np.linspace(0, T, T / 3 + 1)[::-1]
# S = 0
# I = [-1000,-500,0,500,1000]
#
# time_plot = list(time)
# time_plot.reverse()
# ab_res = []
# for i in range(len(I)):
#     res = trajectory(alphaT, I[i], pip, pim, pioo, Deltap, Deltam, sigmap, sigmam, tldsigma, time)
#     ab_res.append([res[3], res[4]])
#
# fig = plt.figure(tight_layout=True,figsize=(20,8))
# gs = gridspec.GridSpec(2, 2)
#
# c = plt.rcParams['axes.prop_cycle'].by_key()['color'] # get default colormap in matlibplot such that each ask bid pair shares same color
#
# ax = fig.add_subplot(gs[:, 0])
# for i in range(len(I)):
#     ax.plot(time_plot[:-1], ab_res[i][0], color=c[i])
#     ax.plot(time_plot[:-1], ab_res[i][1], color=c[i])
# ax.set_ylabel('Bid-Ask Price')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x+9.5*3600)) for x in xticks])
#
# ax = fig.add_subplot(gs[0, 1])
# for j in range(len(I)):
#     ax.plot(time_plot[-100:], ab_res[j][0][-100:])
#     ax.annotate('I = %d' % I[j], (time_plot[-2] , ab_res[j][0][-2] ),
#                     xytext=(time_plot[-2] - 50, ab_res[j][0][-2] + 0.2), arrowprops=dict(arrowstyle='->'), verticalalignment='top')
# ax.set_ylabel('Ask Price')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x+9.5*3600)) for x in xticks])
#
# ax = fig.add_subplot(gs[1, 1])
# for j in range(len(I)):
#     ax.plot(time_plot[-100:], ab_res[j][1][-100:])
#     ax.annotate('I = %d' % I[j], (time_plot[-2] , ab_res[j][1][-2] ),
#                     xytext=(time_plot[-2] - 50, ab_res[j][1][-2] + 0.2), arrowprops=dict(arrowstyle='->'), verticalalignment='top')
# ax.set_ylabel('Bid Price')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x+9.5*3600)) for x in xticks])
#
# fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
#
# plt.show()

# ###################################################################################################
# # Pi(1,1)>0. symetric (pi+,pi-), different I
# # new p
#
# T = 19800
# lmbda = 0.001
# alphaT = -lmbda
# gT = 0
# hT = 0
# # pip+pim-1V0<=pioo<=pip/\pim
# pip = 0.4
# pim = 0.4
# pioo = 0.2
#
# muop = muom = 500
# mutp = mutm = 10**6  # mut>=muo^2
# Ep = 1
# muoop = muoom = muop * Ep
# mutop = mutom = mutp * Ep
# muttp = muttm = 2*10**6
#
# # # Deltap>0, Deltam<0
# # Deltap = 5
# # Deltam = -5
#
# # sigmap = sigmam = tldsigma = 0.00013  # (second to second)
#
# time = np.linspace(0, T, T / 3 + 1)[::-1]
# S = 0
# I = [-2000,-1500,-1000, -500, 0, 500, 1000,1500,2000]
# # I = np.linspace(1000,1500,num = 6)
# time_plot = list(time)
# time_plot.reverse()
# ab_res = []
# for i in range(len(I)):
#     # res = trajectory(alphaT, gT, hT, I[i],S, pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp,
#     #                  muttm, 0.9, -0.9, 0.53, 0.53, 1.07, time)
#     # new p
#     res = trajectory_new(S, alphaT, gT, hT, I[i], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
#     ab_res.append([res[3], res[4]])
# #
# # fig = plt.figure(tight_layout=True, figsize=(20, 8))
# # gs = gridspec.GridSpec(2, 2,width_ratios=[1, 2])
# #
# # c = plt.rcParams['axes.prop_cycle'].by_key()[
# #     'color']  # get default colormap in matlibplot such that each ask bid pair shares same color
# #
# # ax = fig.add_subplot(gs[:, 0])
# # for i in range(len(I)):
# #     ax.plot(time_plot[:-1], ab_res[i][0], color=c[i])
# #     ax.plot(time_plot[:-1], ab_res[i][1], color=c[i])
# # ax.set_ylabel('Bid-Ask Price')
# # ax.set_xlabel('Time')
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # ax = fig.add_subplot(gs[0, 1])
# # for j in range(len(I)):
# #     ax.plot(time_plot[-100:], ab_res[j][0][-100:],label = 'I = %d' % I[j])
# # labelLines(ax.get_lines(),xvals=(T-25,T-50),zorder=2.5,fontsize = 8)
# # ax.set_ylabel('Ask Price')
# # ax.set_xlabel('Time')
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # ax = fig.add_subplot(gs[1, 1])
# # for j in range(len(I)):
# #     ax.plot(time_plot[-100:], ab_res[j][1][-100:],label = 'I = %d' % I[j])
# # labelLines(ax.get_lines(),xvals=(T-50,T-25),zorder=2.5,fontsize = 8)
# # ax.set_ylabel('Bid Price')
# # ax.set_xlabel('Time')
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
# # plt.show()
# #
# # for j in range(len(I)):
# #     plt.plot(time_plot[-100:], [x-y for x,y in zip(ab_res[j][0][-100:],ab_res[j][1][-100:])],label = 'I = %d' % I[j])
# # plt.show()
#
#
# fig = plt.figure(tight_layout=True, figsize=(15,6))
# ax1 = fig.add_axes([0.05, 0.5, .95, 0.4],
#                    xticklabels=[])
# ax2 = fig.add_axes([0.05, 0.1, .95, 0.4])
#
# for j in range(len(I)):
#     ax1.plot(time_plot[-100:], ab_res[j][0][-100:],label = 'I = %d' % I[j])
# labelLines(ax1.get_lines(),xvals=(T-25,T-20),zorder=2.5,fontsize=8)
# ax1.set_ylabel('Ask Spread')
#
# for j in range(len(I)):
#     ax2.plot(time_plot[-100:], ab_res[j][1][-100:],label = 'I = %d' % I[j])
# labelLines(ax2.get_lines(),xvals=(T-25,T-20),zorder=2.5,fontsize=8)
# ax2.set_ylabel('Bid Spread')
# ax2.set_xlabel('Time')
# xticks = ax2.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
# ax2.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# # fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
# plt.savefig('fooa.png', bbox_inches='tight')
#
# fig, ax = plt.subplots(tight_layout=True, figsize=(15,3))
# for j in range(len(I)):
#     ax.plot(time_plot[1:], [x-y for x,y in zip(ab_res[j][0],ab_res[j][1])])
# ax.set_ylabel('Bid-Ask Spread')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# # fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
# plt.savefig('foob.png', bbox_inches='tight')
#

#####################################################################################
# Pi(1,1)=0.25. Different penalty
#
# T = 23400
# lmbda = [0,0.1,1,10,100]
# alphaT = [-x for x in lmbda]
# I= [-1500,-1000,-500,500,1000,1500]
# gT = 0
# hT = 0
# # pip+pim-1V0<=pioo<=pip/\pim
# pip = 0.4
# pim = 0.4
# pioo = 0.25
#
# muop = muom = 500
# mutp = mutm = 10**6  # mut>=muo^2
# Ep = 1
# muoop = muoom = muop * Ep
# mutop = mutom = mutp * Ep
# muttp = muttm = 4*10**6
#
# time = np.linspace(0, T, T / 3 + 1)[::-1]
# S = 0
# I =1200
#
# time_plot = list(time)
# time_plot.reverse()
# ab_res = []
# for i in range(len(alphaT)):
#     res = trajectory_new(S, alphaT[i], gT, hT, I, pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
#     ab_res.append([res[3], res[4]])
#
# fig = plt.figure(tight_layout=True, figsize=(20, 8))
# gs = gridspec.GridSpec(2, 2,width_ratios=[1, 2])
#
# c = plt.rcParams['axes.prop_cycle'].by_key()[
#     'color']  # get default colormap in matlibplot such that each ask bid pair shares same color
#
# ax = fig.add_subplot(gs[:, 0])
# for i in range(len(alphaT)):
#     ax.plot(time_plot[:-1], ab_res[i][0], color=c[i])
#     ax.plot(time_plot[:-1], ab_res[i][1], color=c[i])
# ax.set_ylabel('Bid-Ask Price')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# ax = fig.add_subplot(gs[0, 1])
# for j in reversed(range(len(alphaT))):
#     ax.plot(time_plot[-100:], ab_res[j][0][-100:],label = '$\lambda$ = %.1f' % -alphaT[j])
# labelLines(ax.get_lines(),xvals=(23150,23350),zorder=2.5)
# ax.set_ylabel('Ask Price')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# ax = fig.add_subplot(gs[1, 1])
# for j in range(len(alphaT)):
#     ax.plot(time_plot[-100:], ab_res[j][1][-100:],label = '$\lambda$ = %.1f' % -alphaT[j])
# labelLines(ax.get_lines(),xvals=(23250,23250),zorder=2.5)
# ax.set_ylabel('Bid Price')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
#
# plt.show()

# #################################################################################################
# # Pi(1,1)=0. Different penalty. New Plot
#
# T = 19800
# lmbda = [0,0.001,0.01]
# alphaT = [-x for x in lmbda]
# # I= [-1500,-1050,-500,500,1050,1500]
# I= [500,-500,1050,-1050,1500,-1500]
# gT = 0
# hT = 0
# # pip+pim-1V0<=pioo<=pip/\pim
# pip = 0.4
# pim = 0.4
# pioo = 0.2
#
# muop = muom = 500
# mutp = mutm = 10**6  # mut>=muo^2
# Ep = 1
# muoop = muoom = muop * Ep
# mutop = mutom = mutp * Ep
# muttp = muttm = 2*10**6
#
# # # Deltap>0, Deltam<0
# # Deltap = 5
# # Deltam = -5
# #
# # sigmap = sigmam = tldsigma = 0.00013  # (second to second)
#
# time = np.linspace(0, T, T / 3 + 1)[::-1]
# S = 0
#
# time_plot = list(time)
# time_plot.reverse()
# #
# # fig = plt.figure(tight_layout=True,figsize=(20, 8))
# # gs = gridspec.GridSpec(3, 2)
# #
# c = plt.rcParams['axes.prop_cycle'].by_key()[
#     'color']  # get default colormap in matlibplot such that each ask bid pair shares same color
# # # I = -1500
# # ax = fig.add_subplot(gs[0, 0])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[0], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(),xvals=(T-50,T-50),zorder=2)
# # ax.set_ylabel('Ask Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Ask, Inventory = %d' % I[0])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # ax = fig.add_subplot(gs[0, 1])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[0], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(),xvals=(T-50,T-50),zorder=2)
# # ax.set_ylabel('Bid Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Bid, Inventory = %d' % I[0])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # # for i in reversed(range(len(alphaT))):
# # #     temp = trajectory_new(S, alphaT[i],gT,hT, I[0], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# # #     plt.plot(time_plot[:-1][-100:], [x-y for x,y in zip(temp[3][-100:],temp[4][-100:])], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # # plt.show()
# #
# # # I = -1000
# # ax = fig.add_subplot(gs[1, 0])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[1], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i],label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(), xvals=(T-50,T-50), zorder=2)
# # ax.set_ylabel('Ask Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Ask, Inventory = %d' % I[1])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # ax = fig.add_subplot(gs[1, 1])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[1], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i],label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(), xvals=(T-25,T-25), zorder=1,fontsize = 8, align=False,ha='left',va='bottom')
# # ax.set_ylabel('Bid Price')
# # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Bid, Inventory = %d' % I[1])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[1], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     plt.plot(time_plot[:-1][-100:], [x-y for x,y in zip(temp[3][-100:],temp[4][-100:])], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # plt.show()
# #
# # # ask I = -500
# # ax = fig.add_subplot(gs[2, 0])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[2], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(),xvals=(T-50,T-50),zorder=2)
# # ax.set_ylabel('Ask Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Ask, Inventory = %d' % I[2])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # ax = fig.add_subplot(gs[2, 1])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[2], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(),xvals=(T-50,T-50),zorder=2)
# # ax.set_ylabel('Bid Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Bid, Inventory = %d' % I[2])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
# #
# # plt.show()
# #
# #
# # fig = plt.figure(tight_layout=True,figsize=(20,8))
# # gs = gridspec.GridSpec(3, 2)
# #
# # c = plt.rcParams['axes.prop_cycle'].by_key()[
# #     'color']  # get default colormap in matlibplot such that each ask bid pair shares same color
# #
# # # I = 500
# # ax = fig.add_subplot(gs[0, 0])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[3], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(),xvals=(T-50,T-50),zorder=2)
# # ax.set_ylabel('Ask Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Ask, Inventory = %d' % I[3])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # ax = fig.add_subplot(gs[0, 1])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[3], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(),xvals=(T-50,T-50),zorder=2)
# # ax.set_ylabel('Bid Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Bid, Inventory = %d' % I[3])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # # I = 1000
# # ax = fig.add_subplot(gs[1,0])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[4], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i],label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(), xvals=(T-10,T-10), zorder=1,fontsize = 8, align=False,ha='right',va='bottom')
# # ax.set_ylabel('Ask Price')
# # ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Ask, Inventory = %d' % I[4])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # ax = fig.add_subplot(gs[1,1])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[4], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i],label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(), xvals=(T-50,T-50), zorder=2)
# # ax.set_ylabel('Bid Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Bid, Inventory = %d' % I[4])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # # ask I = 1500
# # ax = fig.add_subplot(gs[2,0])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[5], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(),xvals=(T-50,T-50),zorder=2)
# # ax.set_ylabel('Ask Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Ask, Inventory = %d' % I[5])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # ax = fig.add_subplot(gs[2,1])
# # for i in reversed(range(len(alphaT))):
# #     temp = trajectory_new(S, alphaT[i],gT,hT, I[5], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
# #     ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax.get_lines(),xvals=(T-50,T-50),zorder=2)
# # ax.set_ylabel('Bid Price')
# # ax.set_xlabel('Time')
# # ax.set_title('Optimal Bid, Inventory = %d' % I[5])
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# #
# # fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
# #
# # plt.show()
#
#
#
# fig = plt.figure(figsize=(20,12))
# outer = gridspec.GridSpec(3, 2, wspace=0.2, hspace=0.3)
#
# # I = 500
# inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[0])
#
# ax1 = plt.Subplot(fig, inner[0])
# for i in range(len(alphaT)):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[0], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax1.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax1.get_lines(), xvals=(T - 20, T - 20), zorder=2,fontsize = 5)
# # ax1.legend()
# ax1.set_ylabel('Ask Spread')
# ax1.get_xaxis().set_visible(False)
# ax1.set_title('Optimal Ask and Bid, Inventory = %d' % I[0])
# ax1.set_ylim(0,1.9)
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# fig.add_subplot(ax1)
#
# ax2 = plt.Subplot(fig, inner[1],sharex = ax1)
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[0], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax2.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax2.get_lines(), xvals=(T - 50, T - 50), zorder=2)
# ax2.legend()
# ax2.set_ylabel('Bid Spread')
# ax2.set_xlabel('Time')
# # ax2.set_title('Optimal Ask and Bid, Inventory = %d' % I[j])
# xticks = ax2.get_xticks()
# ax2.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# ax2.set_ylim(-1.9, -0)
# fig.add_subplot(ax2)
# plt.subplots_adjust(hspace=.0)
#
# # I = -500
# inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[1])
#
# ax1 = plt.Subplot(fig, inner[0])
# for i in range(len(alphaT)):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[1], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax1.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax1.get_lines(), xvals=(T - 20, T - 20), zorder=2,fontsize = 5)
# # ax1.legend()
# ax1.set_ylabel('Ask Spread')
# ax1.get_xaxis().set_visible(False)
# ax1.set_title('Optimal Ask and Bid, Inventory = %d' % I[1])
# ax1.set_ylim(0,1.9)
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# fig.add_subplot(ax1)
#
# ax2 = plt.Subplot(fig, inner[1],sharex = ax1)
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[1], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax2.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax2.get_lines(), xvals=(T - 50, T - 50), zorder=2)
# ax2.legend()
# ax2.set_ylabel('Bid Spread')
# ax2.set_xlabel('Time')
# # ax2.set_title('Optimal Ask and Bid, Inventory = %d' % I[j])
# xticks = ax2.get_xticks()
# ax2.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# ax2.set_ylim(-1.9, -0)
# fig.add_subplot(ax2)
# plt.subplots_adjust(hspace=.0)
#
# # I = 1050
# inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[2])
#
# ax1 = plt.Subplot(fig, inner[0])
# for i in range(len(alphaT)):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[2], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     X = time_plot[:-1][-100:]
#     Y = temp[3][-100:]
#     ax1.plot(X,Y, color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
#
# # labelLines(ax1.get_lines(), xvals=(T - 70, T - 70), zorder=2)
# # ax1.legend()
# ax1.set_ylabel('Ask Spread')
# ax1.get_xaxis().set_visible(False)
# ax1.set_title('Optimal Ask and Bid, Inventory = %d' % I[2])
# ax1.set_ylim(0,1.9)
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# fig.add_subplot(ax1)
#
#
# ax2 = plt.Subplot(fig, inner[1],sharex = ax1)
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[2], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax2.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax2.get_lines(), xvals=(T - 50, T - 50), zorder=2)
# ax2.legend()
# ax2.set_ylabel('Bid Spread')
# ax2.set_xlabel('Time')
# # ax2.set_title('Optimal Ask and Bid, Inventory = %d' % I[j])
# xticks = ax2.get_xticks()
# ax2.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# ax2.set_ylim(-1.9, -0)
# fig.add_subplot(ax2)
# plt.subplots_adjust(hspace=.0)
#
# # I = -1050
# inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[3])
#
# ax1 = plt.Subplot(fig, inner[0])
# for i in range(len(alphaT)):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[3], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax1.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax1.get_lines(), xvals=(T - 70, T - 70), zorder=2)
# # ax1.legend()
# ax1.set_ylabel('Ask Spread')
# ax1.get_xaxis().set_visible(False)
# ax1.set_title('Optimal Ask and Bid, Inventory = %d' % I[3])
# ax1.set_ylim(0,1.9)
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# fig.add_subplot(ax1)
#
# ax2 = plt.Subplot(fig, inner[1],sharex = ax1)
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[3], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax2.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax2.get_lines(), xvals=(T - 70, T - 70), zorder=2)
# ax2.legend()
# ax2.set_ylabel('Bid Spread')
# ax2.set_xlabel('Time')
# # ax2.set_title('Optimal Ask and Bid, Inventory = %d' % I[j])
# xticks = ax2.get_xticks()
# ax2.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# ax2.set_ylim(-1.9, -0)
# fig.add_subplot(ax2)
# plt.subplots_adjust(hspace=.0)
#
# # I = 1500
# inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[4])
#
# ax1 = plt.Subplot(fig, inner[0])
# for i in range(len(alphaT)):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[4], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax1.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax1.get_lines(), xvals=(T - 20, T - 20), zorder=2,fontsize = 5)
# # ax1.legend()
# ax1.set_ylabel('Ask Spread')
# ax1.get_xaxis().set_visible(False)
# ax1.set_title('Optimal Ask and Bid, Inventory = %d' % I[4])
# ax1.set_ylim(0,1.9)
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# fig.add_subplot(ax1)
#
# ax2 = plt.Subplot(fig, inner[1],sharex = ax1)
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[4], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax2.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax2.get_lines(), xvals=(T - 50, T - 50), zorder=2)
# ax2.legend()
# ax2.set_ylabel('Bid Spread')
# ax2.set_xlabel('Time')
# # ax2.set_title('Optimal Ask and Bid, Inventory = %d' % I[j])
# xticks = ax2.get_xticks()
# ax2.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# ax2.set_ylim(-1.9, 0)
# fig.add_subplot(ax2)
# plt.subplots_adjust(hspace=.0)
#
#
# # I = -1500
# inner = gridspec.GridSpecFromSubplotSpec(2, 1,subplot_spec=outer[5])
#
# ax1 = plt.Subplot(fig, inner[0])
# for i in range(len(alphaT)):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[5], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax1.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax1.get_lines(), xvals=(T - 20, T - 20), zorder=2,fontsize = 5)
# # ax1.legend()
# ax1.set_ylabel('Ask Spread')
# ax1.get_xaxis().set_visible(False)
# ax1.set_title('Optimal Ask and Bid, Inventory = %d' % I[5])
# ax1.set_ylim(0,1.9)
# # xticks = ax.get_xticks()
# # ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# fig.add_subplot(ax1)
#
# ax2 = plt.Subplot(fig, inner[1],sharex = ax1)
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[5], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax2.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax2.get_lines(), xvals=(T - 50, T - 50), zorder=2)
# ax2.legend()
# ax2.set_ylabel('Bid Spread')
# ax2.set_xlabel('Time')
# # ax2.set_title('Optimal Ask and Bid, Inventory = %d' % I[j])
# xticks = ax2.get_xticks()
# ax2.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# ax2.set_ylim(-1.9, 0)
# fig.add_subplot(ax2)
# plt.subplots_adjust(hspace=.0)
#
# # fig.align_labels()
# plt.savefig('fooa.png', bbox_inches='tight')
#
#
# fig, ax = plt.subplots(tight_layout=True, figsize=(15,3))
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i],gT,hT, I[1], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
#     ax.plot(time_plot[:-1][-100:], [x-y for x,y in zip(temp[3][-100:],temp[4][-100:])], color=c[i], label = '$\lambda$ = {}'.format(-alphaT[i]))
# ax.set_ylabel('Bid-Ask Spread')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# labelLines(ax.get_lines(), xvals=(T - 50, T - 50), zorder=2)
# plt.savefig('foob.png', bbox_inches='tight')
#
#
# fig = plt.figure(figsize=(15,2))
# outer = gridspec.GridSpec(1, 2, wspace=0.2, hspace=0.3)
#
# # I = 1050
# inner = gridspec.GridSpecFromSubplotSpec(1, 1,subplot_spec=outer[0])
#
# ax1 = plt.Subplot(fig, inner[0])
# for i in range(len(alphaT)):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[2], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     X = time_plot[:-1][-100:]
#     Y = temp[3][-100:]
#     ax1.plot(X,Y, color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
#
# # labelLines(ax1.get_lines(), xvals=(T - 70, T - 70), zorder=2)
# ax1.legend()
# ax1.set_ylabel('Ask Spread')
# ax1.set_xlabel('Time')
# ax1.set_title('Optimal Ask, Inventory = %d' % I[2])
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# fig.add_subplot(ax1)
#
# # I = -1050
# inner = gridspec.GridSpecFromSubplotSpec(1, 1,subplot_spec=outer[1])
#
# ax2 = plt.Subplot(fig, inner[0])
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i], gT, hT, I[3], pip, pim, pioo, muop, muom, mutp, mutm, muoop, muoom, mutop,
#                           mutom, muttp, muttm, time)
#     ax2.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label='$\lambda$ = {}'.format(-alphaT[i]))
# # labelLines(ax2.get_lines(), xvals=(T - 70, T - 70), zorder=2)
# ax2.legend()
# ax2.set_ylabel('Bid Spread')
# ax2.set_xlabel('Time')
# ax2.set_title('Optimal Bid, Inventory = %d' % I[3])
# xticks = ax2.get_xticks()
# ax2.set_xticklabels([str(datetime.timedelta(seconds=x + 10 * 3600)) for x in xticks])
# fig.add_subplot(ax2)
#
# # fig.align_labels()
# plt.savefig('fooc.png', bbox_inches='tight')