import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import datetime
from labellines import labelLine, labelLines
from matplotlib.ticker import FormatStrFormatter


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


####################################################################################################
# # Pi(1,1)=0. Different I
# # new p
#
# T = 23400
# lmbda = 10
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
# muttp = muttm = 4*10**6
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
# fig = plt.figure(tight_layout=True, figsize=(20, 8))
# gs = gridspec.GridSpec(2, 2,width_ratios=[1, 2])
#
# c = plt.rcParams['axes.prop_cycle'].by_key()[
#     'color']  # get default colormap in matlibplot such that each ask bid pair shares same color
#
# ax = fig.add_subplot(gs[:, 0])
# for i in range(len(I)):
#     ax.plot(time_plot[:-1], ab_res[i][0], color=c[i])
#     ax.plot(time_plot[:-1], ab_res[i][1], color=c[i])
# ax.set_ylabel('Bid-Ask Price')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# ax = fig.add_subplot(gs[0, 1])
# for j in range(len(I)):
#     ax.plot(time_plot[-100:], ab_res[j][0][-100:],label = 'I = %d' % I[j])
# labelLines(ax.get_lines(),xvals=(23350,23350),zorder=2.5)
# ax.set_ylabel('Ask Price')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# ax = fig.add_subplot(gs[1, 1])
# for j in range(len(I)):
#     ax.plot(time_plot[-100:], ab_res[j][1][-100:],label = 'I = %d' % I[j])
# labelLines(ax.get_lines(),xvals=(23350,23350),zorder=2.5)
# ax.set_ylabel('Bid Price')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
# plt.show()

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

######################################################################################
# Pi(1,1)=0. Different penalty. New Plot
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
#
# time_plot = list(time)
# time_plot.reverse()
#
# fig = plt.figure(tight_layout=True,figsize=(20, 8))
# gs = gridspec.GridSpec(3, 2)
#
# c = plt.rcParams['axes.prop_cycle'].by_key()[
#     'color']  # get default colormap in matlibplot such that each ask bid pair shares same color
# # I = -1500
# ax = fig.add_subplot(gs[0, 0])
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i],gT,hT, I[0], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
#     ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label = '$\lambda$ = %.1f' % -alphaT[i])
# labelLines(ax.get_lines(),xvals=(23250,23280),zorder=2)
# ax.set_ylabel('Ask Price')
# ax.set_xlabel('Time')
# ax.set_title('Optimal Ask, Inventory = %d' % I[0])
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# ax = fig.add_subplot(gs[0, 1])
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i],gT,hT, I[0], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
#     ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label = '$\lambda$ = %.1f' % -alphaT[i])
# labelLines(ax.get_lines(),xvals=(23250,23280),zorder=2)
# ax.set_ylabel('Bid Price')
# ax.set_xlabel('Time')
# ax.set_title('Optimal Bid, Inventory = %d' % I[0])
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# # I = -1000
# ax = fig.add_subplot(gs[1, 0])
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i],gT,hT, I[1], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
#     ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i],label = '$\lambda$ = %.1f' % -alphaT[i])
# labelLines(ax.get_lines(), xvals=(23275, 23275), zorder=2)
# ax.set_ylabel('Ask Price')
# ax.set_xlabel('Time')
# ax.set_title('Optimal Ask, Inventory = %d' % I[1])
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# ax = fig.add_subplot(gs[1, 1])
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i],gT,hT, I[1], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
#     ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i],label = '$\lambda$ = 0, 0.01, 1, 10, 100')
# labelLines(ax.get_lines(), xvals=(23280, 23280), zorder=2)
# ax.set_ylabel('Bid Price')
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# ax.set_xlabel('Time')
# ax.set_title('Optimal Bid, Inventory = %d' % I[1])
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# # ask I = -500
# ax = fig.add_subplot(gs[2, 0])
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i],gT,hT, I[2], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
#     ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label = '$\lambda$ = %.1f' % -alphaT[i])
# labelLines(ax.get_lines(),xvals=(23250,23280),zorder=2)
# ax.set_ylabel('Ask Price')
# ax.set_xlabel('Time')
# ax.set_title('Optimal Ask, Inventory = %d' % I[2])
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# ax = fig.add_subplot(gs[2, 1])
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i],gT,hT, I[2], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
#     ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label = '$\lambda$ = %.1f' % -alphaT[i])
# labelLines(ax.get_lines(),xvals=(23250,23280),zorder=2)
# ax.set_ylabel('Bid Price')
# ax.set_xlabel('Time')
# ax.set_title('Optimal Bid, Inventory = %d' % I[2])
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
#
# plt.show()
#
#
# fig = plt.figure(tight_layout=True,figsize=(20,8))
# gs = gridspec.GridSpec(3, 2)
#
# c = plt.rcParams['axes.prop_cycle'].by_key()[
#     'color']  # get default colormap in matlibplot such that each ask bid pair shares same color
#
# # I = 500
# ax = fig.add_subplot(gs[0, 0])
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i],gT,hT, I[3], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
#     ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label = '$\lambda$ = %.1f' % -alphaT[i])
# labelLines(ax.get_lines(),xvals=(23250,23280),zorder=2)
# ax.set_ylabel('Ask Price')
# ax.set_xlabel('Time')
# ax.set_title('Optimal Ask, Inventory = %d' % I[3])
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# ax = fig.add_subplot(gs[0, 1])
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i],gT,hT, I[3], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
#     ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label = '$\lambda$ = %.1f' % -alphaT[i])
# labelLines(ax.get_lines(),xvals=(23250,23280),zorder=2)
# ax.set_ylabel('Bid Price')
# ax.set_xlabel('Time')
# ax.set_title('Optimal Bid, Inventory = %d' % I[3])
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# # I = 1000
# ax = fig.add_subplot(gs[1,0])
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i],gT,hT, I[4], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
#     ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i],label = '$\lambda$ = 0, 0.01, 1, 10, 100')
# labelLines(ax.get_lines(), xvals=(23280, 23280), zorder=2)
# ax.set_ylabel('Ask Price')
# ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
# ax.set_xlabel('Time')
# ax.set_title('Optimal Ask, Inventory = %d' % I[4])
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# ax = fig.add_subplot(gs[1,1])
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i],gT,hT, I[4], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
#     ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i],label = '$\lambda$ = %.1f' % -alphaT[i])
# labelLines(ax.get_lines(), xvals=(23275, 23275), zorder=2)
# ax.set_ylabel('Bid Price')
# ax.set_xlabel('Time')
# ax.set_title('Optimal Bid, Inventory = %d' % I[4])
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# # ask I = 1500
# ax = fig.add_subplot(gs[2,0])
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i],gT,hT, I[5], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
#     ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label = '$\lambda$ = %.1f' % -alphaT[i])
# labelLines(ax.get_lines(),xvals=(23250,23280),zorder=2)
# ax.set_ylabel('Ask Price')
# ax.set_xlabel('Time')
# ax.set_title('Optimal Ask, Inventory = %d' % I[5])
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# ax = fig.add_subplot(gs[2,1])
# for i in reversed(range(len(alphaT))):
#     temp = trajectory_new(S, alphaT[i],gT,hT, I[5], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
#     ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label = '$\lambda$ = %.1f' % -alphaT[i])
# labelLines(ax.get_lines(),xvals=(23250,23280),zorder=2)
# ax.set_ylabel('Bid Price')
# ax.set_xlabel('Time')
# ax.set_title('Optimal Bid, Inventory = %d' % I[5])
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
#
#
# fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
#
# plt.show()

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

####################################################################################################
# Pi(1,1)>0. symetric (pi+,pi-), different I
# new p

# T = 23400
# lmbda = 10
# alphaT = -lmbda
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
#
# fig = plt.figure(tight_layout=True, figsize=(20, 8))
# gs = gridspec.GridSpec(2, 2,width_ratios=[1, 2])
#
# c = plt.rcParams['axes.prop_cycle'].by_key()[
#     'color']  # get default colormap in matlibplot such that each ask bid pair shares same color
#
# ax = fig.add_subplot(gs[:, 0])
# for i in range(len(I)):
#     ax.plot(time_plot[:-1], ab_res[i][0], color=c[i])
#     ax.plot(time_plot[:-1], ab_res[i][1], color=c[i])
# ax.set_ylabel('Bid-Ask Price')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# ax = fig.add_subplot(gs[0, 1])
# for j in range(len(I)):
#     ax.plot(time_plot[-100:], ab_res[j][0][-100:],label = 'I = %d' % I[j])
# labelLines(ax.get_lines(),xvals=(23350,23350),zorder=2.5)
# ax.set_ylabel('Ask Price')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# ax = fig.add_subplot(gs[1, 1])
# for j in range(len(I)):
#     ax.plot(time_plot[-100:], ab_res[j][1][-100:],label = 'I = %d' % I[j])
# labelLines(ax.get_lines(),xvals=(23350,23350),zorder=2.5)
# ax.set_ylabel('Bid Price')
# ax.set_xlabel('Time')
# xticks = ax.get_xticks()
# ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])
#
# fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()
# plt.show()
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

# Pi(1,1)=0. Different penalty. New Plot

T = 23400
lmbda = [0,0.1,1,10,100]
alphaT = [-x for x in lmbda]
I= [-1500,-1200,-500,500,1200,1500]
gT = 0
hT = 0
# pip+pim-1V0<=pioo<=pip/\pim
pip = 0.4
pim = 0.4
pioo = 0.25

muop = muom = 500
mutp = mutm = 10**6  # mut>=muo^2
Ep = 1
muoop = muoom = muop * Ep
mutop = mutom = mutp * Ep
muttp = muttm = 4*10**6

# # Deltap>0, Deltam<0
# Deltap = 5
# Deltam = -5
#
# sigmap = sigmam = tldsigma = 0.00013  # (second to second)

time = np.linspace(0, T, T / 3 + 1)[::-1]
S = 0

time_plot = list(time)
time_plot.reverse()

fig = plt.figure(tight_layout=True,figsize=(20, 8))
gs = gridspec.GridSpec(3, 2)

c = plt.rcParams['axes.prop_cycle'].by_key()[
    'color']  # get default colormap in matlibplot such that each ask bid pair shares same color
# I = -1500
ax = fig.add_subplot(gs[0, 0])
for i in reversed(range(len(alphaT))):
    temp = trajectory_new(S, alphaT[i],gT,hT, I[0], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
    ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label = '$\lambda$ = %.1f' % -alphaT[i])
labelLines(ax.get_lines(),xvals=(23150,23350),zorder=2)
ax.set_ylabel('Ask Price')
ax.set_xlabel('Time')
ax.set_title('Optimal Ask, Inventory = %d' % I[0])
xticks = ax.get_xticks()
ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])

ax = fig.add_subplot(gs[0, 1])
for i in reversed(range(len(alphaT))):
    temp = trajectory_new(S, alphaT[i],gT,hT, I[0], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
    ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label = '$\lambda$ = %.1f' % -alphaT[i])
labelLines(ax.get_lines(),xvals=(23150, 23350),zorder=2)
ax.set_ylabel('Bid Price')
ax.set_xlabel('Time')
ax.set_title('Optimal Bid, Inventory = %d' % I[0])
xticks = ax.get_xticks()
ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])

# I = -1000
ax = fig.add_subplot(gs[1, 0])
for i in reversed(range(len(alphaT))):
    temp = trajectory_new(S, alphaT[i],gT,hT, I[1], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
    ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i],label = '$\lambda$ = %.1f' % -alphaT[i])
labelLines(ax.get_lines(), xvals=(23150, 23350), zorder=2)
ax.set_ylabel('Ask Price')
ax.set_xlabel('Time')
ax.set_title('Optimal Ask, Inventory = %d' % I[1])
xticks = ax.get_xticks()
ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])

ax = fig.add_subplot(gs[1, 1])
for i in reversed(range(len(alphaT))):
    temp = trajectory_new(S, alphaT[i],gT,hT, I[1], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
    ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i],label = '$\lambda$ = %.1f' % -alphaT[i])
labelLines(ax.get_lines(), xvals=(23210, 23390), zorder=2)
ax.set_ylabel('Bid Price')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xlabel('Time')
ax.set_title('Optimal Bid, Inventory = %d' % I[1])
xticks = ax.get_xticks()
ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])

# ask I = -500
ax = fig.add_subplot(gs[2, 0])
for i in reversed(range(len(alphaT))):
    temp = trajectory_new(S, alphaT[i],gT,hT, I[2], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
    ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label = '$\lambda$ = %.1f' % -alphaT[i])
labelLines(ax.get_lines(),xvals=(23150, 23350),zorder=2)
ax.set_ylabel('Ask Price')
ax.set_xlabel('Time')
ax.set_title('Optimal Ask, Inventory = %d' % I[2])
xticks = ax.get_xticks()
ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])

ax = fig.add_subplot(gs[2, 1])
for i in reversed(range(len(alphaT))):
    temp = trajectory_new(S, alphaT[i],gT,hT, I[2], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
    ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label = '$\lambda$ = %.1f' % -alphaT[i])
labelLines(ax.get_lines(),xvals=(23200, 23350),zorder=2)
ax.set_ylabel('Bid Price')
ax.set_xlabel('Time')
ax.set_title('Optimal Bid, Inventory = %d' % I[2])
xticks = ax.get_xticks()
ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])

fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()

plt.show()


fig = plt.figure(tight_layout=True,figsize=(20,8))
gs = gridspec.GridSpec(3, 2)

c = plt.rcParams['axes.prop_cycle'].by_key()[
    'color']  # get default colormap in matlibplot such that each ask bid pair shares same color

# I = 500
ax = fig.add_subplot(gs[0, 0])
for i in reversed(range(len(alphaT))):
    temp = trajectory_new(S, alphaT[i],gT,hT, I[3], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
    ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label = '$\lambda$ = %.1f' % -alphaT[i])
labelLines(ax.get_lines(),xvals=(23160, 23360),zorder=2)
ax.set_ylabel('Ask Price')
ax.set_xlabel('Time')
ax.set_title('Optimal Ask, Inventory = %d' % I[3])
xticks = ax.get_xticks()
ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])

ax = fig.add_subplot(gs[0, 1])
for i in reversed(range(len(alphaT))):
    temp = trajectory_new(S, alphaT[i],gT,hT, I[3], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
    ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label = '$\lambda$ = %.1f' % -alphaT[i])
labelLines(ax.get_lines(),xvals=(23150, 23350),zorder=2)
ax.set_ylabel('Bid Price')
ax.set_xlabel('Time')
ax.set_title('Optimal Bid, Inventory = %d' % I[3])
xticks = ax.get_xticks()
ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])

# I = 1000
ax = fig.add_subplot(gs[1,0])
for i in reversed(range(len(alphaT))):
    temp = trajectory_new(S, alphaT[i],gT,hT, I[4], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
    ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i],label = '$\lambda$ = %.1f' % -alphaT[i])
labelLines(ax.get_lines(), xvals=(23210, 23390), zorder=2)
ax.set_ylabel('Ask Price')
ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
ax.set_xlabel('Time')
ax.set_title('Optimal Ask, Inventory = %d' % I[4])
xticks = ax.get_xticks()
ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])

ax = fig.add_subplot(gs[1,1])
for i in reversed(range(len(alphaT))):
    temp = trajectory_new(S, alphaT[i],gT,hT, I[4], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
    ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i],label = '$\lambda$ = %.1f' % -alphaT[i])
labelLines(ax.get_lines(), xvals=(23150, 23350), zorder=2)
ax.set_ylabel('Bid Price')
ax.set_xlabel('Time')
ax.set_title('Optimal Bid, Inventory = %d' % I[4])
xticks = ax.get_xticks()
ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])

# ask I = 1500
ax = fig.add_subplot(gs[2,0])
for i in reversed(range(len(alphaT))):
    temp = trajectory_new(S, alphaT[i],gT,hT, I[5], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
    ax.plot(time_plot[:-1][-100:], temp[3][-100:], color=c[i], label = '$\lambda$ = %.1f' % -alphaT[i])
labelLines(ax.get_lines(),xvals=(23150, 23350),zorder=2)
ax.set_ylabel('Ask Price')
ax.set_xlabel('Time')
ax.set_title('Optimal Ask, Inventory = %d' % I[5])
xticks = ax.get_xticks()
ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])

ax = fig.add_subplot(gs[2,1])
for i in reversed(range(len(alphaT))):
    temp = trajectory_new(S, alphaT[i],gT,hT, I[5], pip, pim, pioo,muop, muom, mutp, mutm, muoop, muoom, mutop, mutom, muttp, muttm, time)
    ax.plot(time_plot[:-1][-100:], temp[4][-100:], color=c[i], label = '$\lambda$ = %.1f' % -alphaT[i])
labelLines(ax.get_lines(),xvals=(23150, 23350),zorder=2)
ax.set_ylabel('Bid Price')
ax.set_xlabel('Time')
ax.set_title('Optimal Bid, Inventory = %d' % I[5])
xticks = ax.get_xticks()
ax.set_xticklabels([str(datetime.timedelta(seconds=x + 9.5 * 3600)) for x in xticks])

fig.align_labels()  # same as fig.align_xlabels(); fig.align_ylabels()

plt.show()

# res = alpha_res, h_res, g_res, a_res, b_res, a1p_res, a2p_res, a3p_res, a1m_res, a2m_res, a3m_res
# plt.plot(res[2])