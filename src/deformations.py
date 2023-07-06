# Based on https://people.engr.tamu.edu/schaefer/research/mls.pdf

import numpy as np

np.seterr(divide='ignore', invalid='ignore')


def get_common_variables(alpha, eps, p, q, vs):
    w = np.zeros((p.shape[0], vs.shape[0]))
    for i, pi in enumerate(p):
        w[i] = 1 / (np.sum((pi - vs) ** (2 * alpha), axis=1) + eps)

    pstar = np.zeros((vs.shape[0], 2))
    for i, pi in enumerate(p):
        pstar += w[i].reshape((-1, 1)) * pi
    pstar /= w.sum(axis=0).reshape((-1, 1))

    qstar = np.zeros((vs.shape[0], 2))
    for i, qi in enumerate(q):
        qstar += w[i].reshape((-1, 1)) * qi
    qstar /= w.sum(axis=0).reshape((-1, 1))

    phat = (p[:, None, :] - pstar)[:, :, None, :]  # c x v x 1 x 2
    qhat = (q[:, None, :] - qstar)[:, :, None, :]  # c x v x 1 x 2

    return phat, pstar, qhat, qstar, w


def mls_affine_deformation(vs, p, q, alpha=1.0, eps=1e-8):
    phat, pstar, qhat, qstar, w = get_common_variables(alpha, eps, p, q, vs)
    phat_T = np.transpose(phat, (0, 1, 3, 2))

    pTwp_inv = np.linalg.inv(
        (w[:, :, None, None] * (phat_T @ phat)).sum(axis=0)
    )
    wpTq = (w[:, :, None, None] * (phat_T @ qhat)).sum(axis=0)

    M = pTwp_inv @ wpTq  # v x 2 x 2

    return ((vs - pstar)[:, None, :] @ M).squeeze() + qstar  # v x 2


def mls_similarity_deformation(vs, p, q, alpha=1.0, eps=1e-8):
    phat, pstar, qhat, qstar, w = get_common_variables(alpha, eps, p, q, vs)

    qhat_T = np.transpose(qhat, (0, 1, 3, 2))  # c x v x 2 x 1

    neg_phat_perp = phat[..., [1, 0]]
    neg_phat_perp[..., 1] *= -1

    neg_qhat_T_perp = qhat_T[..., [1, 0], :]
    neg_qhat_T_perp[..., 1, :] *= -1

    mu_s = (w[:, :, None, None] * (phat @ np.transpose(phat, (0, 1, 3, 2)))).sum(axis=0)  # c x v x 1 x 1

    augmented_phat = np.concatenate((phat, neg_phat_perp), 2)  # c x v x 2 x 2
    augmented_qhat = np.concatenate((qhat_T, neg_qhat_T_perp), 3)  # c x v x 2 x 2

    M = (w[:, :, None, None] * (augmented_phat @ augmented_qhat)).sum(axis=0) / mu_s  # v x 2 x 2

    return ((vs - pstar)[:, None, :] @ M).squeeze() + qstar  # v x 2


def mls_rigid_deformation(vs, p, q, alpha=1.0, eps=1e-8):
    phat, pstar, qhat, qstar, w = get_common_variables(alpha, eps, p, q, vs)

    phat_T = np.transpose(phat, (0, 1, 3, 2))  # c x v x 2 x 1
    qhat_T = np.transpose(qhat, (0, 1, 3, 2))  # c x v x 2 x 1

    # phat (c x v x 1 x 2)
    phat_perp = phat[..., [1, 0]]  # swop x, y => y, x
    phat_perp[..., 0] *= -1  # y, x => -y, x
    phat_perp_T = np.transpose(phat_perp, (0, 1, 3, 2))
    neg_phat_perp = phat_perp
    neg_phat_perp[..., :] *= -1

    neg_qhat_T_perp = qhat_T[..., [1, 0], :]
    neg_qhat_T_perp[..., 1, :] *= -1

    sum1 = (w[:, :, None, None] * qhat @ phat_T).sum(axis=0)  # c x v x 1 x 1
    sum2 = (w[:, :, None, None] * qhat @ phat_perp_T).sum(axis=0)  # c x v x 1 x 1
    mu_r = np.sqrt(np.square(sum1), np.square(sum2))

    augmented_phat = np.concatenate((phat, neg_phat_perp), 2)  # c x v x 2 x 2
    augmented_qhat = np.concatenate((qhat_T, neg_qhat_T_perp), 3)  # c x v x 2 x 2

    M = (w[:, :, None, None] * (augmented_phat @ augmented_qhat)).sum(axis=0) / mu_r  # v x 2 x 2

    return ((vs - pstar)[:, None, :] @ M).squeeze() + qstar  # v x 2
