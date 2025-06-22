# Source: https://github.com/RichardObi/frd-score/blob/main/frd_v1/src/utils.py

import numpy as np
from scipy import linalg
import os
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import seaborn as sns

# adapted from gan-metrics-pytorch
def frechet_distance(feats1, feats2, eps=1e-6, means_only=False):
    # feats1 and feats2 are N1 x M and N2 x M matrices
    m1 = np.mean(feats1, axis=0)
    s1 = np.cov(feats1, rowvar=False)
    m2 = np.mean(feats2, axis=0)
    s2 = np.cov(feats2, rowvar=False)

    mu1 = np.atleast_1d(m1)
    mu2 = np.atleast_1d(m2)

    sigma1 = np.atleast_2d(s1)
    sigma2 = np.atleast_2d(s2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    if means_only:
        return diff.dot(diff)

    # Product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # Numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            #raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1) +
            np.trace(sigma2) - 2 * tr_covmean)

def plot_tsne(feats1, feats2, feature_name, viz_folder):
    print("Running t-SNE on features...")
    emb = TSNE(n_components=2, perplexity=10, n_iter=10000, verbose=True).fit_transform(
            np.concatenate([feats1, feats2])
                        )

    # plot with seaborn with domain labels
    domain_labels1 = np.zeros(feats1.shape[0])
    domain_labels2 = np.ones(feats2.shape[0])
    all_domain_labels = np.concatenate([domain_labels1, domain_labels2])
    plt.figure(figsize=(6,6), dpi=300)
    g = sns.scatterplot(x=emb[:,0], y=emb[:,1], hue=all_domain_labels, palette="tab10", alpha=0.5)
    plt.title("{} feature representations".format(feature_name))

    legend_handles, _= g.get_legend_handles_labels()
    g.legend(legend_handles, ['1', '2'], 
        bbox_to_anchor=(1,1), 
        title="Dataset")
    
    plt.savefig(os.path.join(viz_folder, "{}_emb.png".format(feature_name)), bbox_inches="tight")