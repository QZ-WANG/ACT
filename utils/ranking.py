import torch

def sigma_selection(ranks, a, max_ratio, outliers=True, rounding=False, batch_size=-1):
    if not isinstance(ranks, torch.Tensor):
        ranks = torch.from_numpy(ranks)

    ranks = ranks.flatten()
    print("%s selection..." % ("Outlier" if outliers else "Inlier"))
    mu, sigma = torch.mean(ranks), torch.std(ranks)
    print("sigma thresholding: mu = %.4f, sigma = %.4f" % (mu, sigma))
    tau = mu + a * sigma
    candidates = torch.where(ranks > tau)[0] if outliers else torch.where(ranks < tau)[0]
    n_selected = candidates.size(0)

    if n_selected > int(max_ratio * ranks.size(0)):
        print("%s samples selected initially, greater than %.2f of all samples." % (n_selected, max_ratio))
        candidates = torch.topk(ranks, k=int(max_ratio * ranks.size(0)), largest=outliers)[1]
        if not rounding:
            print("reduce to %d samples now.\n" % candidates.size(0))
    else:
        if not rounding:
            print("Initial selection valid, %d samples selected.\n" % n_selected)

    if rounding and batch_size > 0:
        n_batches = candidates.size(0) / batch_size
        n_batches = int(n_batches) + 1 if n_batches - int(n_batches) >= 0.5 else int(n_batches)
        candidates = torch.topk(ranks, k=n_batches * batch_size, largest=outliers)[1]
        print("%5d selected after rounding" % candidates.size(0))
    return candidates


def pseudo_labeling_eval(candidates, ground_truth, outlier=True):
    values, counts = torch.cat((ground_truth, candidates)).unique(return_counts=True)
    correct = values[counts == 2].size(0)
    in_or_out = "outlier" if outlier else "inlier"

    if outlier:
        # Calculate precision and recall
        n_out = ground_truth.size(0)
        precision = correct / candidates.size(0)
        recall = correct / n_out

    print("%s pseudo labelling using source discriminator accuracy(precision): %.4f\n" % (in_or_out, correct / candidates.size(0)))
    if outlier:
        print("Precision: %.4f, Recall: %.4f, n_out: %4d, n_correct: %4d, n_candi: %4d" %
              (precision, recall, n_out, correct, candidates.size(0)))
