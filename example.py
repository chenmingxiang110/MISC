def gen_live_mask(size, step, update_mean=0.25, update_std=0.25, mature_threshold=0.5):
    b_size = np.random.random() * 1.5 + 1.0

    mask = np.zeros([size, size])
    dist_linf = np.max(np.abs(np.array(
        np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    ) - np.array([size//2,size//2])[:,None,None]), axis=0)

    d_e = step * update_mean / mature_threshold
    d_v = step * update_std ** 2 / mature_threshold ** 2
    d_v_ = np.sqrt(d_v)

    _rnd = np.random.normal(d_e, d_v_, size=(size,size))
    iter_map = (_rnd - dist_linf) * (_rnd > dist_linf)

    rnd = iter_map * mature_threshold / update_mean
    rnd = np.random.normal(size=rnd.shape) * rnd * update_std + rnd * update_mean # aliveness
    # make it rounded
    norm_val = 1 / np.random.random() + 1
    if norm_val<10:
        rnd = norm_warp(rnd, norm_val)

    b0, b1 = int(np.floor(b_size)), int(np.ceil(b_size))
    w = b1 - b_size
    res = rnd
    res = cv2.blur(res, (b0, b0)) * w + cv2.blur(res, (b1, b1)) * (1-w)
    # res += np.random.normal(size=res.shape) * 0.1 * (np.abs(res) > 1e-6)
    res = np.clip(res, 0, 1)
    res *= (dist_linf+1e-8<step)

    return res
