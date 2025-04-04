imgs = []
max_full = -1
size = 16
update_mean = 0.25
update_std = 0.25
mature_threshold = 0.5
n_steps_max = 24
n_keep = 8
noise_std_a, noise_std_b, noise_std_power = 1.5, 1.1, 2
noise_stds = np.linspace(noise_std_a, noise_std_b, n_steps_max) ** noise_std_power

for _ in trange(24):
    n_full = 0

    hs = torch.zeros([1,32,size,size]).to(device)
    mask = np.zeros([size,size])
    for step in range(n_steps_max):
        if max_full>0 and n_full>=max_full:
            continue
        mask_ = build_next_live_mask(mask, mature_threshold=mature_threshold)
        update = np.random.normal(update_mean, update_std, size=(size, size))
        mask_new = mask + update * mask_
        mask = mask_new

        with torch.no_grad():
            m0 = np.clip(mask, 0, 1)[None,None]
            m1 = build_next_live_mask(m0[0,0])[None,None]
            xs = torch.from_numpy(m0.astype(np.float32)).to(device) * hs.to(device)

            # noise = (torch.randn(xs.shape) * torch.exp(torch.randn(xs.shape[0],1,1,1)*0.5 - 1)).to(device)
            noise = (torch.randn(xs.shape) * torch.rand(xs.shape[0],1,1,1) * noise_stds[step]).to(device)
            noise = noise * torch.from_numpy(m0.astype(np.float32)).to(device)

            m1 = build_next_live_mask(m0[0,0])[None,None]

            dx = model_ca(xs + noise) * torch.from_numpy(m1.astype(np.float32)).to(device)
            hs = xs + noise + dx

            img_h = decode_single_z(hs[0], zs_mean, zs_std)

        if np.mean(m1)+1e-8>=1:
            n_full+=1
        else:
            n_full = 0

    for step in range(n_keep):
        if max_full>0 and n_full>=max_full:
            continue
        mask_ = build_next_live_mask(mask, mature_threshold=mature_threshold)
        update = np.random.normal(update_mean, update_std, size=(size, size))
        mask_new = mask + update * mask_
        mask = mask_new

        with torch.no_grad():
            dx = model_ca(hs)
            hs = hs + dx

            img_h = decode_single_z(hs[0], zs_mean, zs_std)

        if np.mean(m1)+1e-8>=1:
            n_full+=1
        else:
            n_full = 0

    imgs.append(img_h)

plt.figure(figsize=(6 * 8, 6 * 3))
for i, img in enumerate(imgs):
    plt.subplot(3,8,i+1)
    plt.imshow(img)
plt.show()
