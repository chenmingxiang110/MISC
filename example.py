项目背景
在N设备相关算法（包括小指令相关技术开发，以及无界营销在内的推广活动）开发过程中，OCR或广义屏幕理解算法是不可缺少的一环。使用多模态模型方案与当前业务结合，可有效提高算法识别准确率，并降低人力资源成本。

业务价值
1. 提高现有算法准确率，确保当前业务更加平稳有效运行，可以保障产品功能在线下更完整地呈现，提高产品在用户中的口碑，形成产品优化和用户增长的正循环。
2. 有效减少人力标注成本，提高小指令相关场景的自动化、智能化水平。
3. 促进其他收银、文字处理相关业务的产品迭代（例如：小票/发票的智能结构化、防欺诈与合规审查等）。
4. 量变引起质变，驱动业务创新。当算法准确率提高到更高水平时，利于探索从前不敢实现的全新营销、推广、用户体验相关新产品。

技术价值
VLM-OCR（基于视觉语言模型的OCR技术）相比传统OCR在多个维度上实现了突破，其核心技术价值包括
1. 多模态理解能力：自动关联互相有联系的上下文，无需事先标注，自动实现结构化输出，解决当前标注人力成本问题。
2. 复杂场景适应性：突破传统OCR的局限，对模糊、倾斜、遮挡文本具有更强鲁棒性，提高各场景下任务泛化能力。
3. 端到端输出：减少前处理、后处理依赖，减少超参数调试轮数，提高算法迭代速度。
4. 通过跨模态分析，实现交互式OCR，动态响应用户需求，创造更多使用场景。
5. 提高公司在业界、学界的辨识度。

预计产出
小指令/无界营销相关业务落地：小指令、无界营销等场景识别能力提升；相关标注人力成本下降90%以上。
相关高质量学术论文1-2篇


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
