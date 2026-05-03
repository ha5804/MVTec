import torch
import torch.nn.functional as F

class PatchCore:
    def __init__(self, model, k=1000, device="cpu"):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.k = k  # coreset size

        self.memory_bank = None
        self.features = []

        # hook 등록
        self.model.layer2.register_forward_hook(self._hook)
        self.model.layer3.register_forward_hook(self._hook)

    def _hook(self, module, input, output):
        self.features.append(output)

    def _extract_features(self, x):
        self.features = []

        with torch.no_grad():
            _ = self.model(x)

        f2, f3 = self.features

        # resize
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode='bilinear', align_corners=False)

        # concat
        f = torch.cat([f2, f3], dim=1)

        # patch
        B, C, H, W = f.shape
        patch = f.permute(0, 2, 3, 1).reshape(B, H*W, C)

        return patch, H, W

    def fit(self, dataset):
        patches = []

        for i in range(len(dataset)):
            img, _ = dataset[i]
            img = img.unsqueeze(0).to(self.device)

            patch, _, _ = self._extract_features(img)
            patches.append(patch)

        # concat
        patches = torch.cat(patches, dim=0)
        patches = patches.reshape(-1, patches.shape[-1])

        # coreset
        self.memory_bank = self._greedy_coreset(patches, self.k)

    def _greedy_coreset(self, x, k):
        n = x.shape[0]
        k = min(k, n)

        idx = torch.randint(0, n, (1,), device=x.device)
        selected = [idx.item()]

        dist = torch.cdist(x[idx], x).squeeze(0)

        for _ in range(k - 1):
            idx = torch.argmax(dist)
            selected.append(idx.item())

            new_dist = torch.cdist(x[idx].unsqueeze(0), x).squeeze(0)
            dist = torch.minimum(dist, new_dist)

        return x[selected]

    def predict(self, img):
        img = img.unsqueeze(0).to(self.device)

        patch, H, W = self._extract_features(img)
        patch = patch.squeeze(0)

        # 거리 계산
        dist = torch.cdist(patch, self.memory_bank)

        min_dist, _ = dist.min(dim=1)

        # image score
        score = min_dist.max()

        # heatmap
        heatmap = min_dist.reshape(H, W)

        return score, heatmap
