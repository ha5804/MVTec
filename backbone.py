from torchvision import models, transforms
model = models.resnet18(weights = 
                        models.ResNet18_Weights.IMAGENET1K_v1)

class PatchCore(nn.Module):
    def __init__(
        self,
        backbone,
        per_memory_bank_size=4,
        memory_bank_path="./memory_bank",
        device=device,
        target_dim=384 * 3,  # |D|
        patch_size=3,  # |P|
        d=3,  # nearest
    ):
        """
        Args:
            backbone: torch.nn.Module
            per_memory_bank_size: int
            memory_bank_path: str
            device: torch.device
            target_dim: int
            patch_size: int
            d: int
        """

        super().__init__()
        self.backbone = model
        self.backbone.eval()

        self.layer2_output, self.layer3_output = None, None
        self.register_hook_for_layer2()  # register hook for layer2
        self.register_hook_for_layer3()  # register hook for layer3

        self.memory_bank = None
        self.per_memory_bank_size = per_memory_bank_size
        self.memory_bank_path = memory_bank_path
        self.device = device
        self.target_dim = target_dim
        self.average_pool = nn.AdaptiveAvgPool1d(self.target_dim)
        self.patch_size = patch_size
        self.d = d

    def register_hook_for_layer2(self):
        self.backbone.layer2.register_forward_hook(self._register_hook_for_layer2)

    def _register_hook_for_layer2(self, module, input, output):  # (B, 128, 28, 28)
        layer2_output = output
        self.layer2_output = layer2_output  # (B, 128, 28, 28)
        # self.layer2_output = None

    def register_hook_for_layer3(self):
        self.backbone.layer3.register_forward_hook(self._register_hook_for_layer3)

    def _register_hook_for_layer3(self, module, input, output):  # (B, 256, 14, 14)
        layer3_output = output
        layer3_output = nn.functional.interpolate(
            layer3_output, scale_factor=2, mode="bilinear"
        )
        self.layer3_output = layer3_output  # (B, 256, 28, 28)

    def save_memory_bank(self, train_batch, file_name):
        """
        training batch를 받아서 memory bank에 저장한다.
        """
        self.backbone(train_batch)
        path = os.path.join(self.memory_bank_path, file_name)
        new_memory = self.patch_collection(self.patch_size)
        new_memory = new_memory.reshape(-1, new_memory.shape[-1])
        torch.save(new_memory, path)

    def loader_memory_bank(self):
        """
        memory bank를 이터레이터로 불러온다.
        """
        file_list = os.listdir(self.memory_bank_path)
        file_list = [file for file in file_list if file.endswith(".pth")]
        file_list.sort()
        for file in file_list:
            path = os.path.join(self.memory_bank_path, file)
            new_memory = torch.load(path, map_location=self.device)

            yield new_memory

    def forward(self, x):
        self.backbone(x)
        query = self.patch_collection(self.patch_size)  # (N, |A|, D)
        l2 = torch.Tensor([]).to(device)
        for memory in self.loader_memory_bank():
            l2 = torch.cat((l2, self.cal_l2(query, memory)), dim=1)
            del memory

        s_ = self.get_anomaly_score(l2)
        s = self.update_anomaly_score(s_, l2, self.d)
        return s

    def get_anomaly_score(self, l2):
        """
        Args:
            l2: (query, N', |A|)
        """
        min_l2 = l2.min(dim=1).values  # (query, |A|)
        max_min_l2 = min_l2.max(dim=1).values  # (query, )
        return max_min_l2

    def update_anomaly_score(self, s_, l2: torch.Tensor, d):
        """
        Args:
            l2: (query, N', |A|)
            d = nearest
        """
        m_train = l2.min(dim=1)  # (query, |A|)
        m_test = m_train.values.max(dim=1)  # (query)
        m_for_test = l2[:, :, m_test.indices]  # (query, N')
        m_train_nearest = m_for_test.topk(k=d, dim=1).values  # (query, d)

        update_weight = 1 - (
            torch.exp(m_test.values) / torch.exp(m_train_nearest).sum(dim=1)
        )
        s = s_ * update_weight
        return s

    def cal_l2(self, query, memory_bank):
        """
        return: (query_size, memory_bank_size, |A|)
        """
        memory_bank = memory_bank.unsqueeze(1)  # (N', 1, D)
        N, D = memory_bank.shape[0], memory_bank.shape[2]
        memory_bank = memory_bank.expand(N, query.shape[1], D)  # (N', |A|, D)
        l2 = []
        for q in query:
            q = q.unsqueeze(0)  # (1, |A|, D)
            diff = memory_bank - q  # (N', |A|, D)
            l2_ = diff.square().sum(dim=2)  # (N', |A|)
            l2_ = l2_.sqrt()
            l2.append(l2_)
            del q
        l2 = torch.stack(l2, dim=0)  # (query, N', |A|)
        return l2

    def feature(self, h, w):
        """
        return: (N, C)
        """
        H, W = self.layer2_output.shape[2], self.layer2_output.shape[3]
        if not (0 <= h < H and 0 <= w < W):
            return torch.tensor([]).to(self.device)
        layer2 = self.layer2_output[:, :, h, w]  # (B, C)

        # TODO: 아래 코드
        if self.layer3_output is not None:
            layer3 = self.layer3_output[:, :, h, w]  # (B, C')
        else:
            layer3 = torch.tensor([]).to(self.device)

        feature = torch.cat((layer2, layer3), dim=1)
        return feature

    def neighborhood_features(self, h, w, patch_size):
        """
        return: (N, |P|, C) -> path_size x patch_size
        """
        features = []
        for i in range(math.floor(-patch_size / 2), math.floor(patch_size / 2)):
            for j in range(math.floor(-patch_size / 2), math.floor(patch_size / 2)):
                feature = self.feature(h + i, w + j)
                if feature.shape[0] == 0:
                    continue
                features.append(feature)
        features = torch.stack(features, dim=1)
        return features

    def patch(self, h, w, patch_size):
        """
        return: (N, D)
        """
        features = self.neighborhood_features(h, w, patch_size)
        features = features.permute(0, 2, 1)  # (N, C, |P|)
        features = features.reshape(features.shape[0], -1)  # (N, C X |P|)
        features = self.average_pool(features)  # (N, target_dim) = (N, D)
        return features

    def patch_collection(self, patch_size):
        """
        return: (N, |A|, D)
        """
        H, W = self.layer2_output.shape[2], self.layer2_output.shape[3]

        patch_collection = []
        # average pooling으로 나눌 수 있게 중심만 고려한다.
        for h in range(math.ceil(patch_size / 2), H - math.ceil(patch_size / 2)):
            for w in range(math.ceil(patch_size / 2), W - math.ceil(patch_size / 2)):
                patch = self.patch(h, w, patch_size)
                patch_collection.append(patch)
        patch_collection = torch.stack(patch_collection, dim=1)
        return patch_collection