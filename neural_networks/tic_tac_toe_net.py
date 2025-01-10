from torch import Tensor, nn, softmax


class ConvBlock(nn.Module):
    """
    Initial convolution block before the residual blocks. Extracts low level features and smooths
    data to help deep layers discover more complex features. Also transforms data into correct
    spacial dimensions.
    """

    def __init__(self, num_in: int, num_out: int):
        super().__init__()
        self.conv = nn.Conv2d(num_in, num_out, (3, 3), 1, 1)
        self.batch_norm = nn.BatchNorm2d(num_out)
        self.relu = nn.ReLU()

    def forward(self, x):
        # Low level feature extraction.
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class ResBlock(nn.Module):
    """
    Residual block to allow for skip connection to mitigate vanishing gradient problem in deep
    networks.

    By adding the original input during the skip connection, it gives a "head start".
    Instead of learning a very complex direct mapping y = f(x), a residual mapping y = f(x) + x i.e
    the residual between the input and output.
    """

    def __init__(self, num_in: int, num_out: int):
        super().__init__()
        self.conv_1 = nn.Conv2d(num_in, num_out, (3, 3), 1, 1)
        self.batch_norm_1 = nn.BatchNorm2d(num_out)
        self.relu_1 = nn.ReLU()
        self.conv_2 = nn.Conv2d(num_in, num_out, (3, 3), 1, 1)
        self.batch_norm_2 = nn.BatchNorm2d(num_out)
        self.relu_3 = nn.ReLU()

    def forward(self, x):
        # Preserve residual for skip connection.
        residual = x
        # Feature extraction.
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        # Further feature extraction.
        x = self.conv_2(x)
        x = self.batch_norm_2(x)
        # Apply skip connection.
        x += residual
        x = self.relu_3(x)
        return x


class PolicyHead(nn.Module):
    """
    Consumes feature map from residual blocks and returns a vector of probabilities for each
    possible move.
    """

    def __init__(self, num_in: int, num_out: int, num_feats: int):
        super().__init__()
        self.conv_1 = nn.Conv2d(num_in, num_feats, (3, 3), 1, 1)
        self.batch_norm_1 = nn.BatchNorm2d(num_feats)
        self.relu_1 = nn.ReLU()
        self.flatten_2 = nn.Flatten()
        self.linear_2 = nn.Linear(num_feats * num_out, num_out)

    def forward(self, x):
        # Feature extraction.
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        # Transform feature map to move probabilities.
        x = self.flatten_2(x)
        x = self.linear_2(x)
        return x


class ValueHead(nn.Module):
    """
    Consumes feature map from residual blocks and returns a scaler score representing how good the
    current board position is.
    """

    def __init__(self, num_in: int, action_size: int, num_feats: int):
        super().__init__()
        self.conv_1_out_size = num_feats * action_size
        self.conv1 = nn.Conv2d(num_in, self.conv_1_out_size, (3, 3), 1, 1)
        self.batch_norm_1 = nn.BatchNorm2d(self.conv_1_out_size)
        self.relu_1 = nn.ReLU()
        self.flatten_2_out_size = self.conv_1_out_size * action_size
        self.flatten_2 = nn.Flatten()
        self.linear_2 = nn.Linear(self.flatten_2_out_size, num_feats)
        self.relu_2 = nn.ReLU()
        self.linear_3 = nn.Linear(num_feats, 1)
        self.tanh_3 = nn.Tanh()

    def forward(self, x):
        # Feature extraction.
        x = self.conv1(x)
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        # Feature compression.
        x = self.flatten_2(x)
        x = self.linear_2(x)
        x = self.relu_2(x)
        # Map compressed features to scalar score in [-1, 1].
        x = self.linear_3(x)
        x = self.tanh_3(x)
        return x


class TicTacToeNet(nn.Module):
    """
    Residual tower consisting of an initial convolution block, then several residual blocks. The
    output of this is passed to 2 separate heads which calculate the policy and state value.
    """

    def __init__(self, input_feats: Tensor, hparams: dict):
        super().__init__()
        self.action_size = input_feats.shape[2] * input_feats.shape[3]
        self.conv_block = ConvBlock(input_feats.shape[3], hparams["hidden"])
        self.res_blocks = nn.ModuleList(
            [ResBlock(hparams["hidden"], hparams["hidden"])] * hparams["res_blocks"]
        )
        self.policy_head = PolicyHead(hparams["hidden"], self.action_size, hparams["pol_feats"])
        self.value_head = ValueHead(hparams["hidden"], self.action_size, hparams["val_feats"])

    def parse_output(self, policy: Tensor, value: Tensor) -> tuple[Tensor, float]:
        """
        Transform the raw policy and value output into actual probabilities.
        """
        if policy.get_device != -1:
            parsed_policy = softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
        else:
            raise NotImplementedError("Parsing for GPU not implemented yet.")
        parsed_value = value.item()
        return parsed_policy, parsed_value

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        x = self.conv_block(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

    def get_pol_param_grp(self) -> list[Tensor]:
        return list(self.policy_head.parameters())

    def get_val_param_grp(self) -> list[Tensor]:
        return list(self.value_head.parameters())

    def get_oth_param_grp(self) -> list[Tensor]:
        oth_params = []
        for name, param in self.named_parameters():
            if "policy_head" not in name and "value_head" not in name:
                oth_params.append(param)
        return oth_params
