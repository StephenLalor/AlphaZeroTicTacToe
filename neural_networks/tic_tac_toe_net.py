import torch


class ConvBlock(torch.nn.Module):
    """
    Initial convolution block before the residual blocks. Extracts low level features and smooths
    data to help deep layers discover more complex features. Also transforms data into correct
    spacial dimensions.
    """

    def __init__(self, num_in: int, num_out: int):
        super().__init__()
        self.conv = torch.nn.Conv2d(num_in, num_out, kernel_size=3, padding=1)
        self.batch_norm = torch.nn.BatchNorm2d(num_out)
        self.relu = torch.nn.ReLU(inplace=False)

    def forward(self, x):
        # Low level feature extraction.
        x = self.conv(x)
        x = self.batch_norm(x)
        x = self.relu(x)
        return x


class ResBlock(torch.nn.Module):
    """
    Residual block to allow for skip connection to mitigate vanishing gradient problem in deep
    networks.

    By adding the original input during the skip connection, it gives a "head start".
    Instead of learning a very complex direct mapping y = f(x), a residual mapping y = f(x) + x i.e
    the residual between the input and output.
    """

    def __init__(self, num_in: int, num_out: int):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(num_in, num_out, kernel_size=3, padding=1)
        self.batch_norm_1 = torch.nn.BatchNorm2d(num_out)
        self.relu_1 = torch.nn.ReLU(inplace=False)
        self.conv_2 = torch.nn.Conv2d(num_in, num_out, kernel_size=3, padding=1)
        self.batch_norm_2 = torch.nn.BatchNorm2d(num_out)
        self.relu_3 = torch.nn.ReLU(inplace=False)

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
        x = x + residual
        x = self.relu_3(x)
        return x


class PolicyHead(torch.nn.Module):
    """
    Consumes feature map from residual blocks and returns a vector of probabilities for each
    possible move.
    """

    def __init__(self, num_in: int, num_out: int, num_feats: int, dropout: float):
        super().__init__()
        self.conv_1 = torch.nn.Conv2d(num_in, num_feats, kernel_size=3, padding=1)
        self.batch_norm_1 = torch.nn.BatchNorm2d(num_feats)
        self.relu_1 = torch.nn.ReLU(inplace=False)
        self.dropout_1 = torch.nn.Dropout(dropout)
        self.flatten_2 = torch.nn.Flatten()
        self.linear_2 = torch.nn.Linear(num_feats * 9, num_out)

    def forward(self, x):
        # Feature extraction.
        x = self.conv_1(x)
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        x = self.dropout_1(x)
        # Transform feature map to move probabilities.
        x = self.flatten_2(x)
        x = self.linear_2(x)
        return x


class ValueHead(torch.nn.Module):
    """
    Consumes feature map from residual blocks and returns a scaler score representing how good the
    current board position is.
    """

    def __init__(self, num_in: int, num_out: int, num_feats: int, dropout: float):
        super().__init__()
        # TODO: Try kernel size 1 here with padding="same"
        self.conv1 = torch.nn.Conv2d(num_in, num_feats, kernel_size=3, padding=1)
        self.batch_norm_1 = torch.nn.BatchNorm2d(num_feats)
        self.relu_1 = torch.nn.ReLU(inplace=False)
        self.flatten_2 = torch.nn.Flatten()
        self.linear_2 = torch.nn.Linear(9 * num_feats, 2 * num_feats)
        self.batch_norm_2 = torch.nn.BatchNorm1d(2 * num_feats)
        self.relu_2 = torch.nn.ReLU(inplace=False)
        self.dropout_2 = torch.nn.Dropout(dropout)
        self.linear_3 = torch.nn.Linear(2 * num_feats, num_out)
        self.tanh_3 = torch.nn.Tanh()

    def forward(self, x):
        # Feature extraction.
        x = self.conv1(x)
        x = self.batch_norm_1(x)
        x = self.relu_1(x)
        # Feature expansion.
        x = self.flatten_2(x)
        x = self.linear_2(x)
        x = self.batch_norm_2(x)
        x = self.relu_2(x)
        x = self.dropout_2(x)
        # Map compressed features to scalar score in [-1, 1].
        x = self.linear_3(x)
        x = self.tanh_3(x)
        return x


class TicTacToeNet(torch.nn.Module):
    """
    Residual tower consisting of an initial convolution block, then several residual blocks. The
    output of this is passed to 2 separate heads which calculate the policy and state value.
    """

    def __init__(self, hparams: dict):
        super().__init__()
        self.action_size = 9
        # Shared initial convolution block.
        self.conv_block = ConvBlock(num_in=3, num_out=hparams["hidden"])
        # Shared core res-net block.
        self.res_blocks = torch.nn.ModuleList(
            [ResBlock(hparams["hidden"], hparams["hidden"]) for _ in range(hparams["res_blocks"])]
        )
        # Separate policy prediction subnetwork.
        self.policy_head = PolicyHead(
            num_in=hparams["hidden"],
            num_out=self.action_size,
            num_feats=hparams["pol_feats"],
            dropout=hparams["pol_dropout"],
        )
        # Separate value prediction subnetwork.
        self.value_head = ValueHead(
            num_in=hparams["hidden"],
            num_out=1,
            num_feats=hparams["val_feats"],
            dropout=hparams["val_dropout"],
        )
        # Send to device and compile for faster training.
        self = torch.compile(self)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

    def parse_output(self, policy: torch.Tensor, value: torch.Tensor) -> tuple[torch.Tensor, float]:
        """
        Transform the raw policy and value output into actual probabilities.
        """
        if policy.get_device != -1:
            parsed_policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()
        else:
            raise NotImplementedError("Parsing for GPU not implemented yet.")
        parsed_value = value.item()
        return parsed_policy, parsed_value

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        x = self.conv_block(x)
        for res_block in self.res_blocks:
            x = res_block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value

    def get_pol_param_grp(self) -> list[torch.Tensor]:
        return list(self.policy_head.parameters())

    def get_val_param_grp(self) -> list[torch.Tensor]:
        return list(self.value_head.parameters())

    def get_oth_param_grp(self) -> list[torch.Tensor]:
        oth_params = []
        for name, param in self.named_parameters():
            if "policy_head" not in name and "value_head" not in name:
                oth_params.append(param)
        return oth_params
