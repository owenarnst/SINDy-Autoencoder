import torch
from .linear import LinearLayer


class AutoEncoder(torch.nn.Module):

    RELU = "relu"
    SIGMOID = "sigmoid"
    ELU = "elu"

    def __init__(
        self, params: dict = {}, name: str = "encoder", *args, **kwargs
    ) -> None:

        super().__init__(*args, **kwargs)
        self.params = params

        activation = self.params["activation"]
        self.activation_function = self.__get_activation(activation)

        self.weights = (
            [self.params["input_dim"]]
            + self.params["widths"]
            + [self.params["latent_dim"]]
        )
        self.order = self.params["model_order"]

        if self.weights is None:
            raise TypeError("Missing weight param")

        if name == "encoder":
            self.__create_encoder()
        elif name == "decoder":
            self.__create_decoder()

    def __create_encoder(self) -> None:
        """Creates the encoder based on weights and activation function"""
        layers = []
        for curr_weights, next_weights in zip(self.weights[:-1], self.weights[1:]):
            layers.append(
                LinearLayer(
                    curr_weights,
                    next_weights,
                    self.activation_function,
                    len(layers) + 2 == len(self.weights),
                    self.order,
                )
            )
        self.net = torch.nn.Sequential(*layers)

    def __create_decoder(self) -> None:
        """Creates decoder, the weights are swapped and reversed compared to the encoder"""
        layers = []
        for curr_weights, next_weights in zip(
            reversed(self.weights[1:]), reversed(self.weights[:-1])
        ):
            layers.append(
                LinearLayer(
                    curr_weights,
                    next_weights,
                    self.activation_function,
                    len(layers) + 2 == len(self.weights),
                    self.order,
                )
            )

        self.net = torch.nn.Sequential(*layers)

    def __get_activation(self, activation: str = "relu") -> torch.nn.Module:
        match (activation):
            case self.RELU:
                return torch.nn.ReLU()
            case self.SIGMOID:
                return torch.nn.Sigmoid()
            case self.ELU:
                return torch.nn.ELU()
            case _:
                raise TypeError(f"Invalid activation function {activation}")

    def forward(self, x: torch.Tensor) -> list[torch.Tensor]:
        """Forward function of the autoencoder

        Args:
            x (List[Tensor]): either the List has 2 or 3 elements
            if it has 2 elements the model order has to be set to 1
            if it has 3 elements the model order has to be set to 2

        Returns:
            List[torch.Tensor]: returns the forward passed list whit the same number of elements as the input
        """
        return self.net(x)
