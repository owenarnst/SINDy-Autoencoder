from typing import Callable, List
import torch
import torch.nn.functional as F


class LinearLayer(torch.nn.Linear):
    """Custom implementation of layer which includes the activation function
    and its derivative
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        activation_function: torch.nn.Module,
        last: bool = False,
        order: int = 1,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:

        # Constructor for a regular linear layer
        super().__init__(in_features, out_features, bias, device, dtype=torch.float64)

        # Our modifications to the linear layer

        # Store the activation function
        self.activation_function = activation_function

        # Store the activation function's first derivative
        self.activation_derivative = self.__get_activation_derivative()

        # Store the activation function's second derivative
        self.activation_2nd_derivative = self.__get_activation_2nd_derivative()

        # Store the most recently computed
        self.last = last
        if order == 1:
            self.forward = self.forward_dx
        else:
            self.forward = self.forward_ddx

    def forward_dx(self, input: torch.Tensor) -> torch.Tensor:

        print(input)
        x, dx = input[: input.shape[0] // 2], input[input.shape[0] // 2 :]
        dim = self.weight.shape[0]

        if self.last:
            x = F.linear(x, self.weight, self.bias)
            dx = F.linear(dx, self.weight, torch.zeros_like(self.bias))
        else:
            x = F.linear(x, self.weight, self.bias)
            dx = self.activation_derivative(x) * F.linear(
                dx, self.weight, torch.zeros_like(self.bias)
            )
            x = self.activation_function(x)

        return torch.cat((x, dx), dim=0)

    def forward_ddx(self, input: torch.Tensor) -> torch.Tensor:

        slicer = input.shape[0] // 3
        x, dx, ddx = input[:slicer], input[slicer : 2 * slicer], input[2 * slicer :]

        if self.last:
            x = F.linear(x, self.weight, self.bias)
            dx = F.linear(dx, self.weight, torch.zeros_like(self.bias))
            ddx = F.linear(ddx, self.weight, torch.zeros_like(self.bias))
        else:
            x = F.linear(x, self.weight, self.bias)
            dx_ = F.linear(dx, self.weight, torch.zeros_like(self.bias))
            ddx_ = F.linear(ddx, self.weight, torch.zeros_like(self.bias))

            dactivation = self.activation_derivative(x)
            ddactivation = self.activation_2nd_derivative(x, dactivation)

            dx = dactivation * dx_
            ddx = ddactivation * dx_ + dactivation * ddx_

            x = self.activation_function(x)

        return torch.cat((x, dx, ddx), dim=0)

    def __get_activation_derivative(self) -> Callable:
        if isinstance(self.activation_function, torch.nn.ReLU):
            return lambda x: torch.where(x > 0, torch.ones_like(x), torch.zeros_like(x))
        if isinstance(self.activation_function, torch.nn.ELU):
            return lambda x: torch.minimum(x, torch.exp(x))
        if isinstance(self.activation_function, torch.nn.Sigmoid):
            return lambda x: self.dsigmoid(x)

    def __get_activation_2nd_derivative(self) -> Callable:
        if isinstance(self.activation_function, torch.nn.ReLU):
            return lambda x, dx=0: 0
        if isinstance(self.activation_function, torch.nn.ELU):
            return lambda x, dx=0: torch.where(x > 0, torch.exp(x), torch.zeros_like(x))
        if isinstance(self.activation_function, torch.nn.Sigmoid):
            return lambda x, dx=0: self.dsigmoid(dx)

    def dsigmoid(self, x):
        sigmoid = self.activation_function(x)
        return sigmoid * (1 - sigmoid)


# if __name__ == "__main__":
#     linear = LinearLayer(2,1, torch.nn.ReLU(), last=True, order=2)
#     out = linear([torch.ones(2), torch.ones(2), torch.ones(2)])
#     print(out)
