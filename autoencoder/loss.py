import torch


class Loss(torch.nn.Module):
    def __init__(
        self,
        lambda_1: float,
        lambda_2: float,
        lambda_3: float,
        lambda_r: float,
        order: int = 1,
        *args,
        **kwargs
    ) -> None:
        """Custom loss fucnction based on multiple MSEs

        Args:
            lambda_1 (float): loss weight decoder
            lambda_2 (float): loss weight sindy z
            lambda_3 (float): loss weight sindy x
            lambda_r (float): loss weight sindy regularization
            order (int, optional): Order of the model can be 1 or 2. Defaults to 1.
        """
        super().__init__(*args, **kwargs)
        self.lambda_1 = lambda_1
        self.lambda_2 = lambda_2
        self.lambda_3 = lambda_3
        self.lambda_r = lambda_r

        self.regularization = True

        if order == 1:
            self.forward = self.forward_dx
        else:
            self.forward = self.forward_ddx

    def forward_dx(
        self,
        x,
        dx,
        dz,
        dz_pred,
        x_decode,
        dx_decode,
        sindy_coeffs: torch.Tensor,
        coeff_mask,
    ) -> torch.Tensor:

        loss = 0

        loss += self.lambda_1 * torch.mean((x - x_decode) ** 2)
        loss += self.lambda_2 * torch.mean((dz - dz_pred) ** 2)
        loss += self.lambda_3 * torch.mean((dx - dx_decode) ** 2)
        loss += (
            int(self.regularization)
            * self.lambda_r
            * torch.mean(torch.abs(sindy_coeffs) * coeff_mask)
        )

        return loss

    def forward_ddx(
        self,
        x,
        dx,
        dz,
        dz_pred,
        x_decode,
        dx_decode,
        sindy_coeffs: torch.Tensor,
        ddz,
        ddx,
        ddx_decode,
        coeff_mask,
    ) -> torch.Tensor:

        loss = 0

        loss += self.lambda_1 * torch.mean((x - x_decode) ** 2)
        # dz_pred is in this case ddz_pred
        loss += self.lambda_2 * torch.mean((ddz - dz_pred) ** 2)
        loss += self.lambda_3 * torch.mean((ddx - ddx_decode) ** 2)
        loss += (
            int(self.regularization)
            * self.lambda_r
            * torch.mean(torch.abs(sindy_coeffs) * coeff_mask)
        )

        return loss

    def set_regularization(self, include_regularization: bool) -> None:

        self.regularization = include_regularization


# if __name__ == "__main__":
#     loss = Loss(1,1,1,1)

#     X = torch.ones((2, 2))
#     dZ = torch.zeros((2, 1))
#     X_decode = torch.ones((2, 2))
#     sindy = torch.ones((1, 3))

#     print(loss(X, dZ, X_decode, sindy))
